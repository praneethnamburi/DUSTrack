"""
Sparse-blip outlier detection + per-label LK-RSTC interpolation.

A *blip* is a short run of frames in a per-label position trace where the
labeled point jumps away and then returns — the signature of a model
picking the visually-strongest answer for one or a few frames before
snapping back to the temporally-consistent lane. This module finds those
runs (per label, with a robust adaptive threshold) and re-tracks them
with the existing LK-RSTC machinery anchored to the surrounding good
frames. The output is a sparse :class:`VideoAnnotation` containing only
the corrected frames for the blipped labels, suitable as additional
training data via DLC's NaN-tolerant labeled-data pipeline.

The lever: training on these LK-anchored answers nudges the model toward
the temporally-consistent answer over the visually-strongest one, which
is itself a mitigation for bistability in non-adjacent-labeling regimes
(the pia02 motion-rich regime; see ``research/programs/wobble-arc.md``
and the *Roadmap → Next (general-model workflow features)* section of
``specs/dustrack.md``).

Detection is pure numpy and runs without decoding the video. Per blip,
interpolation calls :func:`dustrack.lk_opticalflow.lucas_kanade_rstc`
on the (s-1, e+1) bracket; the middle (e - s + 1) frames of the returned
RSTC path are the corrections.

Reference for the LK-RSTC algorithm:
    Magana-Salgado, U., Namburi, P., Feigin-Almon, M., Pallares-Lopez, R.,
    & Anthony, B. (2023). A comparison of point-tracking algorithms in
    ultrasound videos from the upper limb. BioMedical Engineering OnLine,
    22(1), 52.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .annotations import VideoAnnotation
from .lk_opticalflow import lucas_kanade_rstc


# Scale factor that makes the median absolute deviation a consistent
# estimator of the standard deviation under a normal distribution
# (``sigma_hat = 1.4826 * MAD``). Letting callers pass ``threshold_factor``
# in units of sigma keeps the knob interpretable.
_MAD_TO_SIGMA = 1.4826

# MAD values below this are treated as zero (flat motion); detection
# falls back to a high percentile of the displacement distribution so a
# legitimately-zero-motion label can still flag a spike.
_MAD_ZERO_EPS = 1e-6

# Tolerance for the RSTC endpoint sanity check. RSTC sigmoid weights at
# the endpoints are ~ 1 (within ``epsilon`` floor); the blended path
# should match the input anchors within numerical noise.
_RSTC_ENDPOINT_TOL = 1e-3


@dataclass
class Blip:
    """A single detected outlier run for one label.

    Attributes:
        label: The label key (string digit, matching :attr:`VideoAnnotation.labels`).
        start: First blip frame, inclusive.
        end: Last blip frame, inclusive. ``start == end`` for a single-frame blip.
        anchor_before: ``(x, y)`` at frame ``start - 1`` (the last-known-good).
        anchor_after: ``(x, y)`` at frame ``end + 1`` (the first-known-good after the run).
        threshold: The per-label displacement threshold that flagged the entry.
    """
    label: str
    start: int
    end: int
    anchor_before: tuple[float, float]
    anchor_after: tuple[float, float]
    threshold: float

    @property
    def length(self) -> int:
        """Number of blip frames (inclusive on both ends)."""
        return self.end - self.start + 1


@dataclass
class BlipReport:
    """Detection result for one :class:`VideoAnnotation`.

    Attributes:
        blips: All detected blips across all labels, in label-then-frame order.
        per_label_stats: Per-label diagnostic dict. Keys: ``median_d``, ``mad_d``,
            ``threshold``, ``n_blips``, ``n_skipped_edge`` (spikes at frame 0 /
            last frame that can't be anchor-bracketed), ``n_skipped_long`` (runs
            longer than ``max_blip_length``), ``n_skipped_noreturn`` (spikes
            whose forward scan never finds a return-position match).
        params: The detection knobs used (echoed for traceability).
    """
    blips: list[Blip] = field(default_factory=list)
    per_label_stats: dict[str, dict[str, float | int]] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.blips)

    def by_label(self) -> Mapping[str, list[Blip]]:
        """Group blips by label."""
        ret: dict[str, list[Blip]] = {}
        for b in self.blips:
            ret.setdefault(b.label, []).append(b)
        return ret

    def length_histogram(self) -> dict[int, int]:
        """Count blips by run length."""
        hist: dict[int, int] = {}
        for b in self.blips:
            hist[b.length] = hist.get(b.length, 0) + 1
        return dict(sorted(hist.items()))


def _label_threshold(
    d_finite: np.ndarray, threshold_factor: float
) -> tuple[float, float, float]:
    """Compute (median, mad, threshold) for one label's displacement series.

    Three regimes:
        1. Normal: ``mad > eps`` -> ``threshold = med + factor * 1.4826 * mad``.
        2. Flat-but-spiky: ``mad ~ 0`` but ``range(d) > 0`` (e.g. still
           trace with one outlier spike). Fall back to a midpoint between
           the median and the maximum so the spike clears the threshold
           while non-spike frames sit comfortably below.
        3. Fully constant: ``mad ~ 0 AND range(d) ~ 0`` (e.g. linear
           xy(t) with no jitter). No spikes possible; set
           ``threshold = inf`` so nothing flags.
    """
    if d_finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    med = float(np.median(d_finite))
    mad = float(np.median(np.abs(d_finite - med)))
    if mad < _MAD_ZERO_EPS:
        d_range = float(d_finite.max() - d_finite.min())
        if d_range < _MAD_ZERO_EPS:
            # Fully constant: no variation at all -> nothing to flag.
            threshold = float("inf")
        else:
            # Flat with spikes: midpoint between the median and the
            # max gives every above-median spike a clear flag while
            # leaving the bulk untouched.
            threshold = 0.5 * (med + float(d_finite.max()))
    else:
        threshold = med + threshold_factor * _MAD_TO_SIGMA * mad
    return med, mad, threshold


def _find_blips_in_label(
    xy: np.ndarray,
    d: np.ndarray,
    threshold: float,
    median_d: float,
    *,
    max_blip_length: int,
    return_position_factor: float,
) -> tuple[list[tuple[int, int]], int, int, int]:
    """Bracket blip runs in one label's dense ``(n_frames, 2)`` trace.

    Returns:
        ``(blips, n_skipped_edge, n_skipped_long, n_skipped_noreturn)``
        where ``blips`` is a list of ``(start, end)`` index pairs (both
        inclusive, blip-frame indices into ``xy``).

    Algorithm:
        Walk ``d`` left-to-right. ``d[i]`` is the displacement between
        ``xy[i]`` and ``xy[i+1]``. A blip entry at frame ``s`` requires
        ``d[s-1] > threshold`` (entry from a clean frame). After
        ``max_blip_length`` candidate end frames, give up; otherwise the
        first frame ``e`` whose successor position ``xy[e+1]`` lands
        within ``return_position_factor * median_d * (e - s + 2)`` of
        the anchor ``xy[s-1]`` closes the blip as ``[s, e]``.

    Edge handling:
        - A spike whose entry is the very first displacement (``d[0]``,
          so ``s == 1`` and anchor frame 0 is fine) is anchored normally.
        - A spike whose entry is the last displacement
          (``d[n_frames - 2]``, anchor-after frame ``n_frames`` doesn't
          exist) cannot be bracketed.
        - Either anchor being NaN disqualifies the blip.
        - The detection is per-label-frame; partial coverage (NaN gaps
          in ``xy``) means the displacement series has NaNs which are
          filtered out of the threshold computation and never compared
          against ``threshold`` (a NaN-anchored blip cannot be bracketed
          since the boundary check below requires finite anchors).
    """
    n_frames = xy.shape[0]
    blips: list[tuple[int, int]] = []
    n_skipped_edge = 0
    n_skipped_long = 0
    n_skipped_noreturn = 0

    # Outcome codes returned by the per-spike inner scan.
    OUTCOME_FOUND = "found"
    OUTCOME_LONG = "long"           # ran out of candidates without a return
    OUTCOME_EDGE = "edge"           # anchor-after would be past the end of the trace
    OUTCOME_NORETURN = "noreturn"   # anchor-after was NaN (unlabeled)

    def _scan_for_return(s: int) -> tuple[str, int]:
        """Scan forward from blip start ``s`` for a return-position anchor.

        Returns ``(outcome, e)`` where ``e`` is the bracketed end frame
        (only meaningful when outcome == OUTCOME_FOUND).
        """
        anchor_before_local = xy[s - 1]
        for e in range(s, s + max_blip_length):
            if e + 1 >= n_frames:
                return OUTCOME_EDGE, -1
            anchor_after_local = xy[e + 1]
            if not np.isfinite(anchor_after_local).all():
                return OUTCOME_NORETURN, -1
            run_len = e - s + 1
            tol = return_position_factor * median_d * (run_len + 1)
            if np.linalg.norm(anchor_after_local - anchor_before_local) <= tol:
                return OUTCOME_FOUND, e
        return OUTCOME_LONG, -1

    i = 0
    while i < d.shape[0]:
        di = d[i]
        if not np.isfinite(di) or di <= threshold:
            i += 1
            continue

        # ``d[i]`` is the displacement xy[i] -> xy[i+1]. A blip ENTRY
        # means ``xy[i+1]`` is the first off-trajectory frame, so the
        # blip start is ``s = i + 1`` and the pre-blip anchor is
        # ``xy[i]``.
        s = i + 1
        anchor_before = xy[s - 1]
        if not np.isfinite(anchor_before).all():
            i += 1
            continue

        outcome, e = _scan_for_return(s)
        if outcome == OUTCOME_FOUND:
            blips.append((s, e))
            # Skip past the blip + its return frame so we don't
            # re-flag the return displacement as a new entry.
            i = e + 1
            continue
        if outcome == OUTCOME_EDGE:
            n_skipped_edge += 1
        elif outcome == OUTCOME_LONG:
            n_skipped_long += 1
        else:  # OUTCOME_NORETURN
            n_skipped_noreturn += 1
        i += 1

    return blips, n_skipped_edge, n_skipped_long, n_skipped_noreturn


def detect_blips(
    ann: VideoAnnotation,
    *,
    threshold_factor: float = 5.0,
    max_blip_length: int = 5,
    return_position_factor: float = 3.0,
) -> BlipReport:
    """Find sparse blip runs in a dense-trace annotation, per label.

    Pure-Python: does not decode the video.

    Args:
        ann: A :class:`VideoAnnotation` whose ``data[label]`` is densely
            populated across frames (typically a DLC ``.h5`` predicted
            trace; partial coverage is tolerated — NaN frames are skipped).
        threshold_factor: Per-label entry threshold in units of robust
            sigma (``1.4826 * MAD``) above the median displacement.
            Default 5.0 is conservative; lower flags more candidates.
        max_blip_length: Maximum run length (in frames) the bracketing
            scan will consider before giving up on a spike. Long runs
            are usually real model failures, not blips.
        return_position_factor: Multiplier on the per-blip return-tolerance
            ``factor * median_d * (run_len + 1)``. Default 3.0 allows the
            return anchor to land within ~3 typical-per-frame displacements
            per run-frame of the pre-blip anchor.

    Returns:
        :class:`BlipReport`.
    """
    params = dict(
        threshold_factor=threshold_factor,
        max_blip_length=max_blip_length,
        return_position_factor=return_position_factor,
    )
    report = BlipReport(params=params)

    for label in ann.labels:
        xy = ann.to_trace(label)  # (n_frames, 2) with NaN for missing
        # Per-frame displacement magnitudes. NaN where either endpoint is
        # missing; filtered out of the threshold computation.
        diff = np.diff(xy, axis=0)
        d = np.linalg.norm(diff, axis=1)
        d_finite = d[np.isfinite(d)]

        med, mad, threshold = _label_threshold(d_finite, threshold_factor)
        stats: dict[str, float | int] = {
            "median_d": med,
            "mad_d": mad,
            "threshold": threshold,
            "n_blips": 0,
            "n_skipped_edge": 0,
            "n_skipped_long": 0,
            "n_skipped_noreturn": 0,
            "n_finite_frames": int(np.isfinite(xy).all(axis=1).sum()),
        }

        if not np.isfinite(threshold) or d_finite.size == 0:
            report.per_label_stats[label] = stats
            continue

        ranges, n_edge, n_long, n_noret = _find_blips_in_label(
            xy,
            d,
            threshold=threshold,
            median_d=med,
            max_blip_length=max_blip_length,
            return_position_factor=return_position_factor,
        )
        stats["n_blips"] = len(ranges)
        stats["n_skipped_edge"] = n_edge
        stats["n_skipped_long"] = n_long
        stats["n_skipped_noreturn"] = n_noret
        report.per_label_stats[label] = stats

        for s, e in ranges:
            report.blips.append(
                Blip(
                    label=label,
                    start=s,
                    end=e,
                    anchor_before=(float(xy[s - 1, 0]), float(xy[s - 1, 1])),
                    anchor_after=(float(xy[e + 1, 0]), float(xy[e + 1, 1])),
                    threshold=float(threshold),
                )
            )

    return report


def interpolate_blips(
    ann: VideoAnnotation,
    report: BlipReport,
    *,
    lk_config: dict | None = None,
) -> VideoAnnotation:
    """LK-RSTC re-track every blip and return a sparse corrections layer.

    The returned :class:`VideoAnnotation` shares the source video and
    label set, but ``data[label]`` is populated only at blip frames for
    blipped labels (other labels expose empty ``{}``). Suitable for
    saving as a new annotation layer and consuming as DLC training
    data via the standard NaN-tolerant labeled-data path.

    Args:
        ann: Source dense-trace annotation (the one passed to
            :func:`detect_blips`). Its ``video`` is reused for decode.
        report: Detection result from :func:`detect_blips`.
        lk_config: Optional override for the per-pair LK config (passed
            through to :func:`dustrack.lk_opticalflow.lucas_kanade_rstc`).

    Returns:
        A sparse :class:`VideoAnnotation` with corrections at blip
        frames. The ``fname`` field is set to
        ``<source_stem>_blip_corrections.json`` next to the source; the
        file is NOT written by this function (call ``.save()`` on the
        returned object if you want it persisted).
    """
    lk_kwargs = dict(lk_config) if lk_config else {}

    video = ann.video
    if video is None:
        raise ValueError(
            "interpolate_blips requires a video reader on the source "
            "annotation (ann.video is None)."
        )

    # Build the empty sparse output. One per-label key for every label
    # in the source so consumers can iterate ``out.labels`` and see
    # the same shape; only blipped labels get populated frames.
    sparse: dict[str, dict[int, list[float]]] = {label: {} for label in ann.labels}

    for blip in report.blips:
        start_pts = np.asarray([blip.anchor_before], dtype=np.float32)
        end_pts = np.asarray([blip.anchor_after], dtype=np.float32)
        path = lucas_kanade_rstc(
            video,
            blip.start - 1,
            blip.end + 1,
            start_pts,
            end_pts,
            **lk_kwargs,
        )
        # ``path`` is (e - s + 3, 1, 2). The first and last rows match
        # the input anchors (RSTC sigmoid -> 1 at endpoints); the
        # middle (e - s + 1) rows are the corrected positions for
        # frames [s, e].
        assert path.shape == (blip.end - blip.start + 3, 1, 2), (
            f"Unexpected RSTC path shape {path.shape} for blip {blip}"
        )
        start_err = float(np.linalg.norm(path[0, 0] - np.asarray(blip.anchor_before)))
        end_err = float(np.linalg.norm(path[-1, 0] - np.asarray(blip.anchor_after)))
        if start_err > _RSTC_ENDPOINT_TOL or end_err > _RSTC_ENDPOINT_TOL:
            # Surface but don't fail; RSTC anchor-pin should always hold
            # within numerical noise but a future LK regression should
            # at least be visible.
            print(
                f"[detect_blips] WARNING: RSTC endpoint drift for blip "
                f"{blip!r}: start_err={start_err:.4f} end_err={end_err:.4f}"
            )

        corrections = path[1:-1, 0, :]  # (e - s + 1, 2)
        for offset, xy in enumerate(corrections):
            frame = blip.start + offset
            sparse[blip.label][frame] = [float(xy[0]), float(xy[1])]

    # Construct the sparse output annotation. Pre-build the per-label
    # dict and pass via ``preloaded_json`` so ``VideoAnnotation.__init__``
    # doesn't try to read the (nonexistent) target file from disk.
    # Pass ``vname=None`` and supply the reader via ``video=`` so the
    # ``_parse_inp`` ``utils.is_video`` assert (which probes the
    # filesystem) is bypassed -- the video identity is preserved
    # through the reader instance, not the filename string.
    out_fname = _corrections_fname(ann)
    out = VideoAnnotation(
        fname=out_fname,
        vname=None,
        n_labels=len(ann.labels),
        preloaded_json=sparse,
        video=video,  # share the reader; avoid a second av.open
    )
    return out


def detect_and_interpolate_blips(
    ann: VideoAnnotation,
    *,
    save: bool = True,
    threshold_factor: float = 5.0,
    max_blip_length: int = 5,
    return_position_factor: float = 3.0,
    lk_config: dict | None = None,
) -> tuple[VideoAnnotation, BlipReport]:
    """Convenience wrapper: detect, interpolate, optionally save.

    Args:
        ann: Source dense-trace annotation.
        save: If True (default), write the sparse output to
            ``<source_stem>_blip_corrections.json`` next to the source.
            Refuses to overwrite an existing file (raises
            :class:`FileExistsError`); delete or rename the existing
            file first.
        threshold_factor: See :func:`detect_blips`.
        max_blip_length: See :func:`detect_blips`.
        return_position_factor: See :func:`detect_blips`.
        lk_config: See :func:`interpolate_blips`.

    Returns:
        ``(sparse_annotation, report)``.

    Raises:
        FileExistsError: If ``save=True`` and the output file already
            exists. Per [[benchmark-cleanup-collision]]: never silently
            overwrite a user file.
    """
    report = detect_blips(
        ann,
        threshold_factor=threshold_factor,
        max_blip_length=max_blip_length,
        return_position_factor=return_position_factor,
    )
    out = interpolate_blips(ann, report, lk_config=lk_config)
    if save:
        if os.path.exists(out.fname):
            raise FileExistsError(
                f"Output file already exists at {out.fname}. "
                f"Refusing to overwrite; delete or rename it first."
            )
        out.save()
    return out, report


def _corrections_fname(ann: VideoAnnotation) -> str:
    """Derive the sparse-corrections output path from a source annotation.

    ``<source_stem>_blip_corrections.json`` next to the source file.
    """
    if ann.fname is None:
        raise ValueError("Source annotation has no fname; cannot derive output path.")
    p = Path(ann.fname)
    return str(p.parent / f"{p.stem}_blip_corrections.json")
