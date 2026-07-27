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
from typing import Any, Callable, Mapping

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
    n_frames: int = 0  # total frames in the source annotation

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

    def min_coverage(self) -> float:
        """Smallest per-label fraction of finite frames in the source annotation.

        ``min_coverage() == 1.0`` means every label is densely populated
        across all frames (typical DLC predicted trace); lower values
        mean at least one label has gaps. Used by the UI gate to decide
        whether the active layer is dense enough for blip detection to
        be meaningful (sparse manual layers can be scanned, but the
        per-label MAD threshold isn't well-conditioned on a handful of
        points).

        Returns ``0.0`` when there are no labels, no per-label stats
        recorded, or ``n_frames == 0``.
        """
        if not self.per_label_stats or self.n_frames <= 0:
            return 0.0
        ratios = [
            int(stats.get("n_finite_frames", 0)) / self.n_frames
            for stats in self.per_label_stats.values()
        ]
        return float(min(ratios)) if ratios else 0.0

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

def interpolate_blips(
    ann: VideoAnnotation,
    report: BlipReport,
    *,
    lk_config: dict | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> VideoAnnotation:
    """LK-RSTC re-track every blip and return a sparse corrections layer.

    The returned :class:`VideoAnnotation` shares the source video and
    label set, but ``data[label]`` is populated only at blip frames for
    blipped labels (other labels expose empty ``{}``). Suitable for
    saving as a new annotation layer and consuming as DLC training
    data via the standard NaN-tolerant labeled-data path.

    Args:
        ann: Source dense-trace annotation (the one passed to
            :func:`disagreement_blips`). Its ``video`` is reused for decode.
        report: Detection result from :func:`disagreement_blips`.
        lk_config: Optional override for the per-pair LK config (passed
            through to :func:`dustrack.lk_opticalflow.lucas_kanade_rstc`).
        progress_callback: Optional ``(done_count, total_count)``
            callback fired once per completed blip. Used by the UI
            ProgressOverlay to drive a progress bar; headless callers
            can leave it ``None``. Fires after the per-blip LK has
            completed and the corrections are written into the sparse
            output dict.

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

    total = len(report.blips)
    for done_count, blip in enumerate(report.blips, start=1):
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
                f"[interpolate_blips] WARNING: RSTC endpoint drift for blip "
                f"{blip!r}: start_err={start_err:.4f} end_err={end_err:.4f}"
            )

        corrections = path[1:-1, 0, :]  # (e - s + 1, 2)
        for offset, xy in enumerate(corrections):
            frame = blip.start + offset
            sparse[blip.label][frame] = [float(xy[0]), float(xy[1])]

        if progress_callback is not None:
            progress_callback(done_count, total)

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

def _corrections_fname(ann: VideoAnnotation) -> str:
    """Derive the sparse-corrections output path from a source annotation.

    ``<source_stem>_blip_corrections.json`` next to the source file.
    """
    if ann.fname is None:
        raise ValueError("Source annotation has no fname; cannot derive output path.")
    p = Path(ann.fname)
    return str(p.parent / f"{p.stem}_blip_corrections.json")

# --------------------------------------------------------------------- #
# Flow-based blip detection                                             #
# --------------------------------------------------------------------- #
# FlowBlipResult is the mining-result container (corrections + residual) the
# autorefine curriculum ranks over. Produced now by the unified disagreement
# detector (below), consumed by mithic's select_flow_blips.

@dataclass
class FlowBlipResult:
    """Confidently-wrong frames found by model-vs-flow disagreement.

    ``corrections`` is the flow's answer at each kept frame (the label to
    train on); ``residual`` is the model-vs-flow distance in px, the key to
    rank by -- the biggest errors are the most valuable training signal.
    Both are ``{label: {frame: value}}``.
    """

    corrections: dict
    residual: dict
    n_screened: int = 0
    params: dict = field(default_factory=dict)

    def n_kept(self) -> int:
        return sum(len(v) for v in self.corrections.values())

def low_confidence_frames(likelihood, threshold, *, max_frames=None):
    """Frame indices whose *worst* (min over points) model likelihood is below
    ``threshold`` -- the frames the model is least sure of, for adding to
    training. Ranked most-uncertain first; capped to ``max_frames`` if given.

    The complement of :func:`disagreement_blips`: low confidence surfaces where the
    model *knows* it's unsure (LOST), blips surface where it's confidently
    wrong. ``likelihood`` is ``(N,)`` or ``(N, P)``.
    """
    lk = np.asarray(likelihood, dtype=float)
    if lk.ndim == 1:
        lk = lk[:, None]
    worst = lk.min(axis=1)
    idx = np.where(worst < float(threshold))[0]
    idx = idx[np.argsort(worst[idx], kind="stable")]      # most uncertain first
    if max_frames is not None:
        idx = idx[: int(max_frames)]
    return [int(i) for i in idx]

# --------------------------------------------------------------------------- #
# Unified blip detector: LK-from-previous-frame vs DLC disagreement, 5-sigma.  #
# The ONE blip definition -- reused by the GUI "Interpolate blips" button and  #
# the autorefine curriculum, which layers confidence/agreement/budget on top.  #
# --------------------------------------------------------------------------- #

def _runs_from_mask(mask, max_gap: int = 0):
    """Contiguous ``True`` runs of a boolean mask as ``(start, end)`` inclusive
    index pairs. Runs separated by a gap of ``<= max_gap`` ``False`` frames are
    merged (so a one-frame recovery inside a disturbed stretch doesn't split
    the run the interpolation should span)."""
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return []
    runs, s, prev = [], int(idx[0]), int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev - 1 > max_gap:
            runs.append((s, prev))
            s = i
        prev = i
    runs.append((s, prev))
    return runs

def compute_lk_predictions(positions, video, *, lk_config=None,
                           progress_callback=None):
    """Per-frame LK-from-the-previous-frame prediction of each DLC point.

    For every frame ``m``, seed LK at the model's *previous*-frame prediction
    ``positions[m-1]`` and track it forward ``m-1 -> m``; the landing is the
    flow's answer for where ``m`` should be, independent of the model. The
    disagreement ``|positions[m] - lk_pred[m]|`` is the single blip signal (see
    :func:`disagreement_blips`).

    This is the quantity DUSTrack saves alongside a DLC ``analyze_videos`` run
    -- computed on the frames inference already decodes -- so blip detection is
    later a cheap array op with no second decode. Frame 0 (and any frame with a
    non-finite previous prediction) is left NaN.

    Returns ``lk_pred`` of shape ``(N, P, 2)``, index-aligned to ``positions``.
    """
    from datanavigator.video_reader import VideoReader
    from dustrack.lk_opticalflow import _gray_rgb, _lk_track_frames

    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError("positions must be (N, P, 2)")
    n_frames = positions.shape[0]
    if isinstance(video, (str, Path)):
        video = VideoReader(str(video))
    cfg = dict(lk_config or {})

    lk_pred = np.full_like(positions, np.nan)
    prev = _gray_rgb(video, 0)
    for m in range(1, n_frames):
        cur = _gray_rgb(video, m)
        seed = positions[m - 1]
        if np.isfinite(seed).all():
            lk_pred[m] = _lk_track_frames([prev, cur], seed.astype(np.float32))[-1]
        prev = cur
        if progress_callback is not None and m % 500 == 0:
            progress_callback(m, n_frames)
    return lk_pred

def disagreement_blips(positions, lk_pred, *, labels=None,
                       threshold_factor: float = 5.0, max_gap: int = 1,
                       max_blip_length=None) -> BlipReport:
    """The one blip detector: LK-vs-DLC disagreement, thresholded per label at
    ``median + threshold_factor * 1.4826 * MAD`` (robust 5-sigma), bracketed
    into runs with good-frame anchors.

    Adaptive by construction -- the threshold sits above each point's own
    disagreement floor, so a point can move (or wobble a few px) without
    blipping; only anomalous disagreement flags. Returns a :class:`BlipReport`
    interchangeable with a hand-built one, so :func:`interpolate_blips`
    consumes it unchanged.

    Args:
        positions: ``(N, P, 2)`` model predictions.
        lk_pred: ``(N, P, 2)`` LK-from-previous predictions
            (:func:`compute_lk_predictions`), index-aligned to ``positions``.
        labels: point names (default ``"0".."P-1"``).
        threshold_factor: robust-sigma multiplier (default 5).
        max_gap: merge runs separated by <= this many sub-threshold frames.
        max_blip_length: drop runs longer than this (a long run is a sustained
            failure, not a blip); ``None`` keeps all.
    """
    positions = np.asarray(positions, dtype=float)
    lk_pred = np.asarray(lk_pred, dtype=float)
    if positions.shape != lk_pred.shape:
        raise ValueError("positions and lk_pred must have the same shape")
    n_frames, n_pts = positions.shape[:2]
    labels = list(labels) if labels is not None else [str(i) for i in range(n_pts)]

    disagreement = np.linalg.norm(positions - lk_pred, axis=2)      # (N, P)
    blips: list[Blip] = []
    stats: dict[str, dict] = {}
    for p, lab in enumerate(labels):
        d = disagreement[:, p]
        finite = d[np.isfinite(d)]
        med, mad, threshold = _label_threshold(finite, threshold_factor)
        mask = np.isfinite(d) & (d > threshold)
        n_blips = 0
        for s, e in _runs_from_mask(mask, max_gap):
            if max_blip_length is not None and (e - s + 1) > max_blip_length:
                continue
            if s - 1 < 0 or e + 1 >= n_frames:
                continue                                # no anchor on one side
            ab, aa = positions[s - 1, p], positions[e + 1, p]
            if not (np.isfinite(ab).all() and np.isfinite(aa).all()):
                continue
            blips.append(Blip(label=lab, start=s, end=e,
                              anchor_before=(float(ab[0]), float(ab[1])),
                              anchor_after=(float(aa[0]), float(aa[1])),
                              threshold=float(threshold)))
            n_blips += 1
        stats[lab] = dict(median_d=med, mad_d=mad, threshold=threshold,
                          n_blips=n_blips,
                          n_finite_frames=int(np.isfinite(d).sum()))
    return BlipReport(blips=blips, per_label_stats=stats,
                      params=dict(threshold_factor=threshold_factor,
                                  max_gap=max_gap, source="disagreement"),
                      n_frames=n_frames)

def blipped_positions(ann, report: BlipReport) -> dict:
    """The model's ORIGINAL positions at every flagged blip frame -- the sparse
    ``blips`` inspection layer (what the model got wrong), as
    ``{label: {frame: [x, y]}}``. Empty per-label dicts for un-blipped labels."""
    out: dict[str, dict[int, list[float]]] = {label: {} for label in ann.labels}
    for b in report.blips:
        for f in range(b.start, b.end + 1):
            xy = ann.data.get(b.label, {}).get(f)
            if xy is not None:
                out[b.label][int(f)] = [float(xy[0]), float(xy[1])]
    return out

def deblip_trace(ann, corrections) -> dict:
    """The dense DLC trace with blip frames overwritten by the LK-RSTC
    interpolation -- the ``deblip`` corrected output, as ``{label: {frame:
    [x, y]}}``. ``corrections`` is the sparse layer from :func:`interpolate_blips`.
    The source ``ann`` is left untouched (a per-label copy is spliced)."""
    out = {label: {int(f): [float(v[0]), float(v[1])] for f, v in ann.data[label].items()}
           for label in ann.labels}
    for label in corrections.labels:
        for f, xy in corrections.data[label].items():
            out[label][int(f)] = [float(xy[0]), float(xy[1])]
    return out

def deblip(ann, lk_pred, *, threshold_factor: float = 5.0, max_gap: int = 1,
           max_blip_length=None, lk_config=None, progress_callback=None):
    """One-call de-blip on the unified criterion: detect (LK-vs-DLC disagreement
    at 5-sigma) -> LK-RSTC interpolate -> the two output layer data dicts.

    Returns ``(report, corrections, blips_data, deblip_data)`` where
    ``corrections`` is the sparse interpolated :class:`VideoAnnotation`,
    ``blips_data`` the original blipped positions, and ``deblip_data`` the dense
    corrected trace (both ``{label: {frame: [x, y]}}``). The GUI builds the
    ``blips_``/``deblip_`` layers from the two dicts.
    """
    labels = list(ann.labels)
    n = int(getattr(ann, "n_frames", 0)) or (int(np.asarray(lk_pred).shape[0]))
    positions = np.full((n, len(labels), 2), np.nan)
    for p, label in enumerate(labels):
        for f, xy in ann.data[label].items():
            if 0 <= int(f) < n:
                positions[int(f), p] = xy
    report = disagreement_blips(positions, lk_pred, labels=labels,
                                threshold_factor=threshold_factor, max_gap=max_gap,
                                max_blip_length=max_blip_length)
    corrections = interpolate_blips(ann, report, lk_config=lk_config,
                                    progress_callback=progress_callback)
    return report, corrections, blipped_positions(ann, report), deblip_trace(ann, corrections)

def _lk_layer_data(lk_pred, labels) -> dict:
    """Dense ``{label: {frame: [x, y]}}`` from an ``(N, P, 2)`` LK-prediction
    array, skipping NaN frames (e.g. frame 0 / any point with no previous
    prediction). The payload of the saved ``lk_`` annotation layer."""
    lk_pred = np.asarray(lk_pred, dtype=float)
    out: dict[str, dict[int, list[float]]] = {str(lab): {} for lab in labels}
    for j, lab in enumerate(labels):
        col = lk_pred[:, j]
        for f in range(len(col)):
            if np.isfinite(col[f]).all():
                out[str(lab)][int(f)] = [float(col[f, 0]), float(col[f, 1])]
    return out

def save_lk_layer(video_path, prediction_h5, out_fname, *, lk_config=None,
                  progress_callback=None, overwrite=False):
    """Compute the LK-from-previous prediction for a DLC prediction ``.h5`` and
    save it as a dense ``lk_`` annotation layer at ``out_fname`` (paired with the
    ``dlc_`` trace, for overlay + blip detection). Opens the video once, shared
    between the LK pass and the layer. Skips an existing file unless
    ``overwrite``. Returns ``out_fname``."""
    import pandas as pd
    from datanavigator.video_reader import VideoReader

    from dustrack.flow_consistency import dlc_positions

    if os.path.exists(out_fname) and not overwrite:
        return out_fname
    positions, labels = dlc_positions(pd.read_hdf(prediction_h5))
    reader = VideoReader(str(video_path))
    lk_pred = compute_lk_predictions(positions, reader, lk_config=lk_config,
                                     progress_callback=progress_callback)
    out = VideoAnnotation(fname=str(out_fname), vname=None, n_labels=len(labels),
                          preloaded_json=_lk_layer_data(lk_pred, labels),
                          video=reader)
    out.save()
    return out_fname

def _lk_layer_worker(args):
    """Picklable ProcessPool worker for :meth:`DLCProject._save_lk_layers`.
    ``args`` is ``(video_path, prediction_h5, out_fname)``; returns
    ``(out_fname, error_or_None)`` so one video's failure doesn't kill the pool."""
    try:
        return save_lk_layer(*args), None
    except Exception as exc:                            # noqa: BLE001
        return args[2], repr(exc)
