"""Repair the model's *own* uncertainty, without a human in the loop.

A trained model tells you where it is lost: prediction likelihood
collapses. On a fresh pia02 participant, ~8% of frames fall below 0.6
while the video median is 1.000. Most of that is short -- median run 6
frames -- and sits between stretches the model is completely sure about.

Those short runs have their answer sitting next to them. Track across
the gap from both confident sides with LK-RSTC, and the corrected path
becomes training data for the next iteration. That is the same lever
:mod:`dustrack.blip` pulls (teach the model the temporally-consistent
answer over the visually-strongest one), driven by the model's own
confidence rather than by position spikes -- the two are complementary,
because a *confidently wrong* prediction has a high likelihood and a
low jerk, and neither detector sees it.

The bracket requirement is what makes this safe to automate, and it is
self-selecting. A gap with confident track on both sides has an answer
available from its neighbours. A gap that runs for thousands of frames
has no bracket, so it is excluded by construction -- and that is
precisely the bistable case, where the model oscillates between two
locally-plausible lanes and *neither* is obviously right. Those need a
human, and this module's job is to hand them over rather than guess.

Guessing is not a neutral failure here. Auto-generated labels that are
wrong are worse than no labels: the diagnosis of bistability is that it
accrues from cumulative training over labels that disagree across
non-adjacent repeats. A repair pass that quietly invents labels in the
ambiguous places would manufacture exactly the pathology it is meant to
relieve. So every gate below is written to *refuse* rather than
approximate.

The refusal that matters most is mid-gap drift. RSTC pins its output to
the anchors at both ends by construction, so endpoint error is
structurally near-zero and proves nothing. What does prove something is
running the forward and reverse tracks independently and asking whether
they agree in the middle: convergence means the answer is determined,
divergence means it is not -- however smooth the blended path looks.

Typical use::

    from dustrack import autorefine
    report = autorefine.find_gaps(h5_path)          # no video decode
    print(report.summary())
    result = autorefine.repair(report, ann)         # LK across each gap
    layer = autorefine.select_training_frames(result)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from dustrack.annotations import VideoAnnotation
from dustrack.lk_opticalflow import lucas_kanade_rstc

__all__ = [
    "Gap",
    "GapReport",
    "RepairResult",
    "find_gaps",
    "gaps_from_blips",
    "repair",
    "select_across_videos",
    "select_training_frames",
]

#: Below this, the model is treated as lost.
LOW_LIKELIHOOD = 0.6

#: A bracket frame must be at least this confident to anchor a repair.
HIGH_LIKELIHOOD = 0.9

#: Confident frames required on *each* side of a gap.
BRACKET_FRAMES = 5

#: Longest gap to attempt. LK is anchored at both ends, but the middle
#: of a long gap is unconstrained and drift grows with distance from
#: both anchors. ~0.7 s at pia02's 67 fps.
MAX_GAP = 45

#: Maximum forward-vs-reverse disagreement (px) for a repair to be
#: trusted. The interosseous points move slowly -- the median
#: frame-to-frame step on a confident stretch is ~0.05 px -- so tracks
#: that agree are agreeing tightly, and several px of divergence is a
#: real disagreement rather than noise.
MAX_TRACK_DISAGREEMENT = 5.0


@dataclass
class Gap:
    """A low-likelihood run with confident track on both sides."""

    label: str
    start: int              # first uncertain frame, inclusive
    end: int                # last uncertain frame, inclusive
    anchor_before: tuple[float, float]
    anchor_after: tuple[float, float]
    min_likelihood: float

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    @property
    def anchor_travel(self) -> float:
        """How far the point moved across the gap, per the anchors."""
        return float(
            np.linalg.norm(
                np.asarray(self.anchor_after) - np.asarray(self.anchor_before)
            )
        )


@dataclass
class GapReport:
    """What :func:`find_gaps` found, and what it declined."""

    gaps: list[Gap] = field(default_factory=list)
    #: Runs rejected, by reason -- the interesting half of the output.
    rejected: dict[str, list[tuple[str, int, int]]] = field(default_factory=dict)
    per_label: dict[str, dict] = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    n_frames: int = 0

    def by_label(self) -> dict[str, list[Gap]]:
        out: dict[str, list[Gap]] = {}
        for g in self.gaps:
            out.setdefault(g.label, []).append(g)
        return out

    def summary(self) -> str:
        lines = [f"{self.n_frames} frames, {len(self.gaps)} repairable gap(s)"]
        for label, stats in sorted(self.per_label.items()):
            lines.append(
                f"  label {label}: {stats['n_low']} uncertain frames "
                f"({100 * stats['n_low'] / max(self.n_frames, 1):.2f}%) in "
                f"{stats['n_runs']} run(s); {stats['n_gaps']} repairable"
            )
        for reason, items in sorted(self.rejected.items()):
            frames = sum(e - s + 1 for _, s, e in items)
            lines.append(f"  declined [{reason}]: {len(items)} run(s), {frames} frames")
        return "\n".join(lines)


@dataclass
class RepairResult:
    """Outcome of :func:`repair`.

    ``corrections`` holds every repaired frame, for inspection.
    ``disagreement`` carries the per-frame forward-vs-reverse distance,
    and that is what decides which of them may become training data.

    Trust is per *frame*, not per gap, because it varies systematically
    within one: the two tracks are pinned together at the anchors and
    drift apart toward the middle. Measured on s061 trial 001, gap-level
    rejection at 5 px discarded 43% of gaps wholesale -- including the
    well-determined frames near their edges. Those edge frames are the
    valuable ones: the model was unsure there, yet the temporal evidence
    is unambiguous, which is precisely the lesson worth training on.
    """

    corrections: VideoAnnotation | None = None
    repaired: list[Gap] = field(default_factory=list)
    #: ``{label: {frame: forward-vs-reverse distance in px}}``.
    disagreement: dict[str, dict[int, float]] = field(default_factory=dict)
    #: Gaps rejected outright -- the two tracks never converged at all.
    untrusted: list[tuple[Gap, float]] = field(default_factory=list)

    def frame_disagreements(self) -> np.ndarray:
        return np.array(
            [d for per in self.disagreement.values() for d in per.values()]
        )

    def n_trusted(self, max_disagreement: float) -> int:
        return int((self.frame_disagreements() <= max_disagreement).sum())

    def summary(self) -> str:
        n = sum(
            len(v) for v in (self.corrections.data.values() if self.corrections else [])
        )
        lines = [
            f"repaired {len(self.repaired)} gap(s) -> {n} corrected frame(s)",
            f"rejected {len(self.untrusted)} gap(s) outright",
        ]
        d = self.frame_disagreements()
        if d.size:
            lines.append(
                "  per-frame forward/reverse disagreement: median "
                f"{np.median(d):.2f} px, p90 {np.percentile(d, 90):.2f}, "
                f"max {d.max():.2f}"
            )
            for thr in (1.0, 2.0, 5.0):
                lines.append(
                    f"    <= {thr:.0f} px: {int((d <= thr).sum())}/{d.size} frames"
                )
        return "\n".join(lines)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True regions as inclusive (start, end) pairs."""
    idx = np.flatnonzero(mask)
    if not idx.size:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def find_gaps(
    predictions,
    *,
    low: float = LOW_LIKELIHOOD,
    high: float = HIGH_LIKELIHOOD,
    bracket: int = BRACKET_FRAMES,
    max_gap: int = MAX_GAP,
    labels: Sequence[str] | None = None,
) -> GapReport:
    """Find low-likelihood runs that confident track brackets on both sides.

    Pure numpy over the prediction table -- no video decode, so this is
    cheap enough to run over a whole cohort.

    Args:
        predictions: A DLC prediction ``DataFrame`` or a path to its h5.
            Likelihood is required, which is why this reads predictions
            rather than a :class:`VideoAnnotation` -- the annotation
            format keeps only ``[x, y]``.
        low: At or below this likelihood the model is treated as lost.
        high: Bracket frames must be at least this confident.
        bracket: Confident frames required on each side.
        max_gap: Longest run to attempt.
        labels: Restrict to these bodyparts (default: all).

    Returns:
        A :class:`GapReport`. Its ``rejected`` map is the load-bearing
        half: runs declined for want of a bracket are the ones a human
        has to look at, and quietly dropping them would hide the work.
    """
    df = pd.read_hdf(predictions) if isinstance(predictions, (str, Path)) else predictions
    scorer = df.columns.levels[0][0]
    bodyparts = list(labels) if labels else list(df.columns.levels[1])

    report = GapReport(
        n_frames=len(df),
        params=dict(low=low, high=high, bracket=bracket, max_gap=max_gap),
    )
    frames = df.index.to_numpy()

    for bp in bodyparts:
        lik = df.loc[:, (scorer, bp, "likelihood")].to_numpy()
        xs = df.loc[:, (scorer, bp, "x")].to_numpy()
        ys = df.loc[:, (scorer, bp, "y")].to_numpy()
        # Layer labels are the annotation-side names ('0'), not the
        # DLC bodypart names ('point0'), so a repaired layer lines up
        # with the rest of the session.
        label = bp[len("point"):] if bp.startswith("point") else bp
        confident = lik >= high

        # A gap is a maximal run of *non-confident* frames, so its
        # neighbours are confident by construction. Defining the gap by
        # one threshold and the bracket by another leaves a middle band
        # (here 0.6-0.9) that satisfies neither, which splits one real
        # problem into many runs that each fail to find an anchor: on
        # s061 trial 001 that yielded 6 repairable gaps out of 1353.
        # ``low`` is now a *severity* filter instead -- a gap has to
        # contain a genuinely lost frame to be worth repairing, rather
        # than merely a mediocre one.
        runs = _runs(~confident)
        n_gaps = 0
        for s, e in runs:
            if lik[s : e + 1].min() > low:
                report.rejected.setdefault("not_severe", []).append((label, s, e))
                continue
            if e - s + 1 > max_gap:
                # No confident bracket within reach. Long runs of this
                # are the bistable case: the model oscillates between
                # two locally-plausible lanes for a sustained stretch
                # and neither is obviously right. Hand it to a human.
                report.rejected.setdefault("too_long", []).append((label, s, e))
                continue
            if s - bracket < 0 or e + bracket >= len(lik):
                report.rejected.setdefault("at_video_edge", []).append((label, s, e))
                continue
            pre = slice(s - bracket, s)
            post = slice(e + 1, e + 1 + bracket)
            if not (confident[pre].all() and confident[post].all()):
                # Should be rare now that runs are defined by confidence
                # -- happens when the confident stretch either side is
                # shorter than ``bracket``.
                report.rejected.setdefault("bracket_too_short", []).append(
                    (label, s, e)
                )
                continue
            report.gaps.append(
                Gap(
                    label=label,
                    start=int(frames[s]),
                    end=int(frames[e]),
                    anchor_before=(float(xs[s - 1]), float(ys[s - 1])),
                    anchor_after=(float(xs[e + 1]), float(ys[e + 1])),
                    min_likelihood=float(lik[s : e + 1].min()),
                )
            )
            n_gaps += 1

        report.per_label[label] = dict(
            n_low=int((lik <= low).sum()),
            n_unconfident=int((~confident).sum()),
            n_runs=len(runs),
            n_gaps=n_gaps,
            n_gap_frames=sum(
                g.length for g in report.gaps if g.label == label
            ),
            mean_likelihood=float(lik.mean()),
        )

    report.gaps.sort(key=lambda g: (g.label, g.start))
    return report


def gaps_from_blips(
    blip_report,
    predictions=None,
    *,
    confident_only: bool = True,
    high: float = HIGH_LIKELIHOOD,
) -> GapReport:
    """Adapt :mod:`dustrack.blip`'s detections into a :class:`GapReport`.

    Likelihood finds where the model knows it is lost; it is blind to
    where the model is *confidently wrong*, which scores high and moves
    smoothly. ``blip`` finds those from the geometry instead. On s061
    trial 001 the two barely overlap: of 3308 detected blips, **2120 sit
    at likelihood >= 0.80** -- many at exactly 1.00 -- so the confidence
    pass never sees them. They are not a mop-up round, they are a peer.

    Routing them through here means both detectors share one repair path
    and one trust metric, rather than the blip path trusting its own
    output by default.

    Args:
        blip_report: A ``dustrack.blip.BlipReport``.
        predictions: Optional prediction table (or h5 path) used to
            record each blip's minimum likelihood and, with
            ``confident_only``, to filter.
        confident_only: Keep only blips the confidence pass would miss,
            so the two rounds do not relabel the same frames.
        high: The confidence bar that defines "would have been caught".
    """
    df = (
        pd.read_hdf(predictions)
        if isinstance(predictions, (str, Path))
        else predictions
    )
    lik = {}
    if df is not None:
        scorer = df.columns.levels[0][0]
        for bp in df.columns.levels[1]:
            label = bp[len("point"):] if bp.startswith("point") else bp
            lik[label] = df.loc[:, (scorer, bp, "likelihood")].to_numpy()

    report = GapReport(
        n_frames=len(df) if df is not None else 0,
        params=dict(source="blip", confident_only=confident_only, high=high),
    )
    for b in blip_report.blips:
        series = lik.get(b.label)
        min_lik = (
            float(series[b.start : b.end + 1].min()) if series is not None else float("nan")
        )
        if confident_only and series is not None and min_lik < high:
            # The confidence pass already owns this one.
            report.rejected.setdefault("also_low_likelihood", []).append(
                (b.label, b.start, b.end)
            )
            continue
        report.gaps.append(
            Gap(
                label=b.label,
                start=b.start,
                end=b.end,
                anchor_before=tuple(b.anchor_before),
                anchor_after=tuple(b.anchor_after),
                min_likelihood=min_lik,
            )
        )
    for label in {g.label for g in report.gaps}:
        report.per_label[label] = dict(
            n_gaps=sum(1 for g in report.gaps if g.label == label),
            n_gap_frames=sum(g.length for g in report.gaps if g.label == label),
        )
    report.gaps.sort(key=lambda g: (g.label, g.start))
    return report


def sample_gaps(report: GapReport, max_gaps: int) -> GapReport:
    """Thin a report to at most ``max_gaps``, spread over the video.

    Repair costs an LK pass per gap (~300 ms), and a refinement round
    only needs a couple of labels per video -- repairing 2000 candidate
    blips to then select two is most of a day's compute for nothing.
    Sampling evenly across the timeline rather than taking the first N
    keeps the candidates representative of the whole video.
    """
    if len(report.gaps) <= max_gaps:
        return report
    by_label: dict[str, list[Gap]] = {}
    for g in report.gaps:
        by_label.setdefault(g.label, []).append(g)

    kept: list[Gap] = []
    share = max(1, max_gaps // max(1, len(by_label)))
    for label, gaps in by_label.items():
        gaps.sort(key=lambda g: g.start)
        if len(gaps) <= share:
            kept.extend(gaps)
            continue
        idx = np.linspace(0, len(gaps) - 1, share).round().astype(int)
        kept.extend(gaps[i] for i in sorted(set(idx.tolist())))

    thinned = GapReport(
        gaps=sorted(kept, key=lambda g: (g.label, g.start)),
        rejected=report.rejected,
        per_label=report.per_label,
        params={**report.params, "sampled_from": len(report.gaps)},
        n_frames=report.n_frames,
    )
    return thinned


def repair(
    report: GapReport,
    ann: VideoAnnotation,
    *,
    max_disagreement: float = MAX_TRACK_DISAGREEMENT,
    progress_callback: Callable[[int, int], None] | None = None,
    lk_config: dict | None = None,
) -> RepairResult:
    """LK-RSTC across each bracketed gap, keeping only trusted repairs.

    The forward and reverse tracks are run independently and compared.
    RSTC's blend pins both ends to the anchors whatever happens in
    between, so a smooth-looking path is not evidence of anything; two
    independent tracks landing on the same mid-gap answer is. Gaps whose
    tracks diverge past ``max_disagreement`` are returned in
    ``untrusted`` rather than corrected -- a wrong auto-label is worse
    than an unrepaired gap, because it teaches the model a lane that
    isn't there.

    Args:
        report: From :func:`find_gaps`.
        ann: Annotation supplying the video reader (``ann.video``).
        max_disagreement: Forward/reverse tolerance, px.
        progress_callback: ``(done, total)``, fired per gap.
        lk_config: Passed through to ``lucas_kanade_rstc``.
    """
    if ann.video is None:
        raise ValueError("repair needs a video reader (ann.video is None)")
    lk_kwargs = dict(lk_config or {})
    sparse: dict[str, dict[int, list[float]]] = {
        label: {} for label in ann.labels
    }
    for g in report.gaps:
        sparse.setdefault(g.label, {})

    result = RepairResult()
    total = len(report.gaps)
    for done, gap in enumerate(report.gaps, start=1):
        start_pts = np.array([gap.anchor_before], dtype=np.float32)
        end_pts = np.array([gap.anchor_after], dtype=np.float32)
        rstc, fwd, rev = lucas_kanade_rstc(
            ann.video,
            gap.start - 1,
            gap.end + 1,
            start_pts,
            end_pts,
            return_paths=True,
            **lk_kwargs,
        )
        # Compare only the interior; the endpoints are the anchors
        # themselves and agree trivially.
        per_frame = np.linalg.norm(
            fwd[1:-1, 0] - rev[1:-1, 0], axis=-1
        )
        worst = float(per_frame.max()) if per_frame.size else 0.0

        if worst > max_disagreement:
            # The two tracks never converged anywhere in this gap --
            # nothing here is worth keeping, not even near the anchors.
            result.untrusted.append((gap, worst))
        else:
            per = result.disagreement.setdefault(gap.label, {})
            for offset, xy in enumerate(rstc[1:-1, 0, :]):
                frame = gap.start + offset
                sparse[gap.label][frame] = [float(xy[0]), float(xy[1])]
                per[frame] = float(per_frame[offset])
            result.repaired.append(gap)

        if progress_callback is not None:
            progress_callback(done, total)

    stem = Path(ann.fname).stem if ann.fname else "autorefine"
    result.corrections = VideoAnnotation(
        fname=f"{stem}_autorefine.json",
        vname=None,
        n_labels=max(1, len(sparse)),
        preloaded_json=sparse,
        video=ann.video,
    )
    return result


def _confident_runs(confident: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest confident frame before / after each index.

    Returns two arrays the length of ``confident``: for position ``i``,
    the index of the closest confident frame strictly before ``i`` (or
    ``-1``) and strictly after (or ``len``). Precomputed once so a
    co-label's LK bracket is a lookup rather than a scan per frame.
    """
    n = len(confident)
    before = np.full(n, -1, dtype=int)
    after = np.full(n, n, dtype=int)
    last = -1
    for i in range(n):
        before[i] = last
        if confident[i]:
            last = i
    nxt = n
    for i in range(n - 1, -1, -1):
        after[i] = nxt
        if confident[i]:
            nxt = i
    return before, after


def _lk_estimate_at(
    ann: VideoAnnotation,
    xs: np.ndarray,
    ys: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    frame: int,
    *,
    max_gap: int,
    max_disagreement: float,
    lk_kwargs: dict,
) -> list[float] | None:
    """LK-RSTC one label across ``frame`` from its confident neighbours.

    A single-frame version of :func:`repair`, used to fill a co-label
    that the model is *not* confident about at a selected frame. Returns
    ``None`` -- refuse -- when there is no confident bracket within
    ``max_gap`` or the forward/reverse tracks disagree past
    ``max_disagreement``, so a co-label is never invented.
    """
    if ann.video is None:
        return None
    a, b = int(before[frame]), int(after[frame])
    if a < 0 or b >= len(xs) or (b - a) > max_gap:
        return None
    start_pts = np.array([[xs[a], ys[a]]], dtype=np.float32)
    end_pts = np.array([[xs[b], ys[b]]], dtype=np.float32)
    rstc, fwd, rev = lucas_kanade_rstc(
        ann.video, a, b, start_pts, end_pts, return_paths=True, **lk_kwargs
    )
    i = frame - a
    if not (0 <= i < len(rstc)):
        return None
    if float(np.linalg.norm(fwd[i, 0] - rev[i, 0])) > max_disagreement:
        return None
    return [float(rstc[i, 0, 0]), float(rstc[i, 0, 1])]


def complete_frames(
    selected: dict[str, dict[int, list[float]]],
    predictions,
    ann: VideoAnnotation,
    *,
    high: float = HIGH_LIKELIHOOD,
    max_gap: int = MAX_GAP,
    max_disagreement: float = MAX_TRACK_DISAGREEMENT,
    lk_config: dict | None = None,
) -> tuple[dict[str, dict[int, list[float]]], dict]:
    """Make every selected frame carry *all* labels before it is trained on.

    A frame is chosen because *one* point had a repairable gap, so the
    other point is absent from ``selected`` there. Written out that way it
    reaches DLC's ``CollectedData`` as a NaN, and DLC reads an absent
    bodypart as an occluded keypoint -- training its confidence head
    *down* on that appearance. Half-labelled frames therefore teach the
    model to be unsure, the opposite of the intent. (Measured: retraining
    s061 on gaps that were almost all point1's collapsed point0's
    likelihood to ~0.60 across every video while its tracked position
    barely moved -- a confidence artifact, not a track that improved.)

    Each label missing at a selected frame is filled, in order:

    * the model's own prediction, when it is confident there
      (likelihood >= ``high``) -- "the other point seems good";
    * else a short bracketed LK-RSTC estimate, kept only if forward and
      reverse agree within ``max_disagreement``;
    * else the whole frame is dropped -- one fewer training frame beats a
      guessed co-label, which is the exact input that breeds bistability.

    Returns the completed ``{label: {frame: [x, y]}}`` (every frame now
    present under every label) and a small tally of how each hole was
    filled.
    """
    df = (
        pd.read_hdf(predictions)
        if isinstance(predictions, (str, Path))
        else predictions
    )
    scorer = df.columns.levels[0][0]
    labels = [bp[len("point"):] if bp.startswith("point") else bp
              for bp in df.columns.levels[1]]
    bp_of = {lab: bp for lab, bp in zip(labels, df.columns.levels[1])}

    lik = {lab: df.loc[:, (scorer, bp_of[lab], "likelihood")].to_numpy()
           for lab in labels}
    xs = {lab: df.loc[:, (scorer, bp_of[lab], "x")].to_numpy() for lab in labels}
    ys = {lab: df.loc[:, (scorer, bp_of[lab], "y")].to_numpy() for lab in labels}
    frame_index = {int(f): i for i, f in enumerate(df.index.to_numpy())}
    brackets = {lab: _confident_runs(lik[lab] >= high) for lab in labels}
    lk_kwargs = dict(lk_config or {})

    frameset = sorted({f for per in selected.values() for f in per})
    completed: dict[str, dict[int, list[float]]] = {lab: {} for lab in labels}
    stats = dict(kept=0, dropped=0, by_prediction=0, by_lk=0, given=0)

    for f in frameset:
        row: dict[str, list[float]] = {}
        ok = True
        i = frame_index.get(int(f))
        for lab in labels:
            if f in selected.get(lab, {}):
                row[lab] = selected[lab][f]
                stats["given"] += 1
            elif i is not None and lik[lab][i] >= high:
                row[lab] = [float(xs[lab][i]), float(ys[lab][i])]
                stats["by_prediction"] += 1
            elif i is not None:
                bef, aft = brackets[lab]
                est = _lk_estimate_at(
                    ann, xs[lab], ys[lab], bef, aft, i,
                    max_gap=max_gap, max_disagreement=max_disagreement,
                    lk_kwargs=lk_kwargs,
                )
                if est is None:
                    ok = False
                    break
                row[lab] = est
                stats["by_lk"] += 1
            else:
                ok = False
                break
        if ok:
            for lab, xy in row.items():
                completed[lab][int(f)] = xy
            stats["kept"] += 1
        else:
            stats["dropped"] += 1

    return completed, stats


def select_across_videos(
    results: dict,
    *,
    n: int = 20,
    max_disagreement: float = 4.0,
    min_spacing: int = 10,
    per_video_cap: int | None = None,
) -> dict[str, dict[str, dict[int, list[float]]]]:
    """Pick this round's ``n`` best labels, spread across videos.

    Rank-based rather than threshold-based: take the ``n`` most
    trustworthy candidates and let the threshold *emerge*, with
    ``max_disagreement`` only as a floor on acceptable quality. The
    round is finished not when a fixed number is reached but when
    nothing under the floor remains -- which is also the natural stop
    for the whole curriculum.

    The per-video quota matters because one model serves every trial of
    an arm. Pure global ranking would hand a whole round's budget to
    whichever video happens to have the tightest LK, teaching the model
    that trial's appearance and no other. The quota is a ceiling, not
    a reservation: a video with nothing good to offer forfeits its share
    to the others rather than dragging in weak labels to fill it.

    Args:
        results: ``{video_key: RepairResult}``.
        n: Total labels wanted this round.
        max_disagreement: Quality floor, px.
        min_spacing: Minimum frame gap between two picks in one video.
        per_video_cap: Override the default ``ceil(n / n_videos)``.

    Returns:
        ``{video_key: {label: {frame: [x, y]}}}`` -- only videos that
        contributed.
    """
    if not results:
        return {}
    cap = per_video_cap or int(np.ceil(n / len(results)))

    # (disagreement, video, label, frame) for everything eligible.
    pool: list[tuple[float, str, str, int]] = []
    for key, res in results.items():
        for label, per in res.disagreement.items():
            for frame, d in per.items():
                if d <= max_disagreement:
                    pool.append((d, key, label, frame))
    pool.sort()

    picked: dict[str, dict[str, dict[int, list[float]]]] = {}
    per_video_count: dict[str, int] = {}

    def _take(enforce_cap: bool) -> None:
        for d, key, label, frame in pool:
            if sum(per_video_count.values()) >= n:
                return
            if enforce_cap and per_video_count.get(key, 0) >= cap:
                continue
            kept = picked.setdefault(key, {}).setdefault(label, {})
            if frame in kept or any(abs(frame - f) < min_spacing for f in kept):
                continue
            kept[frame] = results[key].corrections.data[label][frame]
            per_video_count[key] = per_video_count.get(key, 0) + 1

    _take(enforce_cap=True)
    # Second pass without the cap: a video with nothing under the floor
    # forfeits its share rather than reserving it. Holding the budget
    # back would leave the round underfilled -- if only 3 of 10 trials
    # have good candidates, a strict cap would spend 3/10 of the round
    # and stop.
    if sum(per_video_count.values()) < n:
        _take(enforce_cap=False)

    return {k: v for k, v in picked.items() if any(v.values())}


def select_training_frames(
    result: RepairResult,
    *,
    max_disagreement: float = 2.0,
    per_gap: int = 2,
    max_frames: int | None = 150,
    min_spacing: int = 10,
) -> VideoAnnotation:
    """Thin the trusted repairs down to a training set worth adding.

    Two filters, in order.

    **Trust.** Only frames whose forward and reverse tracks landed
    within ``max_disagreement`` are eligible. This is the gate that
    keeps a wrong label out of the training set, and it is deliberately
    tighter than the gap-level cap in :func:`repair`: a label carrying
    several px of error teaches the model a position that is not there,
    and the failure mode being mitigated -- bistability -- is caused by
    exactly that kind of inconsistency accumulating.

    **Redundancy.** Consecutive frames within a gap are near-duplicates;
    they inflate the training set without adding information. Take
    ``per_gap`` evenly spaced survivors, drop anything within
    ``min_spacing`` of a frame already chosen, and cap the total at
    ``max_frames`` by thinning uniformly -- a budget spent entirely on
    the first minute would leave the rest of the video unrefined.
    """
    corr = result.corrections
    if corr is None:
        raise ValueError("nothing to select from; run repair first")

    chosen: dict[str, list[int]] = {}
    for gap in result.repaired:
        per = result.disagreement.get(gap.label, {})
        frames = [
            f
            for f in sorted(corr.data.get(gap.label, {}))
            if gap.start <= f <= gap.end and per.get(f, np.inf) <= max_disagreement
        ]
        if not frames:
            continue
        if per_gap >= len(frames):
            picks = frames
        else:
            idx = np.linspace(0, len(frames) - 1, per_gap).round().astype(int)
            picks = [frames[i] for i in sorted(set(idx.tolist()))]
        kept = chosen.setdefault(gap.label, [])
        for f in picks:
            if not kept or f - kept[-1] >= min_spacing:
                kept.append(f)

    if max_frames is not None:
        total = sum(len(v) for v in chosen.values())
        if total > max_frames:
            # Thin uniformly across the video, per label, so coverage
            # stays spread rather than front-loaded.
            for label, frames in chosen.items():
                keep_n = max(1, round(len(frames) * max_frames / total))
                idx = np.linspace(0, len(frames) - 1, keep_n).round().astype(int)
                chosen[label] = [frames[i] for i in sorted(set(idx.tolist()))]

    data = {
        label: {f: corr.data[label][f] for f in frames}
        for label, frames in chosen.items()
    }
    for label in corr.labels:
        data.setdefault(label, {})

    stem = Path(corr.fname).stem if corr.fname else "autorefine"
    return VideoAnnotation(
        fname=f"{stem}_training.json",
        vname=None,
        n_labels=max(1, len(data)),
        preloaded_json=data,
        video=corr.video,
    )
