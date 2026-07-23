"""Residual between a model's predictions and the optical flow it should obey.

A frame-independent model -- a per-frame CNN like DLC's ResNet -- can place
a point somewhere its own neighbours contradict: the optical flow between
two adjacent frames says the point drifted a pixel, the model says it
jumped fifty. That contradiction is visible with **no ground truth** and
**independent of how fast the point is really moving** -- real motion moves
the flow too, so it cancels; only a model error survives. It is the honest
"is the model wrong here?" signal, and unlike prediction likelihood it does
not go quiet where the model is confidently wrong.

The primitive here computes that residual, drift-free. For each frame it
re-seeds Lucas-Kanade at the model's prediction on the *neighbour* frame
and steps one frame in -- forward from ``n-1``, reverse from ``n+1`` -- and
reports how far the model's own prediction at ``n`` sits from where the flow
landed. Re-seeding every frame (rather than following a long anchored track)
is what keeps it local and drift-free: the residual at ``n`` depends only on
frames ``n-1, n, n+1``.

It is deliberately **model- and task-agnostic**: it takes an array of
positions and a video, nothing about DLC or a particular study. Three
consumers are already in view, and the split exists so they share one
definition:

* :mod:`dustrack.blip` turns a large residual into a repair candidate
  (the confidently-wrong frames likelihood misses);
* the same residual is a natural **training signal** -- a point the flow
  contradicts is a point to down-weight or relabel;
* and it is an **analysis axis** in its own right -- *where*, and *why*,
  does a trained model disagree with the flow it should obey? (LK does not
  always win locally, and the residual is how you find where it doesn't.)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datanavigator.video_reader import VideoReader

from dustrack.lk_opticalflow import _gray_rgb, _lk_track_frames

__all__ = ["FlowResidual", "flow_residual", "dlc_positions"]


@dataclass
class FlowResidual:
    """Per-frame model-vs-flow residual over a set of evaluated frames.

    All arrays are aligned on :attr:`frames` (axis 0) and, where they have
    one, on :attr:`labels` (axis 1).
    """

    frames: np.ndarray          # (M,) the frames actually evaluated
    residual: np.ndarray        # (M, P) mean of fwd/rev ||pred - lk||, px
    lk_estimate: np.ndarray     # (M, P, 2) mean fwd/rev flow landing = the correction
    agreement: np.ndarray       # (M, P) ||lk_fwd - lk_rev|| -- is the flow determined?
    labels: list                # length P, index-aligned to axis 1

    def as_dict(self, label: str) -> dict[int, float]:
        """``{frame: residual}`` for one label, NaNs dropped."""
        p = self.labels.index(label)
        col = self.residual[:, p]
        ok = np.isfinite(col)
        return {int(f): float(v) for f, v in zip(self.frames[ok], col[ok])}


def dlc_positions(df, *, label_prefix: str = "point"):
    """``(positions (N,P,2), labels)`` from a DLC prediction ``DataFrame``.

    A convenience bridge only -- :func:`flow_residual` itself never sees a
    DataFrame, so it stays usable for any tracker.
    """
    scorer = df.columns.levels[0][0]
    bodyparts = list(df.columns.levels[1])
    labels = [bp[len(label_prefix):] if bp.startswith(label_prefix) else bp
              for bp in bodyparts]
    pos = np.stack(
        [df.loc[:, (scorer, bp, ["x", "y"])].to_numpy() for bp in bodyparts],
        axis=1,
    )
    return pos, labels


def flow_residual(
    positions,
    video,
    *,
    frames=None,
    labels=None,
    lk_config: dict | None = None,
) -> FlowResidual:
    """Model-vs-flow residual, re-seeded every frame (drift-free).

    Args:
        positions: ``(N, P, 2)`` model predictions, indexed by frame
            ``0..N-1`` (dense: row ``m`` is the prediction at frame ``m``).
        video: a ``VideoReader``/``utils.Video`` or a path.
        frames: which frames to evaluate. ``None`` = every interior frame.
            Pass a candidate subset to pay LK only where it matters -- the
            residual at ``m`` needs only frames ``m-1, m, m+1`` decoded.
        labels: point names for the ``P`` axis (default ``"0".."P-1"``).
        lk_config: overrides for the OpenCV LK call (window, pyramid, ...).

    Returns:
        A :class:`FlowResidual`. A frame is skipped (NaN row) when it lacks
        a finite prediction on itself or a neighbour.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError("positions must be (N, P, 2)")
    n_frames, n_pts = positions.shape[:2]
    labels = list(labels) if labels is not None else [str(i) for i in range(n_pts)]
    if len(labels) != n_pts:
        raise ValueError("labels length must match positions' point axis")
    if isinstance(video, (str, Path)):
        video = VideoReader(str(video))
    cfg = dict(lk_config or {})

    if frames is None:
        cand = np.arange(1, n_frames - 1)
    else:
        cand = np.array(
            sorted({int(f) for f in frames if 1 <= int(f) <= n_frames - 2}),
            dtype=int,
        )

    residual = np.full((len(cand), n_pts), np.nan)
    lk_estimate = np.full((len(cand), n_pts, 2), np.nan)
    agreement = np.full((len(cand), n_pts), np.nan)
    if not len(cand):
        return FlowResidual(cand, residual, lk_estimate, agreement, labels)

    needed = sorted(set(cand) | set(cand - 1) | set(cand + 1))
    gray = {int(m): _gray_rgb(video, int(m)) for m in needed}

    for k, m in enumerate(cand):
        prev, cur, nxt = positions[m - 1], positions[m], positions[m + 1]
        ok = (np.isfinite(prev).all(1) & np.isfinite(cur).all(1)
              & np.isfinite(nxt).all(1))
        if not ok.any():
            continue
        # cv2 LK cannot take NaN seeds; feed 0 and mask the results back out.
        seed_fwd = np.where(np.isfinite(prev), prev, 0.0).astype(np.float32)
        seed_rev = np.where(np.isfinite(nxt), nxt, 0.0).astype(np.float32)
        lk_fwd = _lk_track_frames([gray[m - 1], gray[m]], seed_fwd, **cfg)[-1]
        lk_rev = _lk_track_frames([gray[m + 1], gray[m]], seed_rev, **cfg)[-1]
        d_fwd = np.linalg.norm(cur - lk_fwd, axis=1)
        d_rev = np.linalg.norm(cur - lk_rev, axis=1)
        residual[k] = np.where(ok, 0.5 * (d_fwd + d_rev), np.nan)
        lk_estimate[k] = np.where(ok[:, None], 0.5 * (lk_fwd + lk_rev), np.nan)
        agreement[k] = np.where(ok, np.linalg.norm(lk_fwd - lk_rev, axis=1), np.nan)

    return FlowResidual(cand, residual, lk_estimate, agreement, labels)
