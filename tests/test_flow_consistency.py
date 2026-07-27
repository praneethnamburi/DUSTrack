"""Tests for :mod:`dustrack.flow_consistency`.

The residual is only meaningful if LK actually tracks, so these run real
Lucas-Kanade over a synthetic *translating textured* video (a blurred
random field rolled a pixel per frame -- trackable, unlike per-pixel
noise). Ground truth is therefore exact: a point sitting on the true track
must read a near-zero residual, and an injected jump must read back its own
size with the forward/reverse flow still agreeing (so the correction is
trusted).
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from dustrack import flow_consistency as fc

class _FakeFrame:
    def __init__(self, arr):
        self._a = arr

    def asnumpy(self):
        return self._a

class _FakeVideo:
    """Minimal VideoReader stand-in: ``video[i].asnumpy()`` -> gray frame."""

    def __init__(self, frames):
        self._f = frames

    def __getitem__(self, i):
        return _FakeFrame(self._f[int(i)])

    def __len__(self):
        return len(self._f)

def make_video(n=30, dx=1, dy=1, size=220):
    rng = np.random.default_rng(0)
    base = (rng.random((size, size)).astype(np.float32) * 255)
    base = cv2.GaussianBlur(base, (0, 0), 2.0)          # LK-trackable texture
    frames = [
        np.roll(np.roll(base, m * dy, axis=0), m * dx, axis=1).astype(np.uint8)
        for m in range(n)
    ]
    return _FakeVideo(frames), dx, dy

def true_track(n, dx, dy, q):
    return np.array([[q[0] + m * dx, q[1] + m * dy] for m in range(n)], float)

def two_point_positions(n, dx, dy, q0=(80, 90), q1=(140, 130)):
    p0 = true_track(n, dx, dy, q0)
    p1 = true_track(n, dx, dy, q1)
    return np.stack([p0, p1], axis=1)                    # (n, 2, 2)

class TestFlowResidual:
    def test_on_track_residual_is_near_zero(self):
        vid, dx, dy = make_video()
        pos = two_point_positions(30, dx, dy)
        fr = fc.flow_residual(pos, vid)
        # every interior frame evaluated, both points
        assert fr.residual.shape == (28, 2)
        assert np.nanmax(fr.residual) < 3.0             # sub-pixel LK error only

    def test_injected_jump_reads_back_its_size(self):
        vid, dx, dy = make_video()
        pos = two_point_positions(30, dx, dy)
        pos[15, 0] = pos[15, 0] + [30.0, 0.0]           # 30 px blip on point0 @ 15
        fr = fc.flow_residual(pos, vid)
        row = int(np.where(fr.frames == 15)[0][0])
        assert 25 < fr.residual[row, 0] < 35            # recovers ~30
        assert fr.residual[row, 1] < 3.0                # point1 untouched

    def test_correction_points_back_at_the_truth(self):
        vid, dx, dy = make_video()
        pos = two_point_positions(30, dx, dy)
        truth = pos[15, 0].copy()
        pos[15, 0] = truth + [30.0, 0.0]
        fr = fc.flow_residual(pos, vid)
        row = int(np.where(fr.frames == 15)[0][0])
        # the flow estimate is the corrected label, and the two directions
        # agree (a single-frame jump is determined from both sides)
        assert np.linalg.norm(fr.lk_estimate[row, 0] - truth) < 3.0
        assert fr.agreement[row, 0] < 3.0

    def test_frames_subset_only_pays_where_asked(self):
        vid, dx, dy = make_video()
        pos = two_point_positions(30, dx, dy)
        fr = fc.flow_residual(pos, vid, frames=[10, 15, 20])
        assert fr.frames.tolist() == [10, 15, 20]
        assert fr.residual.shape == (3, 2)

    def test_nan_prediction_is_skipped(self):
        vid, dx, dy = make_video()
        pos = two_point_positions(30, dx, dy)
        pos[10, 0] = [np.nan, np.nan]
        fr = fc.flow_residual(pos, vid, frames=[10])
        assert np.isnan(fr.residual[0, 0])              # point0 skipped
        assert np.isfinite(fr.residual[0, 1])           # point1 fine

    def test_labels_flow_through_and_as_dict(self):
        vid, dx, dy = make_video()
        pos = two_point_positions(30, dx, dy)
        pos[15, 1] = pos[15, 1] + [20.0, 0.0]
        fr = fc.flow_residual(pos, vid, labels=["0", "1"])
        d = fr.as_dict("1")
        # the jump reads back at 15; frames not adjacent to it stay small
        assert d[15] > 15 and all(v < 5 for k, v in d.items() if abs(k - 15) > 1)

    def test_blip_neighbours_elevate_residual_but_fail_the_trust_gate(self):
        """A one-frame jump seeds its neighbours' fwd/rev estimates from the
        wrong place, so their residual rises too -- but their two flow
        directions then disagree, so it is the agreement gate, not the
        residual alone, that isolates the true frame."""
        vid, dx, dy = make_video()
        pos = two_point_positions(30, dx, dy)
        pos[15, 0] = pos[15, 0] + [30.0, 0.0]
        fr = fc.flow_residual(pos, vid)
        idx = {int(f): k for k, f in enumerate(fr.frames)}
        assert fr.residual[idx[15], 0] > 25 and fr.agreement[idx[15], 0] < 3.0
        assert fr.agreement[idx[14], 0] > 5.0 or fr.agreement[idx[16], 0] > 5.0

class TestDlcPositions:
    def test_bridges_a_dlc_frame(self):
        import pandas as pd
        cols = pd.MultiIndex.from_product(
            [["sc"], ["point0", "point1"], ["x", "y", "likelihood"]],
            names=["scorer", "bodyparts", "coords"],
        )
        df = pd.DataFrame(
            [[1, 2, 0.9, 3, 4, 0.8], [5, 6, 0.7, 7, 8, 0.6]], columns=cols
        )
        pos, labels = fc.dlc_positions(df)
        assert labels == ["0", "1"]
        assert pos.shape == (2, 2, 2)
        assert pos[0, 0].tolist() == [1, 2] and pos[1, 1].tolist() == [7, 8]
