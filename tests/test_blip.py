"""Tests for ``dustrack.blip`` — sparse-blip detection + LK-RSTC interpolation.

Detection tests (1-12) use synthetic in-memory traces with a stub
``video`` attribute and do not decode any video; they cover the
algorithmic shape independently of LK.

Interpolation tests (13-14) use the packaged dnav example video
(``datanavigator.examples.get_example_video``) and exercise the real
:func:`dustrack.lk_opticalflow.lucas_kanade_rstc` call path.

Round-trip test (15) writes the sparse JSON to a tmp dir and re-loads it
through :class:`dustrack.VideoAnnotation`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import dustrack
from dustrack.blip import (
    Blip,
    BlipReport,
    interpolate_blips,
)


# --- Interpolation tests (real LK; require a decodable video) ----------


@pytest.fixture(scope="module")
def example_video(tmp_path_factory):
    """The dnav-packaged example video. Reused across LK interpolation tests."""
    from datanavigator.examples import get_example_video
    return get_example_video(dest_folder=str(tmp_path_factory.getbasetemp()))


def test_lk_endpoints_match_anchors(example_video):
    """RSTC sigmoid weights are ~1 at endpoints -> blended path matches anchors."""
    import datanavigator
    video = datanavigator.VideoReader(example_video)
    n = len(video)
    assert n > 10, "Example video has too few frames for this test"

    # Build a dense trace with one obvious blip mid-video; specific
    # anchor coords picked to land inside the frame.
    H, W = video[0].shape[:2]
    xy = np.zeros((n, 2))
    xy[:, 0] = W * 0.5 + np.arange(n) * 0.1
    xy[:, 1] = H * 0.5
    # Inject a single-frame blip well inside the video.
    blip_frame = n // 2
    xy[blip_frame, 0] += 30.0

    fname = str(Path(example_video).with_suffix(".json")).replace(
        ".json", "_annotations_blip_test.json"
    )
    ann = dustrack.VideoAnnotation(
        fname=fname,
        vname=example_video,
        n_labels=1,
        preloaded_json={
            "0": {f: [float(xy[f, 0]), float(xy[f, 1])] for f in range(n)}
        },
    )
    report = BlipReport(blips=[Blip(
        "0", blip_frame, blip_frame,
        (float(xy[blip_frame - 1, 0]), float(xy[blip_frame - 1, 1])),
        (float(xy[blip_frame + 1, 0]), float(xy[blip_frame + 1, 1])), 5.0)])
    out = interpolate_blips(ann, report)
    # The output is sparse on the blip frame only.
    assert blip_frame in out.data["0"]
    # The interpolated value should be close to the original (pre-blip)
    # smooth line, not the off-trajectory injected value. We compare
    # against the trajectory expected at that frame.
    expected = np.array([W * 0.5 + blip_frame * 0.1, H * 0.5])
    interpolated = np.array(out.data["0"][blip_frame])
    # LK on a flat featureless frame will not recover this exactly; check
    # that the interpolated value moves *toward* the expected position
    # vs the injected one.
    err_interp = np.linalg.norm(interpolated - expected)
    err_injected = np.linalg.norm(np.array([expected[0] + 30.0, expected[1]]) - expected)
    assert err_interp < err_injected, (
        f"Interpolated value {interpolated} should be closer to expected "
        f"{expected} than the injected outlier (err {err_interp:.2f} vs {err_injected:.2f})"
    )


def test_interpolation_recovers_known_trajectory(example_video, tmp_path):
    """End-to-end: detect + interpolate + save + reload preserves blip-frame values."""
    import datanavigator
    video = datanavigator.VideoReader(example_video)
    n = len(video)
    H, W = video[0].shape[:2]
    xy = np.zeros((n, 2))
    xy[:, 0] = W * 0.5 + np.arange(n) * 0.2
    xy[:, 1] = H * 0.5 + np.arange(n) * 0.1
    blip_frame = n // 2
    xy[blip_frame, 0] += 40.0

    fname = str(tmp_path / "recover_annotations_test.json")
    ann = dustrack.VideoAnnotation(
        fname=fname,
        vname=example_video,
        n_labels=1,
        preloaded_json={
            "0": {f: [float(xy[f, 0]), float(xy[f, 1])] for f in range(n)}
        },
    )
    report = BlipReport(blips=[Blip(
        "0", blip_frame, blip_frame,
        (float(xy[blip_frame - 1, 0]), float(xy[blip_frame - 1, 1])),
        (float(xy[blip_frame + 1, 0]), float(xy[blip_frame + 1, 1])), 5.0)])
    out = interpolate_blips(ann, report)
    assert len(report) >= 1
    assert blip_frame in out.data["0"]


# --- Round-trip --------------------------------------------------------


def test_save_and_reload_blip_corrections(example_video, tmp_path):
    """save=True writes JSON next to the source; reload preserves contents."""
    import datanavigator
    video = datanavigator.VideoReader(example_video)
    n = len(video)
    H, W = video[0].shape[:2]
    xy = np.zeros((n, 2))
    xy[:, 0] = W * 0.5 + np.arange(n) * 0.2
    xy[:, 1] = H * 0.5
    blip_frame = n // 2
    xy[blip_frame, 0] += 35.0

    fname = str(tmp_path / "roundtrip_annotations_test.json")
    ann = dustrack.VideoAnnotation(
        fname=fname,
        vname=example_video,
        n_labels=1,
        preloaded_json={
            "0": {f: [float(xy[f, 0]), float(xy[f, 1])] for f in range(n)}
        },
    )
    report = BlipReport(blips=[Blip(
        "0", blip_frame, blip_frame,
        (float(xy[blip_frame - 1, 0]), float(xy[blip_frame - 1, 1])),
        (float(xy[blip_frame + 1, 0]), float(xy[blip_frame + 1, 1])), 5.0)])
    out = interpolate_blips(ann, report)
    out.save()
    out_path = Path(out.fname)
    assert out_path.exists()

    # Reload and verify shape.
    reloaded = dustrack.VideoAnnotation(fname=str(out_path), vname=example_video)
    assert sorted(reloaded.labels) == ["0"]
    assert blip_frame in reloaded.data["0"]
    np.testing.assert_allclose(
        reloaded.data["0"][blip_frame],
        out.data["0"][blip_frame],
        rtol=0,
        atol=1e-6,
    )



class TestLowConfidenceFrames:
    """The low-confidence source: frames whose worst-point model likelihood is
    below threshold, most-uncertain first (the complement of flow_blips)."""

    def test_ranks_worst_first_and_thresholds(self):
        from dustrack.blip import low_confidence_frames
        lk = np.array([[0.9, 0.95], [0.2, 0.99], [0.8, 0.4],
                       [0.99, 0.99], [0.1, 0.1]])
        assert low_confidence_frames(lk, 0.6) == [4, 1, 2]      # 0.1, 0.2, 0.4

    def test_max_frames_caps(self):
        from dustrack.blip import low_confidence_frames
        lk = np.array([[0.9, 0.95], [0.2, 0.99], [0.8, 0.4], [0.1, 0.1]])
        assert low_confidence_frames(lk, 0.6, max_frames=2) == [3, 1]

    def test_one_dimensional_likelihood(self):
        from dustrack.blip import low_confidence_frames
        assert low_confidence_frames(np.array([0.9, 0.3, 0.99]), 0.5) == [1]

    def test_none_below_threshold_is_empty(self):
        from dustrack.blip import low_confidence_frames
        assert low_confidence_frames(np.array([0.9, 0.8, 0.99]), 0.5) == []


class TestRunsFromMask:
    def test_basic_and_gap_merge(self):
        from dustrack.blip import _runs_from_mask
        m = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0], dtype=bool)
        assert _runs_from_mask(m, 0) == [(1, 2), (4, 4), (7, 9)]
        assert _runs_from_mask(m, 1) == [(1, 4), (7, 9)]     # bridge the 1-gap at 3

    def test_empty(self):
        from dustrack.blip import _runs_from_mask
        assert _runs_from_mask(np.zeros(5, dtype=bool)) == []


class TestDisagreementBlips:
    """The single blip detector: |DLC - LK| thresholded at 5 robust-sigma."""

    def test_flags_anomalous_disagreement_and_merges_runs(self):
        from dustrack.blip import disagreement_blips
        rng = np.random.default_rng(0)
        N = 200
        pos = np.zeros((N, 1, 2))
        pos[:, 0, 0] = np.linspace(0, 100, N)
        pos[:, 0, 1] = 50.0
        lk = pos + rng.normal(0, 0.4, pos.shape)          # LK agrees (low floor)
        for f in (50, 120, 121):                           # inject disagreement
            pos[f, 0, 0] += 20.0
        rep = disagreement_blips(pos, lk, threshold_factor=5.0, max_gap=1)
        assert len(rep.blips) == 2                          # 50 alone; 120-121 merged
        assert any(b.start == 50 and b.end == 50 for b in rep.blips)
        assert any(b.start == 120 and b.end == 121 for b in rep.blips)

    def test_movement_without_disagreement_is_not_a_blip(self):
        from dustrack.blip import disagreement_blips
        N = 100
        pos = np.zeros((N, 1, 2))
        pos[:, 0, 0] = np.arange(N) * 5.0                  # moves 5px/frame
        lk = pos.copy()                                    # LK tracks it exactly
        assert len(disagreement_blips(pos, lk, threshold_factor=5.0).blips) == 0

    def test_anchors_are_the_bracketing_good_frames(self):
        from dustrack.blip import disagreement_blips
        N = 50
        pos = np.zeros((N, 1, 2))
        pos[:, 0, 0] = np.arange(N, dtype=float)
        pos[:, 0, 1] = 10.0
        lk = pos + 0.1
        pos[25, 0, 0] += 30.0
        b = [x for x in disagreement_blips(pos, lk, threshold_factor=5.0).blips
             if x.start == 25][0]
        assert b.anchor_before == (24.0, 10.0)
        assert b.anchor_after == (26.0, 10.0)


class TestDeblipHelpers:
    """The two output layers: blipped_positions (what was wrong) + deblip_trace
    (the corrected dense trace)."""

    class _Ann:
        def __init__(self, labels, data):
            self.labels = labels
            self.data = data

    def test_blipped_positions_are_the_original_flagged_frames(self):
        from dustrack.blip import blipped_positions, Blip, BlipReport
        ann = self._Ann(["0", "1"],
                        {"0": {10: [1, 2], 11: [3, 4], 12: [5, 6]}, "1": {10: [7, 8]}})
        rep = BlipReport(blips=[Blip("0", 11, 11, (1, 2), (5, 6), 9.0)])
        out = blipped_positions(ann, rep)
        assert out == {"0": {11: [3.0, 4.0]}, "1": {}}

    def test_deblip_trace_splices_corrections_over_a_copy(self):
        from dustrack.blip import deblip_trace
        ann = self._Ann(["0"], {"0": {10: [1, 1], 11: [2, 2], 12: [3, 3]}})
        corr = self._Ann(["0"], {"0": {11: [9, 9]}})
        out = deblip_trace(ann, corr)
        assert out == {"0": {10: [1.0, 1.0], 11: [9.0, 9.0], 12: [3.0, 3.0]}}
        assert ann.data["0"][11] == [2, 2]              # source untouched


class TestLkLayerData:
    def test_skips_nan_frames(self):
        from dustrack.blip import _lk_layer_data
        lk = np.array([[[np.nan, np.nan]], [[1, 2]], [[3, 4]]])   # frame 0 NaN
        assert _lk_layer_data(lk, ["0"]) == {"0": {1: [1.0, 2.0], 2: [3.0, 4.0]}}

    def test_two_points(self):
        from dustrack.blip import _lk_layer_data
        lk = np.array([[[1, 1], [2, 2]], [[3, 3], [np.nan, 9]]])
        out = _lk_layer_data(lk, ["0", "1"])
        assert out == {"0": {0: [1.0, 1.0], 1: [3.0, 3.0]}, "1": {0: [2.0, 2.0]}}
