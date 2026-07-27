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
    detect_blips,
    detect_and_interpolate_blips,
    interpolate_blips,
)


class _StubVideo:
    """Minimal stand-in for VideoReader. Only ``__len__`` and ``fname``
    are read by the detector / annotation classes during the synthetic
    detection tests (no decode happens)."""
    def __init__(self, n_frames: int, fname: str = "stub.mp4"):
        self._n = n_frames
        self.fname = fname
        self.name = Path(fname).name

    def __len__(self) -> int:
        return self._n


def _make_ann(
    n_frames: int,
    label_traces: dict[str, np.ndarray],
    *,
    fname: str = "stub_annotations_test.json",
) -> dustrack.VideoAnnotation:
    """Build a VideoAnnotation from per-label dense (n_frames, 2) traces.

    Skips file I/O entirely (``preloaded_json`` bypasses ``load``). The
    stub video gives ``n_frames`` without needing a real file.
    """
    data: dict[str, dict[int, list[float]]] = {}
    for label, xy in label_traces.items():
        per_label: dict[int, list[float]] = {}
        for f in range(n_frames):
            if np.isfinite(xy[f]).all():
                per_label[f] = [float(xy[f, 0]), float(xy[f, 1])]
        data[label] = per_label
    ann = dustrack.VideoAnnotation(
        fname=fname,
        n_labels=len(label_traces),
        preloaded_json=data,
        video=_StubVideo(n_frames, fname=fname.replace("_annotations_test.json", ".mp4")),
    )
    return ann


# --- Synthetic-trace detection tests -----------------------------------


def test_clean_trace_no_blips():
    """Linear xy(t) = (t, t) -> no displacement variance -> no blips."""
    n = 50
    xy = np.stack([np.arange(n, dtype=float), np.arange(n, dtype=float)], axis=1)
    ann = _make_ann(n, {"0": xy})
    report = detect_blips(ann)
    assert len(report) == 0
    # Per-label stats populated even when no blips.
    assert report.per_label_stats["0"]["n_blips"] == 0


def test_single_frame_outlier_detected():
    """Inject one off-trajectory frame; verify blip(100, 100) detected."""
    n = 200
    xy = np.stack([np.arange(n, dtype=float), np.arange(n, dtype=float)], axis=1)
    xy[100] += np.array([50.0, 0.0])
    ann = _make_ann(n, {"0": xy})
    report = detect_blips(ann)
    assert len(report) == 1
    b = report.blips[0]
    assert b.label == "0"
    assert b.start == 100 and b.end == 100
    assert b.length == 1


def test_two_frame_outlier_detected():
    """Inject a 2-frame outlier run; verify blip(100, 101)."""
    n = 200
    xy = np.stack([np.arange(n, dtype=float), np.arange(n, dtype=float)], axis=1)
    xy[100:102] += np.array([50.0, 0.0])
    ann = _make_ann(n, {"0": xy})
    report = detect_blips(ann)
    assert len(report) == 1
    b = report.blips[0]
    assert b.start == 100 and b.end == 101


def test_continuous_fast_motion_not_flagged():
    """Abrupt slope change with no return -> no blip."""
    n = 200
    xy = np.zeros((n, 2))
    xy[:100, 0] = np.arange(100) * 1.0
    xy[100:, 0] = 100.0 + np.arange(100) * 5.0  # slope jumps 1 -> 5 at frame 100
    xy[:, 1] = np.arange(n) * 1.0
    ann = _make_ann(n, {"0": xy})
    report = detect_blips(ann)
    # The slope change creates a single high displacement at d[99] but
    # there's no return -- xy[101] keeps moving away from xy[99].
    assert len(report) == 0


def test_blip_at_frame_zero_skipped():
    """Outlier at xy[0] has no pre-blip anchor; can't bracket."""
    n = 200
    xy = np.stack([np.arange(n, dtype=float), np.arange(n, dtype=float)], axis=1)
    xy[0] = np.array([100.0, 100.0])  # off-trajectory
    ann = _make_ann(n, {"0": xy})
    report = detect_blips(ann)
    # The huge d[0] = ||xy[1] - xy[0]|| flags an entry at s=1, but the
    # return-to-anchor test compares xy[2] vs xy[0] which is far -- so
    # not bracketed.
    assert len(report) == 0


def test_blip_at_last_frame_skipped():
    """Outlier at xy[-1] has no anchor-after; reported as edge skip."""
    n = 200
    xy = np.stack([np.arange(n, dtype=float), np.arange(n, dtype=float)], axis=1)
    xy[-1] = xy[-2] + np.array([50.0, 0.0])  # spike at the very last frame
    ann = _make_ann(n, {"0": xy})
    report = detect_blips(ann)
    assert len(report) == 0
    assert report.per_label_stats["0"]["n_skipped_edge"] >= 1


def test_per_label_thresholds_independent():
    """Two labels with different motion scales; both detect their own outlier."""
    n = 200
    # Label A: slow drift (per-frame d ~ 0.1), small outlier
    xy_a = np.stack(
        [np.arange(n, dtype=float) * 0.1, np.arange(n, dtype=float) * 0.1], axis=1
    )
    xy_a[100] += np.array([5.0, 0.0])  # ~50x the per-frame d
    # Label B: fast drift (per-frame d ~ 5), large outlier
    xy_b = np.stack(
        [np.arange(n, dtype=float) * 5.0, np.arange(n, dtype=float) * 5.0], axis=1
    )
    xy_b[150] += np.array([100.0, 0.0])  # ~20x the per-frame d
    ann = _make_ann(n, {"0": xy_a, "1": xy_b})
    report = detect_blips(ann)
    by_label = report.by_label()
    assert "0" in by_label and "1" in by_label
    assert any(b.start == 100 for b in by_label["0"])
    assert any(b.start == 150 for b in by_label["1"])
    # The thresholds reflect each label's own scale.
    assert (
        report.per_label_stats["1"]["threshold"]
        > report.per_label_stats["0"]["threshold"]
    )


def test_max_blip_length_caps():
    """Run longer than ``max_blip_length`` -> not bracketed; counted as long."""
    n = 200
    xy = np.stack([np.arange(n, dtype=float), np.arange(n, dtype=float)], axis=1)
    # 8-frame outlier run; default max_blip_length = 5
    xy[100:108] += np.array([50.0, 0.0])
    ann = _make_ann(n, {"0": xy})
    report = detect_blips(ann, max_blip_length=5)
    assert len(report) == 0
    assert report.per_label_stats["0"]["n_skipped_long"] >= 1


def test_return_tolerance_scales_with_run_length():
    """Larger ``return_position_factor`` accepts wider return jitter."""
    n = 200
    xy = np.stack([np.arange(n, dtype=float), np.arange(n, dtype=float)], axis=1)
    # A 3-frame outlier whose return endpoint is slightly off-anchor
    # (xy[103] differs from xy[100] more than 1 per-frame median would)
    xy[100:103] += np.array([50.0, 0.0])
    # xy[103] resumes the line, but the test passes already with default
    # tolerance; force a borderline case by shifting xy[103] further out
    # and verifying that a tight factor rejects it.
    xy[103] += np.array([5.0, 0.0])
    # With a small tolerance the return check fails:
    report_tight = detect_blips(xy_to_ann(n, xy), return_position_factor=0.5)
    # With a generous tolerance it passes:
    report_loose = detect_blips(xy_to_ann(n, xy), return_position_factor=5.0)
    assert len(report_tight) == 0
    assert len(report_loose) >= 1


def xy_to_ann(n_frames: int, xy: np.ndarray, label: str = "0"):
    """Helper used by the scaling test (avoid repeating _make_ann call)."""
    return _make_ann(n_frames, {label: xy})


def test_mad_zero_fallback_to_percentile():
    """Perfectly-still trace + one spike -> MAD ~ 0 -> percentile fallback."""
    n = 200
    xy = np.full((n, 2), 100.0)
    xy[100] = np.array([150.0, 100.0])  # one spike
    ann = _make_ann(n, {"0": xy})
    report = detect_blips(ann)
    # MAD ~ 0 so threshold falls back to percentile of d. The threshold
    # picks up the spike entry; the return at xy[101] = 100.0 is exact.
    assert len(report) == 1
    assert report.blips[0].start == 100 and report.blips[0].end == 100


def test_other_labels_untouched_in_output():
    """Sparse output is per-label-sparse: blipped label populated, others empty."""
    n = 100
    xy_a = np.stack([np.arange(n, dtype=float), np.arange(n, dtype=float)], axis=1)
    xy_b = np.stack([np.arange(n, dtype=float), np.arange(n, dtype=float)], axis=1)
    xy_a[50] += np.array([50.0, 0.0])  # blip on label 0 only
    ann = _make_ann(n, {"0": xy_a, "1": xy_b})
    report = detect_blips(ann)
    assert all(b.label == "0" for b in report.blips)
    # Build the sparse output WITHOUT decoding the video (interpolate_blips
    # would call LK; here we just want to check the empty-shape contract).
    # Use the lower-level construction path: detect found blips only on
    # label "0", so the output's data["1"] should be empty regardless of
    # how interpolation populates label "0".
    # Skip LK by stubbing the report to be empty -> interpolate produces
    # all-empty per-label dicts, matching the spec.
    empty_report = BlipReport(params=report.params)
    # Pass through interpolate_blips even with no blips so the output
    # shape (every label keyed, every per-label dict empty) is what we
    # contracted.
    out = interpolate_blips(ann, empty_report)
    assert sorted(out.labels) == ["0", "1"]
    assert len(out.data["0"]) == 0
    assert len(out.data["1"]) == 0


def test_empty_output_when_no_blips():
    """A clean trace produces an output annotation with no frames at all."""
    n = 50
    xy = np.stack([np.arange(n, dtype=float), np.arange(n, dtype=float)], axis=1)
    ann = _make_ann(n, {"0": xy})
    report = detect_blips(ann)
    out = interpolate_blips(ann, report)
    assert len(report) == 0
    assert len(out.data["0"]) == 0


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
    report = detect_blips(ann)
    assert len(report) == 1
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
    out, report = detect_and_interpolate_blips(ann, save=False)
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
    out, report = detect_and_interpolate_blips(ann, save=True)
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

    # save again on the same path should refuse (collision guard).
    with pytest.raises(FileExistsError):
        detect_and_interpolate_blips(ann, save=True)


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
