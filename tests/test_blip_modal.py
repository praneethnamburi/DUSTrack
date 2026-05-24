"""Tests for the Detect-blip-outliers UI surface.

Covers:
* The pure-data ``_format_blip_results_text`` modal renderer.
* The ``DUSTrack.detect_blips_workflow`` mpl-fallback path (no Qt,
  default knobs, synchronous detect + interpolate + adopt).
* ``interpolate_blips`` progress-callback semantics.
* The ``Detect blip outliers`` workflow gate (enabled / sparse-layer /
  blip-corrections / no-active-layer).

Live Qt modal exec is not tested (synchronous modals are painful
headless; mirrors the precedent for ``TrainingOptionsDialog`` and
``ConfirmOverlay`` per ``tests/test_save_on_close.py``). Manual smoke
at ``tests/qt_learning/31_blip_modal_smoke.py``.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import dustrack
from dustrack import blip as _blip
from dustrack._overlays import _format_blip_results_text
from dustrack._workflow_gates import _min_label_coverage, evaluate_workflow_gates
from dustrack.gui import DUSTrack


# ---------------------------------------------------------------------
# _format_blip_results_text
# ---------------------------------------------------------------------


class TestFormatBlipResultsText:
    def test_none_report_returns_press_detect_hint(self):
        out = _format_blip_results_text(None)
        assert "Detect" in out

    def test_empty_report_still_shows_per_label_threshold(self):
        report = _blip.BlipReport(
            per_label_stats={
                "0": {"median_d": 0.1, "threshold": 1.5, "n_blips": 0},
            },
            params={},
            n_frames=100,
        )
        out = _format_blip_results_text(report)
        assert "0 blips found" in out
        assert "'0'" in out
        assert "threshold=1.500" in out

    def test_populated_report_renders_counts_histogram_and_skip_line(self):
        b0 = _blip.Blip(label="0", start=10, end=10, anchor_before=(0, 0), anchor_after=(1, 1), threshold=1.0)
        b1 = _blip.Blip(label="0", start=20, end=22, anchor_before=(0, 0), anchor_after=(1, 1), threshold=1.0)
        report = _blip.BlipReport(
            blips=[b0, b1],
            per_label_stats={
                "0": {
                    "median_d": 0.5, "threshold": 2.5, "n_blips": 2,
                    "n_skipped_edge": 1, "n_skipped_long": 3, "n_skipped_noreturn": 0,
                },
            },
            params={},
            n_frames=100,
        )
        out = _format_blip_results_text(report)
        assert "Label '0':  2 blips" in out
        assert "Total: 2 blips" in out
        assert "1:1" in out  # length histogram: one length-1 blip
        assert "3:1" in out  # length histogram: one length-3 blip
        assert "skipped:" in out and "edge=1" in out and "long=3" in out


# ---------------------------------------------------------------------
# detect_blips_workflow -- mpl-fallback path (no Qt window)
# ---------------------------------------------------------------------


def _make_synthetic_ann_with_blip(tmp_path, example_video):
    """A real VideoAnnotation backed by the dnav example video, with
    one injected single-frame blip mid-trajectory."""
    import datanavigator
    video = datanavigator.VideoReader(example_video)
    n = len(video)
    H, W = video[0].shape[:2]
    xy = np.zeros((n, 2))
    xy[:, 0] = W * 0.5 + np.arange(n) * 0.2
    xy[:, 1] = H * 0.5
    blip_frame = n // 2
    xy[blip_frame, 0] += 35.0
    fname = str(tmp_path / "wf_annotations_test.json")
    return dustrack.VideoAnnotation(
        fname=fname,
        vname=example_video,
        n_labels=1,
        preloaded_json={
            "0": {f: [float(xy[f, 0]), float(xy[f, 1])] for f in range(n)}
        },
    )


@pytest.fixture(scope="module")
def example_video(tmp_path_factory):
    from datanavigator.examples import get_example_video
    return get_example_video(dest_folder=str(tmp_path_factory.getbasetemp()))


def test_workflow_mpl_fallback_runs_detect_remove_and_adopts(
    tmp_path, example_video
):
    """No Qt window: sync detect + remove_blips, without-blip layer adopted as active."""
    ann = _make_synthetic_ann_with_blip(tmp_path, example_video)

    adopted = []

    stub = SimpleNamespace(
        ann=ann,
        annotations=SimpleNamespace(names=[]),
        _find_qt_window=lambda: None,
        _adopt_layer=lambda layer, *, set_active, set_overlay: adopted.append(
            (layer, set_active, set_overlay)
        ),
        update=lambda: None,
    )

    out = DUSTrack.detect_blips_workflow(stub)

    assert out is not None
    assert adopted, "Expected the without-blip layer to be adopted"
    (layer, set_active, set_overlay) = adopted[0]
    assert layer is out
    assert set_active is True
    assert set_overlay == ann.name
    # The without-blip file should land on disk with the new suffix.
    assert Path(out.fname).exists()
    assert out.fname.endswith("_blip_removed.json")
    # The without-blip layer is dense: most frames preserved, only blip
    # frames missing.
    assert len(out.data["0"]) > 0
    assert len(out.data["0"]) < ann.n_frames  # at least one blip dropped


def test_workflow_mpl_fallback_reloads_existing_removed_layer(
    tmp_path, example_video
):
    """Re-run with an already-loaded without-blip layer: reload() must be
    called on the existing in-session annotation before _adopt_layer, or
    the user sees stale data (the disk file is fresh but the in-memory
    object holds the prior run's data).
    """
    ann = _make_synthetic_ann_with_blip(tmp_path, example_video)

    # Stand-in "already loaded" without-blip layer that records reload() calls.
    reload_calls = []
    existing = SimpleNamespace(reload=lambda: reload_calls.append(True))
    from dustrack._file_management import VideoFileManager

    # Run once to produce a real without-blip file so we know its layer name.
    first_stub = SimpleNamespace(
        ann=ann,
        annotations=SimpleNamespace(names=[]),
        _find_qt_window=lambda: None,
        _adopt_layer=lambda *a, **k: None,
        update=lambda: None,
    )
    first_out = DUSTrack.detect_blips_workflow(first_stub)
    assert first_out is not None
    output_layer_name = VideoFileManager.canonical_layer_name(first_out.fname)

    # Now simulate the second run, with that layer "already loaded".
    class _StubAnnotations:
        def __init__(self, name, ann):
            self.names = [name]
            self._map = {name: ann}

        def __getitem__(self, key):
            return self._map[key]

    second_stub = SimpleNamespace(
        ann=ann,
        annotations=_StubAnnotations(output_layer_name, existing),
        _find_qt_window=lambda: None,
        _adopt_layer=lambda *a, **k: None,
        update=lambda: None,
    )
    second_out = DUSTrack.detect_blips_workflow(second_stub)
    assert second_out is not None
    assert reload_calls == [True], (
        "Expected existing in-session without-blip layer to be reloaded; "
        "without it, the trace pane shows stale data."
    )


def test_workflow_mpl_fallback_short_circuits_when_no_blips(tmp_path, example_video):
    """No blips found -> nothing adopted, returns None, no file written."""
    import datanavigator
    video = datanavigator.VideoReader(example_video)
    n = len(video)
    H, W = video[0].shape[:2]
    # Clean linear trace, no spike.
    xy = np.zeros((n, 2))
    xy[:, 0] = W * 0.5 + np.arange(n) * 0.2
    xy[:, 1] = H * 0.5
    fname = str(tmp_path / "clean_annotations_test.json")
    ann = dustrack.VideoAnnotation(
        fname=fname,
        vname=example_video,
        n_labels=1,
        preloaded_json={
            "0": {f: [float(xy[f, 0]), float(xy[f, 1])] for f in range(n)}
        },
    )

    adopted = []
    stub = SimpleNamespace(
        ann=ann,
        annotations=SimpleNamespace(names=[]),
        _find_qt_window=lambda: None,
        _adopt_layer=lambda *a, **k: adopted.append(True),
        update=lambda: None,
    )
    out = DUSTrack.detect_blips_workflow(stub)
    assert out is None
    assert not adopted


# ---------------------------------------------------------------------
# interpolate_blips progress_callback
# ---------------------------------------------------------------------


def test_progress_callback_fires_once_per_blip(tmp_path, example_video):
    """Callback receives ``(done, total)`` once per completed blip."""
    import datanavigator
    video = datanavigator.VideoReader(example_video)
    n = len(video)
    H, W = video[0].shape[:2]
    xy = np.zeros((n, 2))
    xy[:, 0] = W * 0.5 + np.arange(n) * 0.2
    xy[:, 1] = H * 0.5
    # Two blips so total > 1.
    xy[n // 3, 0] += 30.0
    xy[(2 * n) // 3, 0] += 30.0
    fname = str(tmp_path / "two_blip_annotations_test.json")
    ann = dustrack.VideoAnnotation(
        fname=fname,
        vname=example_video,
        n_labels=1,
        preloaded_json={
            "0": {f: [float(xy[f, 0]), float(xy[f, 1])] for f in range(n)}
        },
    )
    report = _blip.detect_blips(ann)
    assert len(report) >= 2  # detector should pick both up

    fires: list = []
    _blip.interpolate_blips(ann, report, progress_callback=lambda d, t: fires.append((d, t)))
    assert len(fires) == len(report)
    # Monotonic done count from 1 to total.
    assert [d for d, _ in fires] == list(range(1, len(report) + 1))
    # Total is constant.
    assert {t for _, t in fires} == {len(report)}


# ---------------------------------------------------------------------
# _min_label_coverage helper + workflow gate
# ---------------------------------------------------------------------


class TestMinLabelCoverage:
    def test_empty_ann_returns_zero(self):
        ann = SimpleNamespace(n_frames=0, labels=[], data={})
        assert _min_label_coverage(ann) == 0.0

    def test_dense_ann_returns_one(self):
        ann = SimpleNamespace(
            n_frames=10,
            labels=["0", "1"],
            data={"0": {f: [0, 0] for f in range(10)}, "1": {f: [0, 0] for f in range(10)}},
        )
        assert _min_label_coverage(ann) == 1.0

    def test_returns_min_across_labels(self):
        ann = SimpleNamespace(
            n_frames=10,
            labels=["0", "1"],
            data={"0": {f: [0, 0] for f in range(10)}, "1": {0: [0, 0]}},
        )
        assert _min_label_coverage(ann) == 0.1


class _StubDUSTrack(SimpleNamespace):
    """Carries the class-level ``CORRECTIONS_LAYER_NAME`` attribute the
    gate evaluator reads via ``type(dustrack).CORRECTIONS_LAYER_NAME``;
    SimpleNamespace alone won't expose class attributes."""

    CORRECTIONS_LAYER_NAME = "dlccorr"


def _stub_dustrack(*, ann_name=None, coverage=1.0, dlcproject=None, overlay=None, fname="x.mp4"):
    """Minimal dustrack-shape stub for the gate evaluator."""
    n_frames = 10
    labels = ["0"]
    # Build per-label data matching the requested coverage.
    n_per_label = int(round(coverage * n_frames))
    data = {label: {f: [0.0, 0.0] for f in range(n_per_label)} for label in labels}
    ann = SimpleNamespace(name=ann_name, n_frames=n_frames, labels=labels, data=data)
    return _StubDUSTrack(
        ann=ann,
        _dlcproject=dlcproject,
        _current_overlay=overlay,
        fname=fname,
    )


@pytest.fixture
def _patch_dlc_state(monkeypatch):
    """Patches the gates module's outside-world reads so the test stub
    doesn't need to model a full session."""
    monkeypatch.setattr(
        "dustrack._dlc_paths._session_inside_dlc_project", lambda d: None
    )
    monkeypatch.setattr("dustrack.dlcloader._dlc_load_state", lambda: "done")


def test_gate_enabled_for_dlc_layer(_patch_dlc_state):
    """Layer name with ``dlc_*`` prefix is the canonical enable case."""
    gates = evaluate_workflow_gates(
        _stub_dustrack(ann_name="dlc_iteration-0_snapshot_300", coverage=1.0)
    )
    enabled, tooltip = gates["Detect blip outliers"]
    assert enabled is True
    assert tooltip == ""


def test_gate_enabled_for_dlccorr_layer(_patch_dlc_state):
    """Apply-manual-corrections output is dense; enable."""
    gates = evaluate_workflow_gates(
        _stub_dustrack(ann_name="dlccorr", coverage=1.0)
    )
    enabled, _tooltip = gates["Detect blip outliers"]
    assert enabled is True


def test_gate_enabled_for_lkmovavg_layer(_patch_dlc_state):
    """Jitter-reduced layers (any source) carry the ``lkmovavg``
    substring and are dense; enable."""
    gates = evaluate_workflow_gates(
        _stub_dustrack(ann_name="dlccorr_lkmovavg_0.500", coverage=1.0)
    )
    enabled, _tooltip = gates["Detect blip outliers"]
    assert enabled is True


def test_gate_disabled_for_manual_layer(_patch_dlc_state):
    """Manual annotation layers (typical name: ``iteration-N`` or
    user-chosen) are sparse; disable with the dense-layer hint."""
    gates = evaluate_workflow_gates(
        _stub_dustrack(ann_name="iteration-4", coverage=0.05)
    )
    enabled, tooltip = gates["Detect blip outliers"]
    assert enabled is False
    assert "dense" in tooltip
    assert "dlc_" in tooltip


def test_gate_disabled_for_blip_corrections_layer(_patch_dlc_state):
    """Sparse LK-corrections output: disable so re-detection on the
    output doesn't self-collide on the output filename."""
    gates = evaluate_workflow_gates(
        _stub_dustrack(
            ann_name="snapshot_300_blip_corrections", coverage=0.01
        )
    )
    enabled, tooltip = gates["Detect blip outliers"]
    assert enabled is False
    assert "blip detection" in tooltip


def test_gate_disabled_for_blip_removed_layer(_patch_dlc_state):
    """Dense without-blip output: disable so re-detection on the
    output doesn't self-collide on the output filename."""
    gates = evaluate_workflow_gates(
        _stub_dustrack(ann_name="dlc_iteration-0_removed", coverage=1.0)
    )
    enabled, _tooltip = gates["Detect blip outliers"]
    # The name-only path doesn't end in `_blip_removed` (canonical_layer_name
    # strips the suffix on DLC-stem-derived layers), so we exercise the
    # fname-based check.
    _stub_with_fname = _stub_dustrack(ann_name="dlc_iteration-0_removed")
    _stub_with_fname.ann.fname = "C:/proj/video_blip_removed.json"
    gates = evaluate_workflow_gates(_stub_with_fname)
    enabled, tooltip = gates["Detect blip outliers"]
    assert enabled is False
    assert "blip detection" in tooltip


# ---------------------------------------------------------------------
# remove_blips behaviour
# ---------------------------------------------------------------------


def _make_dense_two_label_ann(tmp_path, example_video):
    """Two-label dense annotation with one blip per label at different frames."""
    import datanavigator
    video = datanavigator.VideoReader(example_video)
    n = len(video)
    H, W = video[0].shape[:2]
    xy0 = np.zeros((n, 2))
    xy0[:, 0] = W * 0.4 + np.arange(n) * 0.2
    xy0[:, 1] = H * 0.5
    xy1 = np.zeros((n, 2))
    xy1[:, 0] = W * 0.6 + np.arange(n) * 0.2
    xy1[:, 1] = H * 0.5
    # Distinct blip frames: label 0 blips at frame n//3; label 1 at 2n//3.
    blip_a = n // 3
    blip_b = (2 * n) // 3
    xy0[blip_a, 0] += 40.0
    xy1[blip_b, 0] += 40.0
    fname = str(tmp_path / "two_label_dense_annotations_test.json")
    ann = dustrack.VideoAnnotation(
        fname=fname,
        vname=example_video,
        n_labels=2,
        preloaded_json={
            "0": {f: [float(xy0[f, 0]), float(xy0[f, 1])] for f in range(n)},
            "1": {f: [float(xy1[f, 0]), float(xy1[f, 1])] for f in range(n)},
        },
    )
    return ann, blip_a, blip_b


def test_remove_blips_per_label_only_preserves_other_labels(
    tmp_path, example_video
):
    """Default (drop_frame_if_any_blip=False): the blipped label's
    entry at the blip frame is removed, but other labels at the same
    frame are preserved."""
    ann, blip_a, blip_b = _make_dense_two_label_ann(tmp_path, example_video)
    report = _blip.detect_blips(ann)
    # The test data is constructed so blips exist; if the detector fails
    # to find them at the default knobs, the test premise is broken.
    by_label = report.by_label()
    assert "0" in by_label and "1" in by_label

    out = _blip.remove_blips(ann, report, drop_frame_if_any_blip=False)
    # Label 0 should have its blip frame(s) dropped; label 1's frames
    # corresponding to label 0's blips should be untouched.
    label0_blip_frames = {
        f for b in by_label["0"] for f in range(b.start, b.end + 1)
    }
    for f in label0_blip_frames:
        assert f not in out.data["0"]
        # Label 1 at the same frame is preserved.
        assert f in out.data["1"]


def test_remove_blips_drop_frame_removes_all_labels_at_blip_frame(
    tmp_path, example_video
):
    """drop_frame_if_any_blip=True: a blip on any label removes every
    label's entry at that frame."""
    ann, blip_a, blip_b = _make_dense_two_label_ann(tmp_path, example_video)
    report = _blip.detect_blips(ann)
    by_label = report.by_label()
    all_blip_frames = {
        f for b in report.blips for f in range(b.start, b.end + 1)
    }

    out = _blip.remove_blips(ann, report, drop_frame_if_any_blip=True)
    for f in all_blip_frames:
        for label in ann.labels:
            assert f not in out.data[label], (
                f"Frame {f} should be dropped from label {label!r} under "
                f"drop_frame_if_any_blip=True"
            )


def test_remove_blips_preserves_non_blip_frames(tmp_path, example_video):
    """Frames untouched by any blip retain their original (x, y)."""
    ann, _, _ = _make_dense_two_label_ann(tmp_path, example_video)
    report = _blip.detect_blips(ann)
    all_blip_frames = {
        f for b in report.blips for f in range(b.start, b.end + 1)
    }

    out = _blip.remove_blips(ann, report)
    # Pick a couple of "clearly clean" frames -- well away from any blip --
    # and assert their values pass through unchanged.
    n = ann.n_frames
    clean_frames = [f for f in (n // 4, n // 2, (3 * n) // 4) if f not in all_blip_frames]
    assert clean_frames, "Test premise: expected at least one clean test frame"
    for f in clean_frames:
        for label in ann.labels:
            np.testing.assert_allclose(
                out.data[label][f], ann.data[label][f], rtol=0, atol=0
            )
