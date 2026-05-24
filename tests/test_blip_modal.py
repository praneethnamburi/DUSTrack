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


def test_workflow_mpl_fallback_runs_detect_interpolate_and_adopts(
    tmp_path, example_video
):
    """No Qt window: sync detect + interpolate, sparse layer adopted as active."""
    ann = _make_synthetic_ann_with_blip(tmp_path, example_video)

    adopted = []

    stub = SimpleNamespace(
        ann=ann,
        _find_qt_window=lambda: None,
        _adopt_layer=lambda layer, *, set_active, set_overlay: adopted.append(
            (layer, set_active, set_overlay)
        ),
        update=lambda: None,
    )

    out = DUSTrack.detect_blips_workflow(stub)

    assert out is not None
    assert adopted, "Expected the sparse layer to be adopted"
    (layer, set_active, set_overlay) = adopted[0]
    assert layer is out
    assert set_active is True
    assert set_overlay == ann.name
    # The sparse file should land on disk.
    assert Path(out.fname).exists()


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


def test_gate_enabled_for_dense_layer(monkeypatch):
    monkeypatch.setattr(
        "dustrack._dlc_paths._session_inside_dlc_project", lambda d: None
    )
    monkeypatch.setattr("dustrack.dlcloader._dlc_load_state", lambda: "done")
    gates = evaluate_workflow_gates(
        _stub_dustrack(ann_name="dlc_iter1", coverage=1.0)
    )
    enabled, tooltip = gates["Detect blip outliers"]
    assert enabled is True
    assert tooltip == ""


def test_gate_disabled_for_sparse_layer(monkeypatch):
    monkeypatch.setattr(
        "dustrack._dlc_paths._session_inside_dlc_project", lambda d: None
    )
    monkeypatch.setattr("dustrack.dlcloader._dlc_load_state", lambda: "done")
    gates = evaluate_workflow_gates(
        _stub_dustrack(ann_name="manual", coverage=0.5)
    )
    enabled, tooltip = gates["Detect blip outliers"]
    assert enabled is False
    assert "densely-labeled" in tooltip


def test_gate_disabled_for_blip_corrections_layer(monkeypatch):
    monkeypatch.setattr(
        "dustrack._dlc_paths._session_inside_dlc_project", lambda d: None
    )
    monkeypatch.setattr("dustrack.dlcloader._dlc_load_state", lambda: "done")
    gates = evaluate_workflow_gates(
        _stub_dustrack(ann_name="snapshot_300_blip_corrections", coverage=0.01)
    )
    enabled, tooltip = gates["Detect blip outliers"]
    assert enabled is False
    assert "blip-corrections" in tooltip
