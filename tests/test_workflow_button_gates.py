"""Tests for the Workflow-group button enable/disable gates.

The gates live in :meth:`DUSTrack._evaluate_workflow_gates`, which is
pulled out of :meth:`_refresh_workflow_button_state` precisely so the
predicates can be unit-tested without standing up a Qt window. Each
test stubs the small amount of state the gates read
(``_dlcproject`` / ``_current_overlay`` / ``ann.name`` / ``fname``)
and asserts ``(enabled, tooltip)`` per button.

The Qt round-trip (``setEnabled`` / ``setToolTip`` actually painting
on the right ``QPushButton``) is left to manual smoke -- matching the
posture documented for ``ConfirmOverlay`` and the pinned palette.
"""
from pathlib import Path

import pytest

from dustrack.dlcinterface import DUSTrack


def _make_dlc_root(folder: Path) -> Path:
    """Mirror the helper used in ``test_open.py`` so the two test
    files don't drift on what "a DLC project root" looks like.
    """
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.yaml").write_text("# fake DLC config\n", encoding="utf-8")
    (folder / "videos").mkdir()
    (folder / "labeled-data").mkdir()
    return folder


class _Ann:
    """Stand-in for an annotation layer with just a ``.name``."""

    def __init__(self, name):
        self.name = name


class _Stub:
    """Stand-in for a DUSTrack instance carrying only the state the
    gate predicates read. We bind :meth:`_evaluate_workflow_gates`
    to it so the method runs against the stub without invoking
    DUSTrack's heavy ``__init__``.
    """

    CORRECTIONS_LAYER_NAME = DUSTrack.CORRECTIONS_LAYER_NAME

    def __init__(
        self,
        fname=None,
        dlcproject=None,
        current_overlay=None,
        active_layer_name=None,
    ):
        self.fname = fname
        self._dlcproject = dlcproject
        self._current_overlay = current_overlay
        self.ann = _Ann(active_layer_name) if active_layer_name else None

    # Bind the actual method so we exercise the real predicate code.
    evaluate = DUSTrack._evaluate_workflow_gates


class _StubProject:
    def __init__(self, config_path):
        self.config_path = config_path


class TestCreateDLCProjectGate:
    def test_bare_video_enabled(self, tmp_path):
        vid = tmp_path / "sample.mp4"
        vid.write_bytes(b"")
        stub = _Stub(fname=str(vid), active_layer_name="manual")
        enabled, tooltip = stub.evaluate()["Create DLC Project"]
        assert enabled is True
        assert tooltip == ""

    def test_video_inside_project_disabled(self, tmp_path):
        root = _make_dlc_root(tmp_path / "proj")
        vid = root / "videos" / "sample.mp4"
        vid.write_bytes(b"")
        stub = _Stub(fname=str(vid), active_layer_name="manual")
        enabled, tooltip = stub.evaluate()["Create DLC Project"]
        assert enabled is False
        assert "Already inside DLC project" in tooltip
        assert "proj" in tooltip  # project root name surfaces in the tip

    def test_dlcproject_set_disables_even_outside_tree(self, tmp_path):
        """Defensive: post-create / Phase-2-open state where
        ``_dlcproject`` is set should disable Create even if the path
        walk-up wouldn't catch it (covers the timing window where
        ``fname`` has been rewired but ``_dlcproject`` arrived first)."""
        root = _make_dlc_root(tmp_path / "proj")
        outside_vid = tmp_path / "elsewhere.mp4"
        outside_vid.write_bytes(b"")
        proj = _StubProject(config_path=str(root / "config.yaml"))
        stub = _Stub(
            fname=str(outside_vid),
            dlcproject=proj,
            active_layer_name="manual",
        )
        enabled, _ = stub.evaluate()["Create DLC Project"]
        assert enabled is False


class TestTrainDLCModelGate:
    def test_no_project_disabled(self, tmp_path):
        vid = tmp_path / "sample.mp4"
        vid.write_bytes(b"")
        stub = _Stub(fname=str(vid), active_layer_name="manual")
        enabled, tooltip = stub.evaluate()["Train DLC model"]
        assert enabled is False
        assert tooltip == "Create a DLC project first."

    def test_project_set_enabled(self, tmp_path):
        root = _make_dlc_root(tmp_path / "proj")
        vid = root / "videos" / "sample.mp4"
        vid.write_bytes(b"")
        proj = _StubProject(config_path=str(root / "config.yaml"))
        stub = _Stub(
            fname=str(vid),
            dlcproject=proj,
            active_layer_name="manual",
        )
        enabled, tooltip = stub.evaluate()["Train DLC model"]
        assert enabled is True
        assert tooltip == ""


class TestApplyManualCorrectionsGate:
    def test_no_overlay_disabled(self, tmp_path):
        vid = tmp_path / "sample.mp4"
        vid.write_bytes(b"")
        stub = _Stub(
            fname=str(vid),
            current_overlay=None,
            active_layer_name="manual",
        )
        enabled, tooltip = stub.evaluate()["Apply manual corrections"]
        assert enabled is False
        assert "overlay layer" in tooltip

    def test_active_is_dlccorr_disabled(self, tmp_path):
        vid = tmp_path / "sample.mp4"
        vid.write_bytes(b"")
        stub = _Stub(
            fname=str(vid),
            current_overlay="dlc_iter1",
            active_layer_name=DUSTrack.CORRECTIONS_LAYER_NAME,
        )
        enabled, tooltip = stub.evaluate()["Apply manual corrections"]
        assert enabled is False
        # Mentions the layer name so the user knows what to change.
        assert "dlccorr" in tooltip

    def test_overlay_set_active_manual_enabled(self, tmp_path):
        vid = tmp_path / "sample.mp4"
        vid.write_bytes(b"")
        stub = _Stub(
            fname=str(vid),
            current_overlay="dlc_iter1",
            active_layer_name="manual",
        )
        enabled, tooltip = stub.evaluate()["Apply manual corrections"]
        assert enabled is True
        assert tooltip == ""


class TestReduceJitterNotGated:
    """Reduce jitter is intentionally NOT in the gates dict; its
    precondition is a data property (every frame fully annotated) and
    the name-based proxy was rejected. This test pins that decision
    so a future refactor doesn't quietly add the wrong gate back.
    """

    def test_reduce_jitter_absent_from_gates(self, tmp_path):
        vid = tmp_path / "sample.mp4"
        vid.write_bytes(b"")
        stub = _Stub(fname=str(vid), active_layer_name="manual")
        gates = stub.evaluate()
        assert "Reduce jitter" not in gates
