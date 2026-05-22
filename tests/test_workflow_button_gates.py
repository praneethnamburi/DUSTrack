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

from dustrack import dlcinterface, dlcloader
from dustrack.dlcinterface import DUSTrack


@pytest.fixture(autouse=True)
def _dlc_loaded():
    """Force the lazy DLC loader to ``"done"`` for the existing gate
    tests, which pre-date the lazy-load refactor and assume DLC is
    available synchronously. The new "Loading DeepLabCut…" gate gets
    dedicated tests in :class:`TestCreateDLCProjectGateDuringLoad`
    that flip the state explicitly.
    """
    prior = dlcloader._DLC_LOAD_STATE
    dlcloader._DLC_LOAD_STATE = "done"
    try:
        yield
    finally:
        dlcloader._DLC_LOAD_STATE = prior


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


class TestCreateDLCProjectGateDuringLoad:
    """Lazy ``import deeplabcut`` (~7 s) runs on a bg thread fired
    from :func:`dustrack.open`. Until the loader resolves, the
    Create DLC Project button stays greyed out with a "Loading
    DeepLabCut…" tooltip -- a subtle "not ready yet" signal using
    the same disabled-button affordance as the other gates.
    """

    @pytest.fixture(autouse=True)
    def _force_pending(self):
        """Override the file-level autouse fixture to flip the
        loader state to ``"pending"`` (the pre-load shape) so the
        gate code exercises the "still loading" branch.
        """
        prior = dlcloader._DLC_LOAD_STATE
        dlcloader._DLC_LOAD_STATE = "pending"
        try:
            yield
        finally:
            dlcloader._DLC_LOAD_STATE = prior

    def test_pending_disables_create_dlc_project(self, tmp_path):
        vid = tmp_path / "sample.mp4"
        vid.write_bytes(b"")
        stub = _Stub(fname=str(vid), active_layer_name="manual")
        enabled, tooltip = stub.evaluate()["Create DLC Project"]
        assert enabled is False
        assert "Loading DeepLabCut" in tooltip

    def test_loading_disables_create_dlc_project(self, tmp_path):
        # Mirror the "loading" state ``_dlc_load_state()`` returns
        # when the bg thread is in flight (``_DLC_LOAD_THREAD`` set,
        # ``_DLC_LOAD_STATE == "pending"``). The gate predicate
        # treats "loading" the same as "pending".
        prior_thread = dlcloader._DLC_LOAD_THREAD
        import threading
        dlcloader._DLC_LOAD_THREAD = threading.Thread(target=lambda: None)
        try:
            vid = tmp_path / "sample.mp4"
            vid.write_bytes(b"")
            stub = _Stub(fname=str(vid), active_layer_name="manual")
            assert dlcinterface._dlc_load_state() == "loading"
            enabled, tooltip = stub.evaluate()["Create DLC Project"]
            assert enabled is False
            assert "Loading DeepLabCut" in tooltip
        finally:
            dlcloader._DLC_LOAD_THREAD = prior_thread

    def test_missing_disables_create_dlc_project(self, tmp_path):
        """``find_spec`` said yes, but the actual import raised
        (broken torch / partial DLC install). The button stays
        greyed with a "DeepLabCut failed to load" tooltip rather
        than letting the click raise a generic ImportError.
        """
        dlcloader._DLC_LOAD_STATE = "missing"
        vid = tmp_path / "sample.mp4"
        vid.write_bytes(b"")
        stub = _Stub(fname=str(vid), active_layer_name="manual")
        enabled, tooltip = stub.evaluate()["Create DLC Project"]
        assert enabled is False
        assert "DeepLabCut failed to load" in tooltip

    def test_project_membership_wins_over_loading(self, tmp_path):
        """When the session is already inside a DLC project, the
        "Already inside" tooltip takes precedence over "Loading…"
        -- the click would refuse on the membership ground first.
        """
        root = _make_dlc_root(tmp_path / "proj")
        vid = root / "videos" / "sample.mp4"
        vid.write_bytes(b"")
        stub = _Stub(fname=str(vid), active_layer_name="manual")
        enabled, tooltip = stub.evaluate()["Create DLC Project"]
        assert enabled is False
        assert "Already inside DLC project" in tooltip
        assert "Loading DeepLabCut" not in tooltip
