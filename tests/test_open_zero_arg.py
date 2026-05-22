"""Tests for the 1.2.0a3 no-arg / list-form :func:`dustrack.open` surface.

The actual Qt picker exec is mocked out -- the modal-loop spin is
manual-smoke territory (same convention used for ConfirmOverlay and
the training-options modal). What's exercised here:

- ``_prompt_for_videos`` cancel vs success return shapes.
- ``open()`` with ``path=None`` invokes the picker, returns ``None``
  on cancel, otherwise threads the picked list through the existing
  dispatch.
- List-form ``open([p])`` is dispatch-equivalent to ``open(p)``.
- Multi-element ``open([p0, p1, ...])`` lands ``[p1, ...]`` on
  ``tracker._video_queue`` as ``Path`` objects.
- ``str`` and ``Path`` entries coexist in a list call.
- Empty sequence raises ``ValueError``.

Phase 2 (DLC project) routing is left alone in these tests -- the
queue/picker plumbing sits at the front of ``open()`` and doesn't care
which phase wins downstream.
"""
from pathlib import Path

import pytest

import dustrack
from dustrack.dlcinterface import _prompt_for_videos


class _FakeTracker:
    """Setattr-friendly stand-in for the DUSTrack constructor result."""


class _CapturingConstructor:
    """A ``DUSTrack`` replacement that records its call args and returns
    a fresh ``_FakeTracker`` each invocation. Lets tests assert which
    path drove dispatch without spinning up Qt."""

    def __init__(self):
        self.calls = []
        self.trackers = []

    def __call__(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        tracker = _FakeTracker()
        self.trackers.append(tracker)
        return tracker


@pytest.fixture
def fake_dustrack(monkeypatch):
    """Patch ``dustrack.dlcinterface.DUSTrack`` with a capturing stub."""
    stub = _CapturingConstructor()
    monkeypatch.setattr("dustrack.dlcinterface.DUSTrack", stub)
    return stub


# ---------------------------------------------------------------------
# _prompt_for_videos
# ---------------------------------------------------------------------


class TestPromptForVideos:
    """Cover the two return-shape contracts: ``list[Path]`` on success,
    ``None`` on cancel."""

    def test_cancel_returns_none(self, monkeypatch):
        # ``QFileDialog.getOpenFileNames`` returns ``([], '')`` on cancel.
        def _fake_get_open_file_names(parent, caption, directory, filter_):
            return ([], "")

        try:
            from qtpy.QtWidgets import QFileDialog
        except ImportError:
            pytest.skip("qtpy not installed in this env")

        monkeypatch.setattr(
            QFileDialog, "getOpenFileNames", staticmethod(_fake_get_open_file_names)
        )

        assert _prompt_for_videos() is None

    def test_success_returns_list_of_paths(self, monkeypatch, tmp_path):
        v0 = tmp_path / "a.mp4"
        v1 = tmp_path / "b.mp4"
        v0.write_bytes(b"")
        v1.write_bytes(b"")

        def _fake_get_open_file_names(parent, caption, directory, filter_):
            return ([str(v0), str(v1)], "Videos (*.mp4 ...)")

        try:
            from qtpy.QtWidgets import QFileDialog
        except ImportError:
            pytest.skip("qtpy not installed in this env")

        monkeypatch.setattr(
            QFileDialog, "getOpenFileNames", staticmethod(_fake_get_open_file_names)
        )

        result = _prompt_for_videos()
        assert result == [v0, v1]
        assert all(isinstance(p, Path) for p in result)


# ---------------------------------------------------------------------
# open(path=None) -- legacy fallback path (no seed modal available)
#
# These tests force ``_open_seed_session`` to return None, which
# triggers the pre-1.2.0a3 direct-picker fallback path inside
# :func:`dustrack.open`. The 1.2.0a3 seed-modal happy path is covered
# in TestOpenNoArgSeedModal below.
# ---------------------------------------------------------------------


@pytest.fixture
def force_legacy_fallback(monkeypatch):
    """Make :func:`dustrack.open` skip the seed-modal path and hit the
    legacy ``_prompt_for_videos`` fallback. Used by the tests that
    pre-date the seed-modal cut to keep their direct-picker contract
    intact under the new launch flow."""
    monkeypatch.setattr(
        "dustrack.dlcinterface._open_seed_session",
        lambda **_: None,
    )


class TestOpenNoArg:
    def test_picker_cancel_returns_none(
        self, monkeypatch, fake_dustrack, force_legacy_fallback,
    ):
        monkeypatch.setattr(
            "dustrack.dlcinterface._prompt_for_videos", lambda parent=None: None
        )
        result = dustrack.open()
        assert result is None
        # No constructor call should have happened.
        assert fake_dustrack.calls == []

    def test_picker_single_pick_dispatches_phase1(
        self, monkeypatch, tmp_path, fake_dustrack, force_legacy_fallback,
    ):
        vid = tmp_path / "ok.mp4"
        vid.write_bytes(b"")
        monkeypatch.setattr(
            "dustrack.dlcinterface._prompt_for_videos",
            lambda parent=None: [vid],
        )

        tracker = dustrack.open()

        assert tracker is not None
        assert len(fake_dustrack.calls) == 1
        # ``DUSTrack(str(p), layer_name, **kwargs)`` -- first positional
        # is the stringified path, second is the resolved layer name.
        args = fake_dustrack.calls[0]["args"]
        assert args[0] == str(vid)
        assert args[1] == "iteration-0"
        # Single-pick path -- queue is empty.
        assert tracker._video_queue == []

    def test_picker_multi_pick_bare_videos_rejected(
        self, monkeypatch, tmp_path, fake_dustrack, force_legacy_fallback,
    ):
        """1.2.0a3 contract: multi-video sessions require every entry
        to belong to a single DLC project. Bare videos in tmp_path
        have no DLC project around them, so a multi-pick of bare
        videos must raise.
        """
        v0 = tmp_path / "a.mp4"
        v1 = tmp_path / "b.mp4"
        v2 = tmp_path / "c.mp4"
        for v in (v0, v1, v2):
            v.write_bytes(b"")
        monkeypatch.setattr(
            "dustrack.dlcinterface._prompt_for_videos",
            lambda parent=None: [v0, v1, v2],
        )

        with pytest.raises(ValueError, match="not inside a DLC project"):
            dustrack.open()
        # No construction should have happened.
        assert fake_dustrack.calls == []


# ---------------------------------------------------------------------
# 1.2.0a3 seed-modal happy paths.
#
# These tests cover the new launch flow: ``dustrack.open()`` constructs
# a seed-mode DUSTrack against the packaged synthetic seed video and
# mounts the welcome modal on top. The modal's exec_() is the surface
# under test -- we mock the overlay factory to return a stub whose
# ``exec_()`` returns the picked list (or None on dismiss).
# ---------------------------------------------------------------------


class _StubOpenVideoOverlay:
    """Minimal stand-in for :class:`OpenVideoOverlay` used in tests.
    The class produced by :func:`_make_open_video_overlay_class` exec
    blocks on a Qt event loop; the stub returns its ``_result`` value
    immediately.
    """

    def __init__(self, main_window, *, recent_sessions, _stub_result):
        self.main_window = main_window
        self.recent_sessions = list(recent_sessions or [])
        self._stub_result = _stub_result

    def exec_(self):
        return self._stub_result


def _patch_overlay_factory(monkeypatch, *, picked):
    """Patch ``_make_open_video_overlay_class`` so the seed-modal
    flow returns the ``picked`` list (or None on dismiss) without
    spinning up Qt. Returns the captured (main_window, recent)
    arguments via a single dict mutated in-place.
    """
    captured = {}

    def _stub_factory():
        def _stub_init(main_window, *, recent_sessions):
            captured["main_window"] = main_window
            captured["recent_sessions"] = recent_sessions
            return _StubOpenVideoOverlay(
                main_window,
                recent_sessions=recent_sessions,
                _stub_result=picked,
            )
        return _stub_init

    monkeypatch.setattr(
        "dustrack.dlcinterface._make_open_video_overlay_class", _stub_factory
    )
    return captured


@pytest.fixture
def stub_seed_session(monkeypatch):
    """Replace :func:`_open_seed_session` with a stub that returns a
    minimal fake tracker capable of standing in for the real
    seed-mode DUSTrack. Tests assert against the captured tracker's
    ``replace_active_with`` calls.
    """
    class _SeedFake:
        def __init__(self):
            self._is_seed_session = True
            self.figure = type("_F", (), {"__init__": lambda s: None})()
            self.replace_called_with = None
            self.replace_kwargs = None

        def _find_qt_window(self):
            return type("_QW", (), {})()  # truthy stub

        def replace_active_with(self, picked, *, layer_name=None, **kw):
            self.replace_called_with = picked
            self.replace_kwargs = {"layer_name": layer_name, **kw}
            return [0]

    seed = _SeedFake()
    monkeypatch.setattr(
        "dustrack.dlcinterface._open_seed_session",
        lambda **_: seed,
    )
    # Stub ``plt.close`` so the test doesn't try to interact with
    # matplotlib on the seed-fake's stub figure.
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)
    return seed


class TestOpenNoArgSeedModal:
    def test_modal_cancel_returns_none(
        self, monkeypatch, stub_seed_session,
    ):
        _patch_overlay_factory(monkeypatch, picked=None)
        result = dustrack.open()
        assert result is None
        # The seed-fake was constructed but ``replace_active_with``
        # was never called (modal cancel exits before the swap).
        assert stub_seed_session.replace_called_with is None

    def test_modal_pick_routes_through_replace_active_with(
        self, monkeypatch, stub_seed_session, tmp_path,
    ):
        vid = tmp_path / "picked.mp4"
        vid.write_bytes(b"")
        _patch_overlay_factory(monkeypatch, picked=[vid])
        tracker = dustrack.open()
        assert tracker is stub_seed_session
        # The pick was forwarded to replace_active_with as-is.
        assert stub_seed_session.replace_called_with == [vid]
        # _is_seed_session flipped to False before the swap.
        assert stub_seed_session._is_seed_session is False

    def test_modal_pick_threads_layer_name_kwarg(
        self, monkeypatch, stub_seed_session, tmp_path,
    ):
        vid = tmp_path / "picked.mp4"
        vid.write_bytes(b"")
        _patch_overlay_factory(monkeypatch, picked=[vid])
        dustrack.open(layer_name="custom_layer")
        assert stub_seed_session.replace_kwargs["layer_name"] == "custom_layer"

    def test_modal_recent_sessions_threaded_from_config(
        self, monkeypatch, stub_seed_session, tmp_path,
    ):
        # Pre-seed the config so recent_sessions is non-empty when the
        # modal is constructed. Isolate the config so we don't pollute
        # the real ~/.dustrack/config.json.
        from dustrack import _config
        cfg_dir = tmp_path / ".dustrack"
        cfg_path = cfg_dir / "config.json"
        monkeypatch.setattr(_config, "_USER_CONFIG_DIR", cfg_dir)
        monkeypatch.setattr(_config, "_USER_CONFIG_PATH", cfg_path)
        existing = tmp_path / "existing.mp4"
        existing.write_bytes(b"")
        _config.record_recent_session([existing])

        captured = _patch_overlay_factory(monkeypatch, picked=None)
        dustrack.open()
        # The modal received the unified recent-sessions list.
        recent = captured["recent_sessions"]
        assert len(recent) == 1
        assert recent[0] == [existing.resolve()]


# ---------------------------------------------------------------------
# open([...]) -- list-form dispatch
# ---------------------------------------------------------------------


class TestOpenListForm:
    def test_single_element_list_equiv_to_scalar(
        self, tmp_path, fake_dustrack
    ):
        vid = tmp_path / "ok.mp4"
        vid.write_bytes(b"")

        tracker_scalar = dustrack.open(vid)
        tracker_list = dustrack.open([vid])

        # Both constructions saw the same first positional + layer name.
        assert fake_dustrack.calls[0]["args"] == fake_dustrack.calls[1]["args"]
        # Both queues are empty.
        assert tracker_scalar._video_queue == []
        assert tracker_list._video_queue == []

    def test_multi_element_list_bare_videos_rejected(
        self, tmp_path, fake_dustrack
    ):
        """1.2.0a3: bare-video multi-video lists raise (must share a
        single DLC project). See ``test_open_multi_video.py`` for the
        positive Phase 2 path."""
        v0 = tmp_path / "a.mp4"
        v1 = tmp_path / "b.mp4"
        v2 = tmp_path / "c.mp4"
        for v in (v0, v1, v2):
            v.write_bytes(b"")

        with pytest.raises(ValueError, match="not inside a DLC project"):
            dustrack.open([v0, v1, v2])
        assert fake_dustrack.calls == []

    def test_mixed_str_and_path_entries_rejected_for_bare(
        self, tmp_path, fake_dustrack
    ):
        """Mixed str/Path entries don't bypass the single-project
        validation."""
        v0 = tmp_path / "a.mp4"
        v1 = tmp_path / "b.mp4"
        for v in (v0, v1):
            v.write_bytes(b"")

        with pytest.raises(ValueError, match="not inside a DLC project"):
            dustrack.open([str(v0), v1])
        assert fake_dustrack.calls == []

    def test_tuple_form_bare_videos_rejected(
        self, tmp_path, fake_dustrack
    ):
        """Tuple shape (vs. list) gets the same validation."""
        v0 = tmp_path / "a.mp4"
        v1 = tmp_path / "b.mp4"
        for v in (v0, v1):
            v.write_bytes(b"")

        with pytest.raises(ValueError, match="not inside a DLC project"):
            dustrack.open((v0, v1))
        assert fake_dustrack.calls == []

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError, match="empty path sequence"):
            dustrack.open([])

    def test_list_with_missing_file_raises_filenotfound(self, tmp_path):
        bogus = tmp_path / "nope.mp4"
        with pytest.raises(FileNotFoundError):
            dustrack.open([bogus])


# ---------------------------------------------------------------------
# Scalar path still sets _video_queue (always-set contract)
# ---------------------------------------------------------------------


class TestVideoQueueAlwaysSet:
    def test_scalar_open_sets_empty_queue(self, tmp_path, fake_dustrack):
        vid = tmp_path / "ok.mp4"
        vid.write_bytes(b"")
        tracker = dustrack.open(vid)
        # Consumers shouldn't need a getattr-with-default dance.
        assert hasattr(tracker, "_video_queue")
        assert tracker._video_queue == []
