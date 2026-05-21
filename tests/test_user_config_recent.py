"""Tests for the cross-session recent-videos / recent-folders history.

The history lives in ``~/.dustrack/config.json`` -- tests monkeypatch
the module-level path constants in :mod:`dustrack._config` to a
``tmp_path`` location so they don't pollute the real user config.

Three concerns covered here:

1. ``_config`` accessors: dedup, cap, stale-on-disk filter, and the
   ``get_last_video_picker_dir`` derivation chain.
2. ``_prompt_for_videos`` passes the derived directory into the
   QFileDialog call (mocked).
3. ``DUSTrack._record_session_in_history`` writes the right
   ``recent_videos`` / ``recent_folders`` entries for single-video
   and multi-video sessions.
"""
from pathlib import Path

import pytest

from dustrack import _config


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Redirect the per-user config store to ``tmp_path`` for the
    duration of one test. Yields the resolved ``config.json`` path so
    tests can inspect / pre-seed it directly."""
    cfg_dir = tmp_path / ".dustrack"
    cfg_path = cfg_dir / "config.json"
    monkeypatch.setattr(_config, "_USER_CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(_config, "_USER_CONFIG_PATH", cfg_path)
    # ``seed.py`` re-exports the same names -- patch there too so any
    # callers that imported via ``dustrack.seed`` see the redirect.
    import dustrack.seed as seed
    monkeypatch.setattr(seed, "_USER_CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(seed, "_USER_CONFIG_PATH", cfg_path)
    return cfg_path


# ---------------------------------------------------------------------
# _read_user_config / _write_user_config (lifted from seed.py)
# ---------------------------------------------------------------------


class TestUserConfigRoundTrip:
    def test_missing_file_returns_empty(self, isolated_config):
        assert _config._read_user_config() == {}

    def test_round_trip(self, isolated_config):
        _config._write_user_config({"key": "value"})
        assert _config._read_user_config() == {"key": "value"}

    def test_unreadable_file_returns_empty(self, isolated_config):
        # Corrupt JSON: empty-dict fallback rather than a crash that
        # would strand callers on a stale config.
        isolated_config.parent.mkdir(parents=True, exist_ok=True)
        isolated_config.write_text("not json")
        assert _config._read_user_config() == {}

    def test_seed_reexport_still_works(self, isolated_config):
        # ``dustrack.seed._read_user_config`` must keep behaving --
        # the seed-bundles flow accesses it directly.
        from dustrack.seed import _read_user_config, _write_user_config
        _write_user_config({"x": 1})
        assert _read_user_config() == {"x": 1}


# ---------------------------------------------------------------------
# recent_videos / recent_folders accessors
# ---------------------------------------------------------------------


class TestRecentVideos:
    def test_empty_on_fresh_install(self, isolated_config):
        assert _config.get_recent_videos() == []

    def test_record_then_read(self, isolated_config, tmp_path):
        v = tmp_path / "vid.mp4"
        v.write_bytes(b"")
        _config.record_recent_video(v)
        result = _config.get_recent_videos()
        assert result == [v.resolve()]

    def test_recency_order_most_recent_first(self, isolated_config, tmp_path):
        v0 = tmp_path / "a.mp4"
        v1 = tmp_path / "b.mp4"
        v2 = tmp_path / "c.mp4"
        for v in (v0, v1, v2):
            v.write_bytes(b"")
        _config.record_recent_video(v0)
        _config.record_recent_video(v1)
        _config.record_recent_video(v2)
        result = _config.get_recent_videos()
        assert result == [v2.resolve(), v1.resolve(), v0.resolve()]

    def test_dedup_moves_to_front(self, isolated_config, tmp_path):
        v0 = tmp_path / "a.mp4"
        v1 = tmp_path / "b.mp4"
        for v in (v0, v1):
            v.write_bytes(b"")
        _config.record_recent_video(v0)
        _config.record_recent_video(v1)
        # Re-record v0 -- should move to front, not duplicate.
        _config.record_recent_video(v0)
        result = _config.get_recent_videos()
        assert result == [v0.resolve(), v1.resolve()]
        # Underlying JSON list has no dup either.
        raw = _config._read_user_config()["recent_videos"]
        assert len(raw) == 2

    def test_cap_drops_oldest(self, isolated_config, tmp_path):
        # Record _RECENT_LIST_CAP + 5 videos; the oldest 5 fall off.
        vids = []
        for i in range(_config._RECENT_LIST_CAP + 5):
            v = tmp_path / f"v{i:03d}.mp4"
            v.write_bytes(b"")
            vids.append(v)
            _config.record_recent_video(v)
        result = _config.get_recent_videos()
        assert len(result) == _config._RECENT_LIST_CAP
        # First entry is the most-recent insertion.
        assert result[0] == vids[-1].resolve()
        # The earliest five must be gone.
        for early in vids[:5]:
            assert early.resolve() not in result

    def test_stale_entry_filtered_on_read(self, isolated_config, tmp_path):
        # Record a real file, then delete it -- ``get_recent_videos``
        # must skip the missing entry without raising. The on-disk
        # JSON keeps the stale string so a re-mount of a network
        # drive can recover the entry later.
        v = tmp_path / "transient.mp4"
        v.write_bytes(b"")
        _config.record_recent_video(v)
        v.unlink()
        assert _config.get_recent_videos() == []
        raw = _config._read_user_config()["recent_videos"]
        assert len(raw) == 1  # stale entry preserved on disk


class TestRecentFolders:
    def test_empty_on_fresh_install(self, isolated_config):
        assert _config.get_recent_folders() == []

    def test_record_then_read(self, isolated_config, tmp_path):
        d = tmp_path / "sessions"
        d.mkdir()
        _config.record_recent_folder(d)
        assert _config.get_recent_folders() == [d.resolve()]

    def test_stale_folder_filtered_on_read(self, isolated_config, tmp_path):
        import shutil
        d = tmp_path / "ephemeral"
        d.mkdir()
        _config.record_recent_folder(d)
        shutil.rmtree(d)
        assert _config.get_recent_folders() == []

    def test_dedup_and_cap(self, isolated_config, tmp_path):
        # Quick smoke -- shares the helper with recent_videos so the
        # detailed behavior is covered there. Confirm dedup happens.
        d = tmp_path / "shared"
        d.mkdir()
        _config.record_recent_folder(d)
        _config.record_recent_folder(d)
        raw = _config._read_user_config()["recent_folders"]
        assert raw == [str(d.resolve())]


# ---------------------------------------------------------------------
# get_last_video_picker_dir (derivation chain)
# ---------------------------------------------------------------------


class TestLastVideoPickerDir:
    def test_none_on_fresh_install(self, isolated_config):
        assert _config.get_last_video_picker_dir() is None

    def test_derives_from_recent_videos(self, isolated_config, tmp_path):
        v = tmp_path / "sub" / "vid.mp4"
        v.parent.mkdir()
        v.write_bytes(b"")
        _config.record_recent_video(v)
        assert _config.get_last_video_picker_dir() == v.parent.resolve()

    def test_falls_back_to_recent_folders_if_videos_empty(
        self, isolated_config, tmp_path
    ):
        d = tmp_path / "session"
        d.mkdir()
        _config.record_recent_folder(d)
        assert _config.get_last_video_picker_dir() == d.resolve()

    def test_videos_win_over_folders(self, isolated_config, tmp_path):
        v = tmp_path / "vid.mp4"
        v.write_bytes(b"")
        d = tmp_path / "other"
        d.mkdir()
        _config.record_recent_folder(d)
        _config.record_recent_video(v)
        # Most-recent video's parent wins; the folder entry is the
        # fallback only when videos is empty.
        assert _config.get_last_video_picker_dir() == v.parent.resolve()

    def test_stale_entries_skipped_in_derivation(
        self, isolated_config, tmp_path
    ):
        # Earlier video deleted -- derivation skips to the next existing
        # entry, NOT to the folders list (since videos still has live
        # entries).
        v_stale = tmp_path / "old.mp4"
        v_stale.write_bytes(b"")
        v_live = tmp_path / "newer" / "vid.mp4"
        v_live.parent.mkdir()
        v_live.write_bytes(b"")
        _config.record_recent_video(v_stale)
        _config.record_recent_video(v_live)
        v_live.unlink()
        # Now only the older one remains live.
        assert _config.get_last_video_picker_dir() == v_stale.parent.resolve()


# ---------------------------------------------------------------------
# _prompt_for_videos passes the derived dir
# ---------------------------------------------------------------------


class TestPickerStartDir:
    def test_picker_directory_is_last_picker_dir(
        self, isolated_config, tmp_path, monkeypatch
    ):
        try:
            from qtpy.QtWidgets import QFileDialog
        except ImportError:
            pytest.skip("qtpy not installed in this env")

        # Pre-seed: open a video so the derivation has a folder to
        # return.
        v = tmp_path / "remembered_dir" / "vid.mp4"
        v.parent.mkdir()
        v.write_bytes(b"")
        _config.record_recent_video(v)

        captured = {}

        def _fake_get_open_file_names(parent, caption, directory, filter_):
            captured["directory"] = directory
            return ([], "")

        monkeypatch.setattr(
            QFileDialog, "getOpenFileNames", staticmethod(_fake_get_open_file_names)
        )

        from dustrack.dlcinterface import _prompt_for_videos
        _prompt_for_videos()

        assert captured["directory"] == str(v.parent.resolve())

    def test_picker_directory_empty_on_fresh_install(
        self, isolated_config, monkeypatch
    ):
        try:
            from qtpy.QtWidgets import QFileDialog
        except ImportError:
            pytest.skip("qtpy not installed in this env")

        captured = {}

        def _fake_get_open_file_names(parent, caption, directory, filter_):
            captured["directory"] = directory
            return ([], "")

        monkeypatch.setattr(
            QFileDialog, "getOpenFileNames", staticmethod(_fake_get_open_file_names)
        )

        from dustrack.dlcinterface import _prompt_for_videos
        _prompt_for_videos()

        # Fresh-install: no recent entries, fall back to "" (OS default).
        assert captured["directory"] == ""


# ---------------------------------------------------------------------
# DUSTrack._record_session_in_history (close-guard tail behavior)
# ---------------------------------------------------------------------


class _StubDustrack:
    """Minimal duck-type for ``_record_session_in_history``: only
    ``fname`` and ``_video_queue`` are read."""

    def __init__(self, fname, queue=None):
        self.fname = fname
        self._video_queue = queue or []

    # The real method lives on ``_DUSTrackBase``; rebind here so we
    # can call it without constructing a full DUSTrack.
    _record_session_in_history = None  # filled in below


def _bind_history_method():
    from dustrack.dlcinterface import DUSTrack
    _StubDustrack._record_session_in_history = (
        DUSTrack._record_session_in_history
    )


_bind_history_method()


class TestRecordSessionInHistory:
    def test_single_video_session_records_video_only(
        self, isolated_config, tmp_path
    ):
        v = tmp_path / "vid.mp4"
        v.write_bytes(b"")
        stub = _StubDustrack(fname=str(v))
        stub._record_session_in_history()

        assert _config.get_recent_videos() == [v.resolve()]
        # No folder record on a single-video session.
        assert _config.get_recent_folders() == []

    def test_multi_video_session_records_both(
        self, isolated_config, tmp_path
    ):
        d = tmp_path / "session"
        d.mkdir()
        v0 = d / "a.mp4"
        v1 = d / "b.mp4"
        v2 = d / "c.mp4"
        for v in (v0, v1, v2):
            v.write_bytes(b"")
        stub = _StubDustrack(fname=str(v0), queue=[v1, v2])
        stub._record_session_in_history()

        # Only the active video lands in recent_videos -- the queue
        # entries are session-state, not "videos the user worked on".
        # If we wanted them recorded individually we'd need to hook on
        # nav rather than close.
        assert _config.get_recent_videos() == [v0.resolve()]
        # Common parent of [v0, v1, v2] is ``d`` -- recorded.
        assert _config.get_recent_folders() == [d.resolve()]

    def test_no_fname_is_a_noop(self, isolated_config):
        stub = _StubDustrack(fname=None)
        stub._record_session_in_history()
        assert _config.get_recent_videos() == []
        assert _config.get_recent_folders() == []

    def test_mixed_drives_skip_folder_record(
        self, isolated_config, tmp_path, monkeypatch
    ):
        # Simulate the Windows "different drives" case: monkeypatch
        # os.path.commonpath to raise ValueError (the platform shape).
        # ``recent_videos`` still gets the active path; the folder
        # write is silently skipped.
        v = tmp_path / "a.mp4"
        v.write_bytes(b"")

        import os
        def _raise(_paths):
            raise ValueError("paths don't share a drive")
        monkeypatch.setattr(os.path, "commonpath", _raise)

        stub = _StubDustrack(fname=str(v), queue=[Path("D:/elsewhere/b.mp4")])
        stub._record_session_in_history()

        assert _config.get_recent_videos() == [v.resolve()]
        assert _config.get_recent_folders() == []
