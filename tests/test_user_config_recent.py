"""Tests for the unified cross-session history (1.2.0a3 seed-window cut).

The history lives in ``~/.dustrack/config.json`` -- tests monkeypatch
the module-level path constants in :mod:`dustrack._config` to a
``tmp_path`` location so they don't pollute the real user config.

Coverage:

1. ``_config`` accessors: unified ``recent_sessions`` round-trip,
   dedup, cap, stale filter, ``get_last_video_picker_dir`` derivation.
2. Legacy ``recent_videos`` / ``recent_folders`` migration to the
   unified list.
3. Back-compat ``record_recent_video`` / ``get_recent_videos`` /
   ``record_recent_folder`` / ``get_recent_folders`` accessors stay
   working against the unified storage.
4. ``_prompt_for_videos`` passes the derived directory into the
   QFileDialog call (mocked).
5. ``DUSTrack._record_session_in_history`` writes the right
   ``recent_sessions`` entry for single-video and multi-video
   sessions, and skips for ``_is_seed_session = True``.
"""
import json
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
# Unified recent_sessions accessor
# ---------------------------------------------------------------------


class TestRecentSessions:
    def test_empty_on_fresh_install(self, isolated_config):
        assert _config.get_recent_sessions() == []

    def test_record_then_read_single(self, isolated_config, tmp_path):
        v = tmp_path / "vid.mp4"
        v.write_bytes(b"")
        _config.record_recent_session([v])
        assert _config.get_recent_sessions() == [[v.resolve()]]

    def test_record_then_read_multi(self, isolated_config, tmp_path):
        v0 = tmp_path / "a.mp4"
        v1 = tmp_path / "b.mp4"
        v2 = tmp_path / "c.mp4"
        for v in (v0, v1, v2):
            v.write_bytes(b"")
        _config.record_recent_session([v0, v1, v2])
        assert _config.get_recent_sessions() == [[v0.resolve(), v1.resolve(), v2.resolve()]]

    def test_recency_order_most_recent_first(self, isolated_config, tmp_path):
        v0 = tmp_path / "a.mp4"
        v1 = tmp_path / "b.mp4"
        v2 = tmp_path / "c.mp4"
        for v in (v0, v1, v2):
            v.write_bytes(b"")
        _config.record_recent_session([v0])
        _config.record_recent_session([v1])
        _config.record_recent_session([v2])
        result = _config.get_recent_sessions()
        assert result == [[v2.resolve()], [v1.resolve()], [v0.resolve()]]

    def test_dedup_moves_to_front(self, isolated_config, tmp_path):
        v0 = tmp_path / "a.mp4"
        v1 = tmp_path / "b.mp4"
        for v in (v0, v1):
            v.write_bytes(b"")
        _config.record_recent_session([v0])
        _config.record_recent_session([v1])
        # Re-record [v0] -- should move to front, not duplicate.
        _config.record_recent_session([v0])
        result = _config.get_recent_sessions()
        assert result == [[v0.resolve()], [v1.resolve()]]

    def test_dedup_distinguishes_single_from_multi(self, isolated_config, tmp_path):
        # [v0] and [v0, v1] are different sessions -- both kept.
        v0 = tmp_path / "a.mp4"
        v1 = tmp_path / "b.mp4"
        for v in (v0, v1):
            v.write_bytes(b"")
        _config.record_recent_session([v0])
        _config.record_recent_session([v0, v1])
        result = _config.get_recent_sessions()
        assert result == [[v0.resolve(), v1.resolve()], [v0.resolve()]]

    def test_cap_drops_oldest(self, isolated_config, tmp_path):
        # Record _RECENT_LIST_CAP + 5 sessions; the oldest 5 fall off.
        sessions = []
        for i in range(_config._RECENT_LIST_CAP + 5):
            v = tmp_path / f"v{i:03d}.mp4"
            v.write_bytes(b"")
            sessions.append(v)
            _config.record_recent_session([v])
        result = _config.get_recent_sessions()
        assert len(result) == _config._RECENT_LIST_CAP
        # First entry is the most-recent insertion.
        assert result[0] == [sessions[-1].resolve()]
        # The earliest five must be gone.
        live_active_paths = {entry[0] for entry in result}
        for early in sessions[:5]:
            assert early.resolve() not in live_active_paths

    def test_stale_entry_filtered_on_read(self, isolated_config, tmp_path):
        # Record a real file, then delete it -- ``get_recent_sessions``
        # must skip the missing entry without raising. The on-disk
        # JSON keeps the stale string so a re-mount of a network
        # drive can recover the entry later.
        v = tmp_path / "transient.mp4"
        v.write_bytes(b"")
        _config.record_recent_session([v])
        v.unlink()
        assert _config.get_recent_sessions() == []
        raw = _config._read_user_config()["recent_sessions"]
        assert len(raw) == 1  # stale entry preserved on disk

    def test_empty_paths_is_noop(self, isolated_config):
        _config.record_recent_session([])
        assert _config.get_recent_sessions() == []
        # No key gets written for an empty record.
        assert "recent_sessions" not in _config._read_user_config()


# ---------------------------------------------------------------------
# Legacy ``recent_videos`` / ``recent_folders`` -> ``recent_sessions``
# migration
# ---------------------------------------------------------------------


class TestLegacyMigration:
    def test_legacy_videos_migrate_to_one_element_sessions(
        self, isolated_config, tmp_path
    ):
        v0 = tmp_path / "a.mp4"
        v1 = tmp_path / "b.mp4"
        for v in (v0, v1):
            v.write_bytes(b"")
        # Pre-seed: pre-1.2.0a3 on-disk shape (string list).
        _config._write_user_config({
            "recent_videos": [str(v0.resolve()), str(v1.resolve())],
        })
        # First read migrates to ``recent_sessions``.
        assert _config.get_recent_sessions() == [
            [v0.resolve()], [v1.resolve()],
        ]
        # Legacy key dropped from disk.
        cfg = _config._read_user_config()
        assert "recent_videos" not in cfg
        assert "recent_sessions" in cfg

    def test_legacy_folders_migrate_to_one_element_sessions(
        self, isolated_config, tmp_path
    ):
        d = tmp_path / "session"
        d.mkdir()
        _config._write_user_config({
            "recent_folders": [str(d.resolve())],
        })
        assert _config.get_recent_sessions() == [[d.resolve()]]
        cfg = _config._read_user_config()
        assert "recent_folders" not in cfg

    def test_legacy_videos_first_then_folders(self, isolated_config, tmp_path):
        # Both keys present: video entries lead, folder entries follow.
        v = tmp_path / "vid.mp4"
        v.write_bytes(b"")
        d = tmp_path / "folder"
        d.mkdir()
        _config._write_user_config({
            "recent_videos": [str(v.resolve())],
            "recent_folders": [str(d.resolve())],
        })
        assert _config.get_recent_sessions() == [
            [v.resolve()], [d.resolve()],
        ]

    def test_migration_is_idempotent(self, isolated_config, tmp_path):
        v = tmp_path / "vid.mp4"
        v.write_bytes(b"")
        _config._write_user_config({
            "recent_videos": [str(v.resolve())],
        })
        # Two reads in a row: same result, no duplication.
        first = _config.get_recent_sessions()
        second = _config.get_recent_sessions()
        assert first == second == [[v.resolve()]]

    def test_migration_preserves_existing_sessions(
        self, isolated_config, tmp_path
    ):
        # If both ``recent_sessions`` and a legacy key are present
        # (hand-edited config), the legacy entries are appended
        # (without duplicating existing sessions).
        v_new = tmp_path / "new.mp4"
        v_old = tmp_path / "old.mp4"
        for v in (v_new, v_old):
            v.write_bytes(b"")
        _config._write_user_config({
            "recent_sessions": [[str(v_new.resolve())]],
            "recent_videos": [str(v_old.resolve()), str(v_new.resolve())],
        })
        # ``v_new`` already in unified list -- not duplicated.
        # ``v_old`` appended.
        result = _config.get_recent_sessions()
        # Active path is the only filter -- result entries:
        assert [str(p) for entry in result for p in entry] == [
            str(v_new.resolve()),
            str(v_old.resolve()),
        ]


# ---------------------------------------------------------------------
# Back-compat accessors
# ---------------------------------------------------------------------


class TestBackCompatAccessors:
    def test_record_recent_video_writes_unified(
        self, isolated_config, tmp_path
    ):
        v = tmp_path / "vid.mp4"
        v.write_bytes(b"")
        _config.record_recent_video(v)
        # Stored as a 1-element session.
        assert _config.get_recent_sessions() == [[v.resolve()]]
        # Back-compat reader returns the same path.
        assert _config.get_recent_videos() == [v.resolve()]

    def test_get_recent_videos_drops_multi_entries_dir_entries(
        self, isolated_config, tmp_path
    ):
        # Multi-element entries surface as their active path
        # (bundle 0 video) in the legacy ``get_recent_videos`` view.
        v0 = tmp_path / "a.mp4"
        v1 = tmp_path / "b.mp4"
        for v in (v0, v1):
            v.write_bytes(b"")
        d = tmp_path / "folder"
        d.mkdir()
        _config.record_recent_session([v0, v1])
        _config.record_recent_session([d])  # folder entry
        # Legacy view sees the multi-active and drops the dir-entry.
        assert _config.get_recent_videos() == [v0.resolve()]

    def test_record_recent_folder_writes_unified(
        self, isolated_config, tmp_path
    ):
        d = tmp_path / "session"
        d.mkdir()
        _config.record_recent_folder(d)
        assert _config.get_recent_sessions() == [[d.resolve()]]
        assert _config.get_recent_folders() == [d.resolve()]


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
        _config.record_recent_session([v])
        assert _config.get_last_video_picker_dir() == v.parent.resolve()

    def test_falls_back_to_recent_folders_if_videos_empty(
        self, isolated_config, tmp_path
    ):
        d = tmp_path / "session"
        d.mkdir()
        _config.record_recent_session([d])
        assert _config.get_last_video_picker_dir() == d.resolve()

    def test_videos_win_over_folders(self, isolated_config, tmp_path):
        v = tmp_path / "vid.mp4"
        v.write_bytes(b"")
        d = tmp_path / "other"
        d.mkdir()
        _config.record_recent_session([d])
        _config.record_recent_session([v])
        # Most-recent video's parent wins; the folder entry is the
        # fallback only when no file entries are live.
        assert _config.get_last_video_picker_dir() == v.parent.resolve()

    def test_stale_entries_skipped_in_derivation(
        self, isolated_config, tmp_path
    ):
        # Earlier video deleted -- derivation skips to the next existing
        # entry. ``recent_sessions`` filters at read time, so a missing
        # active-path entry is never seen by the derivation.
        v_stale = tmp_path / "old.mp4"
        v_stale.write_bytes(b"")
        v_live = tmp_path / "newer" / "vid.mp4"
        v_live.parent.mkdir()
        v_live.write_bytes(b"")
        _config.record_recent_session([v_stale])
        _config.record_recent_session([v_live])
        v_live.unlink()
        # Now only the older one is a live file entry.
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
        _config.record_recent_session([v])

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


class _StubBundle:
    """Tiny stand-in for ``_BundleState`` -- only ``fname`` is read by
    ``_record_session_in_history``."""

    def __init__(self, fname):
        from pathlib import Path
        self.fname = Path(fname)


class _StubDustrack:
    """Minimal duck-type for ``_record_session_in_history``: ``fname``
    and ``_bundles`` are read; ``_is_seed_session`` (optional, default
    False) gates the seed-tracker skip."""

    def __init__(self, fname, bundle_paths=None, is_seed=False):
        self.fname = fname
        if bundle_paths is None:
            self._bundles = []
        else:
            self._bundles = [_StubBundle(p) for p in bundle_paths]
        self._is_seed_session = is_seed

    # The real method lives on ``DUSTrack``; rebind here so we
    # can call it without constructing a full DUSTrack.
    _record_session_in_history = None  # filled in below


def _bind_history_method():
    from dustrack.dlcinterface import DUSTrack
    _StubDustrack._record_session_in_history = (
        DUSTrack._record_session_in_history
    )


_bind_history_method()


class TestRecordSessionInHistory:
    def test_single_video_session_records_one_element_entry(
        self, isolated_config, tmp_path
    ):
        v = tmp_path / "vid.mp4"
        v.write_bytes(b"")
        # 1.2.0a3: single-video sessions go through _init_bundles which
        # populates _bundles with one entry.
        stub = _StubDustrack(fname=str(v), bundle_paths=[v])
        stub._record_session_in_history()

        assert _config.get_recent_sessions() == [[v.resolve()]]

    def test_multi_video_session_records_full_bundle_list(
        self, isolated_config, tmp_path
    ):
        d = tmp_path / "session"
        d.mkdir()
        v0 = d / "a.mp4"
        v1 = d / "b.mp4"
        v2 = d / "c.mp4"
        for v in (v0, v1, v2):
            v.write_bytes(b"")
        stub = _StubDustrack(fname=str(v0), bundle_paths=[v0, v1, v2])
        stub._record_session_in_history()

        # Full bundle list recorded, in queue order.
        assert _config.get_recent_sessions() == [
            [v0.resolve(), v1.resolve(), v2.resolve()]
        ]

    def test_no_fname_is_a_noop(self, isolated_config):
        stub = _StubDustrack(fname=None)
        stub._record_session_in_history()
        assert _config.get_recent_sessions() == []

    def test_seed_session_skips_history_write(
        self, isolated_config, tmp_path
    ):
        # The seed-tracker (Phase 1.2.0a3 modal-host launch) must not
        # pollute the recent list with the synthetic seed_video.mp4
        # path. ``_is_seed_session = True`` short-circuits the write.
        v = tmp_path / "seed_video.mp4"
        v.write_bytes(b"")
        stub = _StubDustrack(fname=str(v), bundle_paths=[v], is_seed=True)
        stub._record_session_in_history()
        assert _config.get_recent_sessions() == []

    def test_legacy_bundleless_fallback_records_active_only(
        self, isolated_config, tmp_path
    ):
        # Defensive: if ``_bundles`` was never populated (test harness
        # bypassed _init_bundles), the recorder falls back to the
        # bare ``fname`` as a 1-element entry.
        v = tmp_path / "vid.mp4"
        v.write_bytes(b"")
        stub = _StubDustrack(fname=str(v))  # bundle_paths=None -> _bundles=[]
        stub._record_session_in_history()
        assert _config.get_recent_sessions() == [[v.resolve()]]


# ---------------------------------------------------------------------
# Pollution prune (1.2.0a3 cleanup added 2026-05-22)
#
# The read-path drops never-useful entries (seed asset, system temp dir)
# from disk on next read. Real users with pre-fix configs get a clean
# history automatically; new pollution is also blocked at the write
# side as defense-in-depth.
# ---------------------------------------------------------------------


class TestPolutionPrune:
    def test_seed_asset_dropped_from_history(self, isolated_config):
        from dustrack.dlcinterface import _SEED_VIDEO_PATH
        cfg = {
            "recent_sessions": [
                [str(_SEED_VIDEO_PATH)],
                ["C:/real/video.mp4"],
            ],
        }
        isolated_config.parent.mkdir(parents=True, exist_ok=True)
        isolated_config.write_text(json.dumps(cfg))
        raw = _config._read_sessions_raw()
        # Seed asset entry gone; real entry survives.
        assert [str(_SEED_VIDEO_PATH)] not in raw
        assert ["C:/real/video.mp4"] in raw

    def test_prune_persists_to_disk(self, isolated_config):
        from dustrack.dlcinterface import _SEED_VIDEO_PATH
        cfg = {
            "recent_sessions": [
                [str(_SEED_VIDEO_PATH)],
                ["C:/real/video.mp4"],
            ],
        }
        isolated_config.parent.mkdir(parents=True, exist_ok=True)
        isolated_config.write_text(json.dumps(cfg))
        _config._read_sessions_raw()
        # On-disk JSON should now have the seed entry stripped.
        on_disk = json.loads(isolated_config.read_text())
        assert [str(_SEED_VIDEO_PATH)] not in on_disk["recent_sessions"]
        assert ["C:/real/video.mp4"] in on_disk["recent_sessions"]

    def test_record_recent_session_rejects_seed_asset(self, isolated_config):
        from dustrack.dlcinterface import _SEED_VIDEO_PATH
        _config.record_recent_session([_SEED_VIDEO_PATH])
        # Nothing written.
        assert _config.get_recent_sessions() == []

    def test_record_recent_session_rejects_temp_path(self, isolated_config):
        import tempfile
        polluted = Path(tempfile.gettempdir()).resolve() / "abc" / "video.mp4"
        _config.record_recent_session([polluted])
        assert _config.get_recent_sessions() == []

    def test_unrelated_paths_survive_prune(self, isolated_config):
        cfg = {
            "recent_sessions": [
                ["C:/real/a.mp4"],
                ["C:/real/b.mp4"],
                ["\\\\server\\share\\c.mp4"],
            ],
        }
        isolated_config.parent.mkdir(parents=True, exist_ok=True)
        isolated_config.write_text(json.dumps(cfg))
        raw = _config._read_sessions_raw()
        assert len(raw) == 3

    def test_prune_idempotent_no_rewrite_when_clean(
        self, isolated_config,
    ):
        cfg = {
            "recent_sessions": [
                ["C:/real/a.mp4"],
            ],
        }
        isolated_config.parent.mkdir(parents=True, exist_ok=True)
        isolated_config.write_text(json.dumps(cfg))
        # Snapshot mtime, read once, verify mtime didn't change.
        mtime_before = isolated_config.stat().st_mtime_ns
        _config._read_sessions_raw()
        mtime_after = isolated_config.stat().st_mtime_ns
        assert mtime_before == mtime_after
