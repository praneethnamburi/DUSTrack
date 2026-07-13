"""Tests for the Create DLC Project options modal (1.3.0a2).

The Qt dialog itself (``CreateProjectOptionsDialog``) is manual-smoke
per the ConfirmOverlay / TrainingOptionsDialog precedent -- synchronous
modal exec is painful headless. These tests pin the pure-Python pieces:
the default-options builder, the validator, and the config persistence
of the last-used project root.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from dustrack._overlays import (
    _default_create_project_options,
    _validate_create_project_options,
)
from dustrack import _config


class TestDefaultOptions:
    def test_name_is_stem_underscore_layer(self):
        # Forward slashes so Path().stem is portable: on POSIX a backslash
        # isn't a separator, so r"M:\vids\x.mp4" would be one filename.
        opts = _default_create_project_options(
            video_fname="M:/vids/pia02_s001_006_RFA2.mp4",
            layer_name="iteration-0",
            experimenter="praneeth",
        )
        assert opts["name"] == "pia02_s001_006_RFA2_iteration-0"

    def test_path_falls_back_to_video_parent_when_no_last_root(self):
        opts = _default_create_project_options(
            video_fname="M:/vids/v0.mp4",
            layer_name="manual",
            experimenter="x",
            last_project_root=None,
        )
        assert Path(opts["path"]) == Path("M:/vids")

    def test_path_uses_last_project_root_when_supplied(self):
        opts = _default_create_project_options(
            video_fname=r"M:\vids\v0.mp4",
            layer_name="manual",
            experimenter="x",
            last_project_root=r"M:\DLC_MODELS",
        )
        assert Path(opts["path"]) == Path(r"M:\DLC_MODELS")

    def test_experimenter_passed_through(self):
        opts = _default_create_project_options(
            video_fname="v0.mp4",
            layer_name="manual",
            experimenter="alice",
        )
        assert opts["experimenter"] == "alice"

    def test_no_seed_video_leak(self):
        """Regression: with the attach_bundle fix, self.fname is the real
        video, so the derived name never says 'seed_video'."""
        opts = _default_create_project_options(
            video_fname=r"M:\us_videos_for_tracking2\pia02_s001_006_RFA2.mp4",
            layer_name="iteration-0",
            experimenter="praneeth",
        )
        assert "seed_video" not in opts["name"]


class TestValidate:
    def test_valid_options_pass(self):
        ok, msg = _validate_create_project_options(
            {"name": "proj_iteration-0", "path": r"M:\DLC_MODELS",
             "experimenter": "praneeth"}
        )
        # Note: "proj_iteration-0" contains a dash -> should FAIL.
        assert ok is False

    def test_clean_name_passes(self):
        ok, msg = _validate_create_project_options(
            {"name": "proj_iter0", "path": r"M:\DLC_MODELS",
             "experimenter": "praneeth"}
        )
        assert ok is True
        assert msg == ""

    def test_empty_name_fails(self):
        ok, msg = _validate_create_project_options(
            {"name": "", "path": "M:\\x", "experimenter": "p"}
        )
        assert ok is False
        assert "name" in msg.lower()

    def test_dash_in_name_fails(self):
        ok, msg = _validate_create_project_options(
            {"name": "proj-bad", "path": "M:\\x", "experimenter": "p"}
        )
        assert ok is False
        assert "-" in msg

    def test_empty_path_fails(self):
        ok, msg = _validate_create_project_options(
            {"name": "proj_ok", "path": "", "experimenter": "p"}
        )
        assert ok is False
        assert "folder" in msg.lower()

    def test_empty_experimenter_fails(self):
        ok, msg = _validate_create_project_options(
            {"name": "proj_ok", "path": "M:\\x", "experimenter": ""}
        )
        assert ok is False
        assert "experimenter" in msg.lower()

    def test_dash_in_experimenter_fails(self):
        ok, msg = _validate_create_project_options(
            {"name": "proj_ok", "path": "M:\\x", "experimenter": "jane-doe"}
        )
        assert ok is False

    def test_whitespace_stripped(self):
        ok, _msg = _validate_create_project_options(
            {"name": "  proj_ok  ", "path": "  M:\\x  ",
             "experimenter": "  p  "}
        )
        assert ok is True


class TestProjectRootPersistence:
    """``record_project_root`` / ``get_last_project_root`` round-trip
    against an isolated config file."""

    @pytest.fixture
    def isolated_config(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / ".dustrack"
        cfg_dir.mkdir()
        cfg_path = cfg_dir / "config.json"
        monkeypatch.setattr(_config, "_USER_CONFIG_DIR", cfg_dir)
        monkeypatch.setattr(_config, "_USER_CONFIG_PATH", cfg_path)
        return cfg_path

    def test_round_trip(self, isolated_config, tmp_path):
        root = tmp_path / "DLC_MODELS"
        root.mkdir()
        _config.record_project_root(str(root))
        assert _config.get_last_project_root() == root

    def test_get_returns_none_when_unset(self, isolated_config):
        assert _config.get_last_project_root() is None

    def test_get_returns_none_when_path_gone(self, isolated_config, tmp_path):
        gone = tmp_path / "deleted_dir"
        gone.mkdir()
        _config.record_project_root(str(gone))
        gone.rmdir()  # directory no longer exists
        assert _config.get_last_project_root() is None

    def test_record_overwrites(self, isolated_config, tmp_path):
        a = tmp_path / "a"; a.mkdir()
        b = tmp_path / "b"; b.mkdir()
        _config.record_project_root(str(a))
        _config.record_project_root(str(b))
        assert _config.get_last_project_root() == b

    def test_does_not_clobber_other_config_keys(self, isolated_config, tmp_path):
        # Seed an unrelated key, then record a project root; the other
        # key must survive.
        _config._write_user_config({"seed_bundles_root": "M:/bundles"})
        root = tmp_path / "DLC_MODELS"; root.mkdir()
        _config.record_project_root(str(root))
        cfg = _config._read_user_config()
        assert cfg["seed_bundles_root"] == "M:/bundles"
        assert cfg["last_project_root"] == str(root)
