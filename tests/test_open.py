"""Tests for :func:`dustrack.open` and its path-classification helpers.

These exercise the dispatch logic with synthetic filesystem inputs and
the error branches of ``open()``. The Phase 1 / Phase 2 happy paths
require launching the GUI (and, for Phase 2, a real DLC project on
disk), so they're left to manual / integration testing.
"""
from pathlib import Path

import pytest

import dustrack
from dustrack.dlcinterface import (
    HAS_DLC,
    _find_dlc_config,
    _find_video_index,
    _is_dlc_project_root,
)


def _make_dlc_root(folder: Path) -> Path:
    """Lay down the minimal structure ``_is_dlc_project_root`` looks for."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.yaml").write_text("# fake DLC config\n", encoding="utf-8")
    (folder / "videos").mkdir()
    (folder / "labeled-data").mkdir()
    return folder


class TestIsDLCProjectRoot:
    def test_full_structure_matches(self, tmp_path):
        root = _make_dlc_root(tmp_path / "proj")
        assert _is_dlc_project_root(root) is True

    def test_missing_config_yaml(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        (root / "videos").mkdir()
        (root / "labeled-data").mkdir()
        assert _is_dlc_project_root(root) is False

    def test_missing_videos_dir(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        (root / "config.yaml").write_text("", encoding="utf-8")
        (root / "labeled-data").mkdir()
        assert _is_dlc_project_root(root) is False

    def test_missing_labeled_data_dir(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        (root / "config.yaml").write_text("", encoding="utf-8")
        (root / "videos").mkdir()
        assert _is_dlc_project_root(root) is False

    def test_plain_folder_rejected(self, tmp_path):
        folder = tmp_path / "just_a_folder"
        folder.mkdir()
        assert _is_dlc_project_root(folder) is False


class TestFindDLCConfig:
    def test_config_yaml_resolves_to_itself(self, tmp_path):
        root = _make_dlc_root(tmp_path / "proj")
        assert _find_dlc_config(root / "config.yaml") == root / "config.yaml"

    def test_project_folder_resolves_to_config(self, tmp_path):
        root = _make_dlc_root(tmp_path / "proj")
        assert _find_dlc_config(root) == root / "config.yaml"

    def test_video_inside_project_walks_up(self, tmp_path):
        root = _make_dlc_root(tmp_path / "proj")
        vid = root / "videos" / "sample.mp4"
        vid.write_bytes(b"")
        assert _find_dlc_config(vid) == root / "config.yaml"

    def test_bare_video_returns_none(self, tmp_path):
        vid = tmp_path / "sample.mp4"
        vid.write_bytes(b"")
        assert _find_dlc_config(vid) is None

    def test_video_in_plain_folder_returns_none(self, tmp_path):
        folder = tmp_path / "somewhere"
        folder.mkdir()
        vid = folder / "sample.mp4"
        vid.write_bytes(b"")
        assert _find_dlc_config(vid) is None

    def test_stray_config_yaml_rejected(self, tmp_path):
        # config.yaml without the sibling videos/ + labeled-data/ isn't
        # a DLC project — must not be auto-attached.
        cfg = tmp_path / "config.yaml"
        cfg.write_text("not a DLC config", encoding="utf-8")
        assert _find_dlc_config(cfg) is None

    def test_nonexistent_path_returns_none(self, tmp_path):
        assert _find_dlc_config(tmp_path / "does_not_exist.mp4") is None


class TestFindVideoIndex:
    class _StubProject:
        def __init__(self, names):
            self.video_names = names

    def test_match_first(self):
        proj = self._StubProject(["vidA", "vidB", "vidC"])
        assert _find_video_index(proj, "/some/path/vidA.mp4") == 0

    def test_match_middle(self):
        proj = self._StubProject(["vidA", "vidB", "vidC"])
        assert _find_video_index(proj, "vidB.mp4") == 1

    def test_no_match_returns_none(self):
        proj = self._StubProject(["vidA", "vidB"])
        assert _find_video_index(proj, "something_else.mp4") is None

    def test_stem_match_ignores_parent_dir(self):
        # Same stem, different parent dir -> match. Robust to the
        # drive-letter / UNC / posix shuffling that DLCProject already
        # handles via rebase_to_config.
        proj = self._StubProject(["vidA"])
        assert _find_video_index(proj, "/totally/different/path/vidA.mp4") == 0


class TestOpenDispatchErrors:
    """The dispatch error branches don't require a video reader or Qt."""

    def test_nonexistent_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            dustrack.open(tmp_path / "no_such_file.mp4")

    def test_phase1_without_layer_name_raises(self, tmp_path):
        vid = tmp_path / "sample.mp4"
        vid.write_bytes(b"")
        with pytest.raises(ValueError, match="layer_name"):
            dustrack.open(vid)

    def test_plain_directory_raises(self, tmp_path):
        folder = tmp_path / "plain_dir"
        folder.mkdir()
        with pytest.raises(ValueError, match="DLC project"):
            dustrack.open(folder)

    @pytest.mark.skipif(
        HAS_DLC, reason="Only meaningful in the no-deeplabcut install path"
    )
    def test_phase2_without_dlc_raises_importerror(self, tmp_path):
        root = _make_dlc_root(tmp_path / "proj")
        with pytest.raises(ImportError, match="deeplabcut"):
            dustrack.open(root)
