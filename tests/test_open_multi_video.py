"""Tests for the 1.2.0a3 multi-video entry-point dispatch in
:func:`dustrack.open`.

The strict-single-DLC-project contract: every input video must
belong to the same DLC project. Bare videos, mixed projects, and
config.yaml paths all raise. Two entry shapes are the happy paths:

- ``dustrack.open(project_folder)`` queues every video in the
  project (behavior change vs <=1.2.0a2 which opened only the first).
- ``dustrack.open([v0, v1, ...])`` queues exactly those videos in
  order.

Tests stub :class:`DLCProject` and :class:`DUSTrack` so the path
classification + validation can run without a real DLC install or
GPU. The actual swap / hydration machinery is tested separately
(``test_swap_to.py``).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------
# Synthetic project layout helper
# ---------------------------------------------------------------------


def _make_project(tmp_path: Path, name: str = "proj", videos: list = None):
    """Lay down a directory that satisfies ``_is_dlc_project_root``.

    Includes ``config.yaml`` + ``videos/`` + ``labeled-data/`` so the
    structural check fires. Returns ``(root, [video_paths])``.
    """
    root = tmp_path / name
    root.mkdir()
    (root / "config.yaml").write_text("placeholder: yes\n", encoding="utf-8")
    (root / "labeled-data").mkdir()
    vids_dir = root / "videos"
    vids_dir.mkdir()
    video_names = videos if videos is not None else ["v0.mp4", "v1.mp4", "v2.mp4"]
    video_paths = []
    for name in video_names:
        p = vids_dir / name
        p.write_bytes(b"")
        video_paths.append(p)
    return root, video_paths


@pytest.fixture
def fake_dlcproject(monkeypatch, force_has_dlc):
    """Patch :class:`DLCProject` so construction returns a tiny stub
    exposing the attributes the multi-video dispatch reads
    (``config_path``, ``video_list``, ``video_names``). ``force_has_dlc``
    bypasses the dispatch's pre-construction ``HAS_DLC`` guards so the stub
    is reached without a real deeplabcut install."""
    calls = []

    class _StubProject:
        def __init__(self, config_path):
            calls.append(config_path)
            self.config_path = config_path
            root = Path(config_path).parent
            self._root = root
            vids = sorted((root / "videos").glob("*.mp4"))
            self.video_list = [str(v) for v in vids]
            self.video_names = [Path(v).stem for v in vids]

        # Read by the active-bundle Phase 2 dispatch in open().
        def annotate(self, video_index=0, new_annotation_suffix=None, **kw):
            # Mimic DLCProject.annotate's contract enough to satisfy
            # the dispatch: return an object that quacks like a
            # DUSTrack tracker for the bundle init fallback.
            tracker = SimpleNamespace(
                fname=self.video_list[video_index],
                _video_queue=[],
            )
            tracker._init_bundles_calls = []
            return tracker

    # Patch DLCProject at every resolution point. Post-1.2.0rc1 it is
    # imported at module-load from dustrack.dlcinterface into
    # dustrack._open / dustrack.gui / etc. as snapshot bindings -- patching
    # dlcinterface alone doesn't affect their locals. Iterating sys.modules
    # keeps this robust to future binding sites. (HAS_DLC is forced True by
    # the force_has_dlc fixture so the pre-construction guards pass first.)
    import sys

    import dustrack  # noqa: F401 -- ensure submodules are loaded before patching

    for _name, _mod in list(sys.modules.items()):
        if _name == "dustrack" or _name.startswith("dustrack."):
            if getattr(_mod, "DLCProject", None) is not None:
                monkeypatch.setattr(_mod, "DLCProject", _StubProject, raising=False)
    _StubProject._calls = calls
    return _StubProject


@pytest.fixture
def capture_init_bundles(monkeypatch):
    """Patch :func:`_attach_bundles_or_fallback` to record its args
    instead of doing the real bundle setup. Lets tests assert the
    dispatch handed the right ``project`` + ``video_paths`` pair
    without spinning up Qt / artists."""
    calls = []

    def _capture(tracker, project, video_paths):
        calls.append({
            "tracker": tracker,
            "project": project,
            "video_paths": video_paths,
        })
        tracker._video_queue = [Path(v) for v in video_paths[1:]]

    monkeypatch.setattr(
        "dustrack._open._attach_bundles_or_fallback", _capture,
    )
    return calls


# ---------------------------------------------------------------------
# Project-folder dispatch (behavior change vs <=1.2.0a2)
# ---------------------------------------------------------------------


class TestProjectFolderDispatch:
    def test_project_folder_queues_every_video(
        self, tmp_path, fake_dlcproject, capture_init_bundles,
    ):
        """Pointing ``open`` at a project root queues every video in
        ``project.video_list`` (project order, not directory order).
        Behavior change from <=1.2.0a2 which opened only video 0."""
        import dustrack

        root, video_paths = _make_project(tmp_path)
        tracker = dustrack.open(root)

        assert tracker is not None
        # _attach_bundles_or_fallback got the project + every video.
        call = capture_init_bundles[0]
        assert call["project"] is not None
        assert [Path(v) for v in call["video_paths"]] == [
            Path(v) for v in fake_dlcproject._calls[0:1]  # noqa: not used
        ] or True  # we'll assert against the stub's video_list below
        # Get the stub project for the assertion.
        stub_project = call["project"]
        assert [str(p) for p in call["video_paths"]] == stub_project.video_list

    def test_project_folder_no_videos_raises(
        self, tmp_path, fake_dlcproject,
    ):
        import dustrack
        root, _ = _make_project(tmp_path, videos=[])
        # No videos in the project.
        with pytest.raises(ValueError, match="has no videos"):
            dustrack.open(root)

    def test_non_dlc_directory_rejected(self, tmp_path):
        """A bare directory (no DLC project structure) still raises
        ValueError as it did pre-1.2.0a3."""
        import dustrack
        d = tmp_path / "not_a_project"
        d.mkdir()
        with pytest.raises(ValueError, match="doesn't look like"):
            dustrack.open(d)


# ---------------------------------------------------------------------
# List-form dispatch with single-project validation
# ---------------------------------------------------------------------


class TestListFormValidation:
    def test_list_inside_single_project_succeeds(
        self, tmp_path, fake_dlcproject, capture_init_bundles,
    ):
        import dustrack

        root, video_paths = _make_project(tmp_path)
        # Pick a subset.
        subset = [video_paths[0], video_paths[2]]
        tracker = dustrack.open(subset)

        assert tracker is not None
        call = capture_init_bundles[0]
        assert [Path(v) for v in call["video_paths"]] == subset

    def test_list_in_different_projects_rejected(
        self, tmp_path, fake_dlcproject,
    ):
        import dustrack
        root_a, paths_a = _make_project(tmp_path, name="A")
        root_b, paths_b = _make_project(tmp_path, name="B")

        with pytest.raises(
            ValueError, match="span multiple DLC projects"
        ):
            dustrack.open([paths_a[0], paths_b[0]])

    def test_list_with_bare_video_rejected(
        self, tmp_path, fake_dlcproject,
    ):
        import dustrack
        root, paths = _make_project(tmp_path)
        bare = tmp_path / "bare.mp4"
        bare.write_bytes(b"")

        with pytest.raises(ValueError, match="not inside a DLC project"):
            dustrack.open([paths[0], bare])

    def test_list_with_directory_entry_rejected(
        self, tmp_path, fake_dlcproject,
    ):
        import dustrack
        root, paths = _make_project(tmp_path)
        # A directory inside the project is not a video.
        # The validation says "is not a file".
        with pytest.raises(ValueError, match="is not a file"):
            dustrack.open([paths[0], root / "labeled-data"])

    def test_single_element_list_dispatches_as_scalar(
        self, tmp_path, fake_dlcproject, capture_init_bundles,
    ):
        """A 1-element list-form dispatches identically to the scalar
        form -- preserves the pre-1.2.0a3 parity."""
        import dustrack

        root, paths = _make_project(tmp_path)
        # Use a video INSIDE the project so the dispatch picks Phase 2
        # single (the project resolves via _find_dlc_config walk-up).
        tracker = dustrack.open([paths[0]])

        # _attach_bundles_or_fallback got a single-element list.
        assert len(capture_init_bundles[0]["video_paths"]) == 1

    def test_empty_list_raises(self):
        import dustrack
        with pytest.raises(ValueError, match="empty path sequence"):
            dustrack.open([])


# ---------------------------------------------------------------------
# config.yaml dispatch (1.2.0a3 follow-up 2026-05-22)
#
# Behavior change vs the pre-fix dispatch: picking a config.yaml now
# queues every video in the project (multi-video), in
# config['video_sets'] order. Pre-fix this opened video 0 only.
# DLCProject.__init__ runs rebase_to_config so a renamed project
# folder self-heals before enumerate -- covered structurally here
# by relying on the stub's video_list (real rebase tested via the
# DLCProject __init__ path, out of scope for this dispatch test).
# ---------------------------------------------------------------------


class TestConfigYamlDispatch:
    def test_config_yaml_scalar_queues_every_video(
        self, tmp_path, fake_dlcproject, capture_init_bundles,
    ):
        import dustrack
        root, video_paths = _make_project(tmp_path)
        cfg = root / "config.yaml"

        tracker = dustrack.open(cfg)
        assert tracker is not None
        call = capture_init_bundles[0]
        stub_project = call["project"]
        # Every video in the project is queued, in stub.video_list
        # order (i.e. what the project itself reports).
        assert [str(p) for p in call["video_paths"]] == stub_project.video_list

    def test_config_yaml_single_element_list_dispatches_same(
        self, tmp_path, fake_dlcproject, capture_init_bundles,
    ):
        """``open([config.yaml])`` (1-element list-form) dispatches
        identically to scalar ``open(config.yaml)`` -- multi-video.
        This is the path the seed-modal's ``replace_active_with``
        takes when the user picks a config.yaml."""
        import dustrack
        root, video_paths = _make_project(tmp_path)
        cfg = root / "config.yaml"

        tracker = dustrack.open([cfg])
        assert tracker is not None
        call = capture_init_bundles[0]
        stub_project = call["project"]
        assert [str(p) for p in call["video_paths"]] == stub_project.video_list

    def test_config_yaml_no_videos_raises(
        self, tmp_path, fake_dlcproject,
    ):
        import dustrack
        root, _ = _make_project(tmp_path, videos=[])
        cfg = root / "config.yaml"
        with pytest.raises(ValueError, match="has no videos"):
            dustrack.open(cfg)

    def test_config_yaml_in_multi_video_list_rejected(
        self, tmp_path, fake_dlcproject,
    ):
        """Multi-element list-form cannot mix config.yaml with
        videos -- the dispatch is ambiguous, so we raise with a
        clear pointer to the supported shape."""
        import dustrack
        root, paths = _make_project(tmp_path)
        cfg = root / "config.yaml"
        with pytest.raises(ValueError, match="is a DLC config.yaml"):
            dustrack.open([paths[0], cfg])


# ---------------------------------------------------------------------
# _validate_bundle_paths config.yaml branch
# ---------------------------------------------------------------------


class TestValidateBundlePathsConfigYaml:
    def test_validator_returns_multi_video_for_config_yaml(
        self, tmp_path, fake_dlcproject,
    ):
        """``DUSTrack._validate_bundle_paths`` mirrors the
        ``dustrack.open`` dispatch -- a config.yaml input returns a
        ``(project, video_paths)`` tuple ready for multi-video init.
        """
        from dustrack.dlcinterface import DUSTrack
        root, video_paths = _make_project(tmp_path)
        cfg = root / "config.yaml"
        # Bypass __init__ -- we only need the unbound method's logic.
        stub_self = type("_S", (), {})()
        project, paths = DUSTrack._validate_bundle_paths(stub_self, cfg)
        assert project is not None
        assert [str(p) for p in paths] == project.video_list

    def test_validator_recurses_through_single_element_list(
        self, tmp_path, fake_dlcproject,
    ):
        from dustrack.dlcinterface import DUSTrack
        root, video_paths = _make_project(tmp_path)
        cfg = root / "config.yaml"
        # The list-form branch recurses via ``self._validate_bundle_paths(...)``
        # for 1-element lists, so we bind the method onto the stub.
        stub_self = type("_S", (), {})()
        stub_self._validate_bundle_paths = (
            DUSTrack._validate_bundle_paths.__get__(stub_self)
        )
        project, paths = stub_self._validate_bundle_paths([cfg])
        assert project is not None
        assert [str(p) for p in paths] == project.video_list
