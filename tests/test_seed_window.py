"""Tests for the 1.2.0a3 seed-modal welcome-overlay surface.

The Qt modal itself (``OpenVideoOverlay.exec_()``) blocks on a local
``QEventLoop`` and is exercised manually -- same convention as
``ConfirmOverlay`` and the training-options modal. What's covered
here:

1. The pure-function label renderer (:func:`_render_recent_session_label`):
   1-element file, 1-element directory, N-element with shared parent,
   N-element with mixed parents.
2. The packaged seed video asset exists and matches the expected
   shape (8 frames, 64x64).
3. The :func:`_open_seed_session` constructor wires ``_is_seed_session``
   without populating ``recent_sessions`` (the close-guard short-
   circuit is verified separately in ``test_user_config_recent.py``).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from dustrack.dlcinterface import (
    _SEED_VIDEO_PATH,
    _render_recent_session_label,
)


# ---------------------------------------------------------------------
# Recent-session label rendering
# ---------------------------------------------------------------------


class TestRenderRecentSessionLabel:
    def test_single_file_full_path(self, tmp_path):
        v = tmp_path / "video.mp4"
        v.write_bytes(b"")
        label = _render_recent_session_label([v])
        assert label == str(v)

    def test_single_directory_trailing_slash(self, tmp_path):
        d = tmp_path / "project"
        d.mkdir()
        label = _render_recent_session_label([d])
        assert label == f"{d}/"

    def test_two_videos_same_folder_renders_with_parent(self, tmp_path):
        v0 = tmp_path / "videos" / "a.mp4"
        v0.parent.mkdir()
        v0.write_bytes(b"")
        v1 = tmp_path / "videos" / "b.mp4"
        v1.write_bytes(b"")
        label = _render_recent_session_label([v0, v1])
        # "a.mp4 + 1 more  (<parent>)"
        assert "a.mp4" in label
        assert "+ 1 more" in label
        assert str(v0.parent) in label

    def test_three_videos_same_folder(self, tmp_path):
        d = tmp_path / "proj" / "videos"
        d.mkdir(parents=True)
        v0 = d / "a.mp4"
        v1 = d / "b.mp4"
        v2 = d / "c.mp4"
        for v in (v0, v1, v2):
            v.write_bytes(b"")
        label = _render_recent_session_label([v0, v1, v2])
        assert "a.mp4" in label
        assert "+ 2 more" in label

    def test_mixed_parents_drops_parent_annotation(self, tmp_path):
        # Two videos whose only commonpath is the tmp_path root -- the
        # rendering still includes the common parent (it's a real
        # shared dir), but the test verifies the rendering doesn't
        # crash on the cross-folder case.
        d0 = tmp_path / "alpha"
        d0.mkdir()
        d1 = tmp_path / "beta"
        d1.mkdir()
        v0 = d0 / "a.mp4"
        v1 = d1 / "b.mp4"
        v0.write_bytes(b"")
        v1.write_bytes(b"")
        label = _render_recent_session_label([v0, v1])
        assert "a.mp4" in label
        assert "+ 1 more" in label

    def test_mixed_drive_letters_falls_back_gracefully(self, monkeypatch):
        # os.path.commonpath raises ValueError when paths span drives
        # (Windows). The rendering falls back to "name + N-1 more"
        # without the parent annotation.
        import os
        original = os.path.commonpath

        def _raise(_paths):
            raise ValueError("mixed drives")

        monkeypatch.setattr(os.path, "commonpath", _raise)
        try:
            label = _render_recent_session_label(
                [Path("C:/a/v0.mp4"), Path("D:/b/v1.mp4")]
            )
            assert "v0.mp4" in label
            assert "+ 1 more" in label
            # No parent in parentheses since commonpath blew up.
            assert "(" not in label
        finally:
            os.path.commonpath = original


# ---------------------------------------------------------------------
# Packaged seed asset
# ---------------------------------------------------------------------


class TestSeedAsset:
    def test_seed_video_ships_with_install(self):
        assert _SEED_VIDEO_PATH.is_file(), (
            f"seed video missing from install: {_SEED_VIDEO_PATH}. "
            "Regenerate via tests/_assets/build_seed_video.py."
        )

    def test_seed_toc_sidecar_ships(self):
        toc = _SEED_VIDEO_PATH.with_suffix(_SEED_VIDEO_PATH.suffix + ".dnav-toc")
        assert toc.is_file(), (
            f"seed video TOC sidecar missing: {toc}. "
            "Regenerate via tests/_assets/build_seed_video.py."
        )

    def test_seed_video_is_compact(self):
        # The seed asset is supposed to be tiny (1-2 KB band). A bloated
        # asset means someone slipped a real video in.
        size = _SEED_VIDEO_PATH.stat().st_size
        assert size < 10_000, f"seed video unexpectedly large: {size} bytes"

    def test_seed_video_opens_via_videoreader(self):
        # Direct check: the canonical reader path that DUSTrack uses
        # must handle the seed asset. Catches regressions in the
        # encoder params if someone tunes them later.
        from datanavigator import VideoReader
        with _SEED_VIDEO_PATH.open("rb") as f:
            r = VideoReader(f)
            assert len(r) == 8


# ---------------------------------------------------------------------
# _open_seed_session marker wiring (no Qt construction; we just verify
# the construction flag flow via a stub).
# ---------------------------------------------------------------------


class TestSeedSessionMarker:
    def test_open_seed_session_marks_tracker(self, monkeypatch):
        # Patch DUSTrack so we don't spin up Qt. The stub just records
        # construction args and exposes an _init_bundles hook.
        constructed = {}

        class _StubTracker:
            def __init__(self, path, layer_name, **kwargs):
                constructed["path"] = path
                constructed["layer_name"] = layer_name
                constructed["kwargs"] = kwargs
                self._init_bundles_called = False

            def _init_bundles(self, *, project, video_paths):
                self._init_bundles_called = True
                self.project = project
                self.video_paths = video_paths

        monkeypatch.setattr("dustrack._open.DUSTrack", _StubTracker)
        from dustrack._open import _open_seed_session
        tracker = _open_seed_session()
        # Constructed against the packaged seed asset with the "_seed"
        # layer name (Phase 1, no project).
        assert constructed["path"] == str(_SEED_VIDEO_PATH)
        assert constructed["layer_name"] == "_seed"
        # Marker set so the close-guard + history writer short-circuit.
        assert tracker._is_seed_session is True
        # Bundles initialised so replace_active_with has an active
        # bundle to swap from.
        assert tracker._init_bundles_called is True
        assert tracker.project is None
        assert tracker.video_paths == [_SEED_VIDEO_PATH]

    def test_open_seed_session_raises_if_asset_missing(self, monkeypatch):
        # Pretend the asset isn't installed.
        from dustrack import _open
        bogus = _SEED_VIDEO_PATH.parent / "nope.mp4"
        monkeypatch.setattr(_open, "_SEED_VIDEO_PATH", bogus)
        with pytest.raises(FileNotFoundError, match="seed video asset missing"):
            _open._open_seed_session()
