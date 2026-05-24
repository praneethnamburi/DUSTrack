"""Tests for :func:`link_or_copy_videos_into_project`.

The helper is the load-bearing piece of the 1.3.0a2 hardlink-by-default
flow: it places each source video inside ``<project>/videos/`` as a
hardlink when source + project share a volume, falling back to a copy
otherwise. These tests pin the link / copy / fallback matrix and the
sidecar pass-through.

DLC is *not* required to exercise the helper -- it's pure filesystem.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from dustrack._dlc_paths import link_or_copy_videos_into_project


def _make_video(path: Path, payload: bytes = b"video-bytes") -> Path:
    """Write a tiny file at ``path`` and return it (placeholder video)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _same_inode(a: Path, b: Path) -> bool:
    return os.stat(a).st_ino == os.stat(b).st_ino and os.stat(a).st_ino != 0


class TestSameVolumeHardlink:
    """Source + dest share a volume -- default mode hard-links via os.link."""

    def test_link_creates_in_project_path(self, tmp_path):
        src = _make_video(tmp_path / "src" / "v0.mp4")
        proj_videos = tmp_path / "proj" / "videos"
        result = link_or_copy_videos_into_project(proj_videos, [src])
        assert len(result) == 1
        assert result[0] == proj_videos / "v0.mp4"
        assert result[0].exists()

    def test_link_shares_inode(self, tmp_path):
        src = _make_video(tmp_path / "src" / "v0.mp4")
        proj_videos = tmp_path / "proj" / "videos"
        [dst] = link_or_copy_videos_into_project(proj_videos, [src])
        assert _same_inode(src, dst)

    def test_link_preserves_bytes(self, tmp_path):
        src = _make_video(tmp_path / "src" / "v0.mp4", payload=b"hello world")
        proj_videos = tmp_path / "proj" / "videos"
        [dst] = link_or_copy_videos_into_project(proj_videos, [src])
        assert dst.read_bytes() == b"hello world"

    def test_link_keeps_original_filename(self, tmp_path):
        src = _make_video(tmp_path / "src" / "weirdly_named_file.mp4")
        proj_videos = tmp_path / "proj" / "videos"
        [dst] = link_or_copy_videos_into_project(proj_videos, [src])
        assert dst.name == "weirdly_named_file.mp4"

    def test_creates_project_videos_dir(self, tmp_path):
        src = _make_video(tmp_path / "src" / "v0.mp4")
        proj_videos = tmp_path / "proj" / "videos"  # does NOT exist yet
        assert not proj_videos.exists()
        link_or_copy_videos_into_project(proj_videos, [src])
        assert proj_videos.is_dir()


class TestSidecarPassThrough:
    """``.dnav-toc`` sidecar (and arbitrary registered sidecars) ride along."""

    def test_dnav_toc_sidecar_linked(self, tmp_path):
        src = _make_video(tmp_path / "src" / "v0.mp4")
        sidecar = _make_video(
            tmp_path / "src" / "v0.mp4.dnav-toc", payload=b"toc-bytes"
        )
        proj_videos = tmp_path / "proj" / "videos"
        [dst] = link_or_copy_videos_into_project(proj_videos, [src])
        toc_dst = dst.with_name(dst.name + ".dnav-toc")
        assert toc_dst.exists()
        assert _same_inode(sidecar, toc_dst)

    def test_missing_sidecar_silent(self, tmp_path):
        """Source without a sidecar: no error, no sidecar at dst."""
        src = _make_video(tmp_path / "src" / "v0.mp4")
        # No sidecar at <src>.dnav-toc
        proj_videos = tmp_path / "proj" / "videos"
        [dst] = link_or_copy_videos_into_project(proj_videos, [src])
        toc_dst = dst.with_name(dst.name + ".dnav-toc")
        assert not toc_dst.exists()
        # Only the video itself lands in proj_videos
        assert sorted(p.name for p in proj_videos.iterdir()) == ["v0.mp4"]

    def test_custom_sidecar_list(self, tmp_path):
        src = _make_video(tmp_path / "src" / "v0.mp4")
        meta = _make_video(tmp_path / "src" / "v0.mp4.meta", payload=b"meta")
        proj_videos = tmp_path / "proj" / "videos"
        [dst] = link_or_copy_videos_into_project(
            proj_videos, [src], link_sidecars=(".meta",)
        )
        meta_dst = dst.with_name(dst.name + ".meta")
        assert meta_dst.exists()
        # Default .dnav-toc lookup didn't fire on this call
        assert not dst.with_name(dst.name + ".dnav-toc").exists()


class TestIdempotent:
    """Re-running on the same args is a no-op (no overwrite, no error)."""

    def test_rerun_no_error(self, tmp_path):
        src = _make_video(tmp_path / "src" / "v0.mp4")
        proj_videos = tmp_path / "proj" / "videos"
        link_or_copy_videos_into_project(proj_videos, [src])
        # Second call must not raise
        result = link_or_copy_videos_into_project(proj_videos, [src])
        assert result[0] == proj_videos / "v0.mp4"

    def test_rerun_preserves_inode(self, tmp_path):
        src = _make_video(tmp_path / "src" / "v0.mp4")
        proj_videos = tmp_path / "proj" / "videos"
        [dst1] = link_or_copy_videos_into_project(proj_videos, [src])
        ino_before = os.stat(dst1).st_ino
        [dst2] = link_or_copy_videos_into_project(proj_videos, [src])
        assert dst1 == dst2
        assert os.stat(dst2).st_ino == ino_before


class TestExistingCopyReplacement:
    """Mirrors the DLC-side flow: ``create_new_project(copy_videos=False)``
    on Windows-without-symlink-privilege leaves a real copy at
    ``<project>/videos/<stem>.<ext>``. The helper must replace that
    copy with a hard link to the source so disk usage drops to one
    inode."""

    def test_replaces_byte_equal_copy_with_hardlink(self, tmp_path):
        src = _make_video(tmp_path / "src" / "v0.mp4", payload=b"abc" * 1000)
        proj_videos = tmp_path / "proj" / "videos"
        proj_videos.mkdir(parents=True)
        # Simulate DLC's fall-back copy: same bytes, different inode
        dst_pre = proj_videos / "v0.mp4"
        shutil.copy(src, dst_pre)
        assert not _same_inode(src, dst_pre)  # confirm setup
        # Helper must replace the copy with a hard link
        [dst] = link_or_copy_videos_into_project(proj_videos, [src])
        assert dst == dst_pre
        assert _same_inode(src, dst)

    def test_refuses_to_overwrite_different_size_file(self, tmp_path):
        src = _make_video(tmp_path / "src" / "v0.mp4", payload=b"x" * 1000)
        proj_videos = tmp_path / "proj" / "videos"
        proj_videos.mkdir(parents=True)
        # Pre-existing dst with a different file size
        (proj_videos / "v0.mp4").write_bytes(b"completely different bytes")
        with pytest.raises(FileExistsError):
            link_or_copy_videos_into_project(proj_videos, [src])


class TestForceCopy:
    """``link_videos=False`` always uses shutil.copy2, even on same volume."""

    def test_force_copy_different_inode(self, tmp_path):
        src = _make_video(tmp_path / "src" / "v0.mp4")
        proj_videos = tmp_path / "proj" / "videos"
        [dst] = link_or_copy_videos_into_project(
            proj_videos, [src], link_videos=False
        )
        assert dst.exists()
        assert dst.read_bytes() == src.read_bytes()
        assert not _same_inode(src, dst)

    def test_force_copy_sidecar(self, tmp_path):
        src = _make_video(tmp_path / "src" / "v0.mp4")
        sidecar = _make_video(tmp_path / "src" / "v0.mp4.dnav-toc")
        proj_videos = tmp_path / "proj" / "videos"
        [dst] = link_or_copy_videos_into_project(
            proj_videos, [src], link_videos=False
        )
        toc_dst = dst.with_name(dst.name + ".dnav-toc")
        assert toc_dst.exists()
        assert not _same_inode(sidecar, toc_dst)


class TestCrossVolumeFallback:
    """``os.link`` raises OSError on cross-volume; auto mode falls back to copy."""

    def test_auto_mode_falls_back_to_copy(self, tmp_path, capsys):
        src = _make_video(tmp_path / "src" / "v0.mp4")
        proj_videos = tmp_path / "proj" / "videos"
        # Simulate cross-volume by making os.link raise OSError every call.
        cross_volume_err = OSError(17, "cross-volume")
        with patch("dustrack._dlc_paths.os.link", side_effect=cross_volume_err):
            [dst] = link_or_copy_videos_into_project(proj_videos, [src])
        # Fell back to copy: dst exists with the same bytes but different inode
        assert dst.exists()
        assert dst.read_bytes() == src.read_bytes()
        # Stderr warning was emitted
        captured = capsys.readouterr()
        assert "cross-volume" in captured.err or "hard link" in captured.err

    def test_link_true_raises_oserror(self, tmp_path):
        src = _make_video(tmp_path / "src" / "v0.mp4")
        proj_videos = tmp_path / "proj" / "videos"
        with patch(
            "dustrack._dlc_paths.os.link", side_effect=OSError(17, "cross-volume")
        ):
            with pytest.raises(OSError):
                link_or_copy_videos_into_project(
                    proj_videos, [src], link_videos=True
                )

    def test_force_copy_does_not_call_os_link(self, tmp_path):
        src = _make_video(tmp_path / "src" / "v0.mp4")
        proj_videos = tmp_path / "proj" / "videos"
        with patch("dustrack._dlc_paths.os.link") as mock_link:
            link_or_copy_videos_into_project(
                proj_videos, [src], link_videos=False
            )
        mock_link.assert_not_called()


class TestMultipleVideos:
    """Lists of videos preserve order and 1-to-1 destination mapping."""

    def test_multiple_preserves_order(self, tmp_path):
        srcs = [
            _make_video(tmp_path / "src" / f"v{i}.mp4", payload=f"v{i}".encode())
            for i in range(3)
        ]
        proj_videos = tmp_path / "proj" / "videos"
        result = link_or_copy_videos_into_project(proj_videos, srcs)
        assert [p.name for p in result] == ["v0.mp4", "v1.mp4", "v2.mp4"]
        for src, dst in zip(srcs, result):
            assert dst.read_bytes() == src.read_bytes()

    def test_empty_list(self, tmp_path):
        proj_videos = tmp_path / "proj" / "videos"
        result = link_or_copy_videos_into_project(proj_videos, [])
        assert result == []
        # Still creates the directory (downstream caller expects it)
        assert proj_videos.is_dir()
