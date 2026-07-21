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


class TestDestNames:
    """Renaming at link time.

    A hard link is a second directory entry for the same bytes, so the
    in-project name is free to differ from the source's -- no copy, no
    extra storage. pia02 needs this: its telemed exports carry spaces
    and long descriptive names, and DLC appends ~58 characters of
    scorer/snapshot suffix on top, which crowds Windows' 260-character
    path limit.
    """

    def test_renames_video_and_sidecar(self, tmp_path):
        src = _make_video(tmp_path / "src" / "pia02_s061_003 fav piece_b2.mp4")
        (tmp_path / "src" / "pia02_s061_003 fav piece_b2.mp4.dnav-toc").write_bytes(
            b"toc"
        )
        proj_videos = tmp_path / "proj" / "videos"
        result = link_or_copy_videos_into_project(
            proj_videos, [src], dest_names=["pia02_s061_003_LFAc.mp4"]
        )
        assert [p.name for p in result] == ["pia02_s061_003_LFAc.mp4"]
        assert (proj_videos / "pia02_s061_003_LFAc.mp4").exists()
        # The TOC follows the *destination* name, else the reader rebuilds it.
        assert (proj_videos / "pia02_s061_003_LFAc.mp4.dnav-toc").read_bytes() == b"toc"

    def test_content_is_the_source_content(self, tmp_path):
        src = _make_video(tmp_path / "src" / "long name.mp4", payload=b"REAL")
        proj_videos = tmp_path / "proj" / "videos"
        result = link_or_copy_videos_into_project(
            proj_videos, [src], dest_names=["short.mp4"]
        )
        assert result[0].read_bytes() == b"REAL"

    def test_order_is_positional(self, tmp_path):
        srcs = [
            _make_video(tmp_path / "src" / f"v{i}.mp4", payload=f"v{i}".encode())
            for i in range(3)
        ]
        proj_videos = tmp_path / "proj" / "videos"
        result = link_or_copy_videos_into_project(
            proj_videos, srcs, dest_names=["c.mp4", "a.mp4", "b.mp4"]
        )
        assert [p.name for p in result] == ["c.mp4", "a.mp4", "b.mp4"]
        # Positional, not sorted: v0 -> c.mp4.
        assert (proj_videos / "c.mp4").read_bytes() == b"v0"

    def test_length_mismatch_raises(self, tmp_path):
        srcs = [_make_video(tmp_path / "src" / f"v{i}.mp4") for i in range(2)]
        proj_videos = tmp_path / "proj" / "videos"
        with pytest.raises(ValueError, match="1:1"):
            link_or_copy_videos_into_project(
                proj_videos, srcs, dest_names=["only_one.mp4"]
            )

    def test_none_keeps_source_names(self, tmp_path):
        src = _make_video(tmp_path / "src" / "keepme.mp4")
        proj_videos = tmp_path / "proj" / "videos"
        result = link_or_copy_videos_into_project(proj_videos, [src], dest_names=None)
        assert [p.name for p in result] == ["keepme.mp4"]


class TestSymlinkAsHardlink:
    """DLC's create_new_project falls back to copying; this makes its
    first attempt (os.symlink) succeed as a hard link instead."""

    def test_symlink_becomes_a_hardlink(self, tmp_path):
        from dustrack._dlc_paths import symlink_as_hardlink

        src = _make_video(tmp_path / "src.mp4", payload=b"BYTES")
        dst = tmp_path / "dst.mp4"
        with symlink_as_hardlink():
            os.symlink(str(src), str(dst))
        assert dst.read_bytes() == b"BYTES"
        assert not dst.is_symlink()          # a real directory entry
        assert os.stat(src).st_ino == os.stat(dst).st_ino

    def test_restores_os_symlink_afterwards(self, tmp_path):
        from dustrack._dlc_paths import symlink_as_hardlink

        original = os.symlink
        with symlink_as_hardlink():
            assert os.symlink is not original
        assert os.symlink is original

    def test_restores_even_on_exception(self, tmp_path):
        from dustrack._dlc_paths import symlink_as_hardlink

        original = os.symlink
        with pytest.raises(RuntimeError):
            with symlink_as_hardlink():
                raise RuntimeError("boom")
        assert os.symlink is original

    def test_falls_back_to_real_symlink_when_link_fails(self, tmp_path):
        """Cross-volume os.link genuinely cannot work -- DLC's original
        ladder must still be reachable."""
        from dustrack._dlc_paths import symlink_as_hardlink

        called = {}

        def fake_symlink(src, dst, *a, **k):
            called["yes"] = (src, dst)

        with patch("dustrack._dlc_paths.os.link", side_effect=OSError("xdev")):
            real = os.symlink
            os.symlink = fake_symlink
            try:
                with symlink_as_hardlink():
                    os.symlink("a", "b")
            finally:
                os.symlink = real
        assert called["yes"] == ("a", "b")
