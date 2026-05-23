"""Tests for ``dustrack.batch.build_toc`` and ``propagate_toc_to_dlc_project``.

Covers the thin pass-through to ``datanavigator.precompute_toc_folder``
and the DLC-project resolver: project-root vs config.yaml acceptance,
missing config.yaml, missing videos/ subfolder.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from dustrack.batch import (
    _resolve_dlc_videos_dir,
    build_toc,
    propagate_toc_to_dlc_project,
)


FPS = 24
DURATION_S = 1


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None or Path("C:/ffmpeg/bin/ffmpeg.exe").is_file()


pytestmark = pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not available")


def _make_clip(out: Path, *, duration: float = DURATION_S) -> Path:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=64x48:rate={FPS}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "12",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


# ---------- build_toc ----------


def test_build_toc_folder(tmp_path):
    folder = tmp_path / "vids"
    folder.mkdir()
    a = _make_clip(folder / "a.mp4")
    b = _make_clip(folder / "b.mp4", duration=0.5)

    results = build_toc(folder, show_progress=False)
    assert results[str(a)] == "built"
    assert results[str(b)] == "built"
    assert Path(str(a) + ".dnav-toc").exists()
    assert Path(str(b) + ".dnav-toc").exists()


def test_build_toc_single_file(tmp_path):
    a = _make_clip(tmp_path / "solo.mp4")
    results = build_toc(a, show_progress=False)
    assert results[str(a)] == "built"


def test_build_toc_second_call_is_hit(tmp_path):
    folder = tmp_path / "vids"
    folder.mkdir()
    a = _make_clip(folder / "a.mp4")

    build_toc(folder, show_progress=False)
    second = build_toc(folder, show_progress=False)
    assert second[str(a)] == "hit"


def test_build_toc_non_recursive_skips_subdirs(tmp_path):
    folder = tmp_path / "vids"
    sub = folder / "session"
    sub.mkdir(parents=True)
    a = _make_clip(folder / "top.mp4")
    b = _make_clip(sub / "nested.mp4")

    results = build_toc(folder, recursive=False, show_progress=False)
    assert results[str(a)] == "built"
    assert str(b) not in results


def test_build_toc_force_rebuilds(tmp_path):
    folder = tmp_path / "vids"
    folder.mkdir()
    a = _make_clip(folder / "a.mp4")
    build_toc(folder, show_progress=False)
    rebuilt = build_toc(folder, force=True, show_progress=False)
    assert rebuilt[str(a)] == "built"


def test_build_toc_missing_path_reported(tmp_path):
    missing = tmp_path / "nope"
    results = build_toc(missing, show_progress=False)
    assert results == {str(missing): "error: missing"}


def test_build_toc_custom_extensions(tmp_path):
    folder = tmp_path / "vids"
    folder.mkdir()
    a = _make_clip(folder / "ok.mp4")
    other = folder / "ignored.mkv"
    _make_clip(other)

    results = build_toc(folder, extensions=(".mp4",), show_progress=False)
    assert results[str(a)] == "built"
    assert str(other) not in results


def test_build_toc_progress_callback_fires_per_file(tmp_path):
    """The driven path (``progress_callback`` set) iterates per file
    and emits the same statuses dnav reports."""
    folder = tmp_path / "vids"
    folder.mkdir()
    a = _make_clip(folder / "a.mp4")
    b = _make_clip(folder / "b.mp4", duration=0.5)

    calls: list[tuple] = []

    def cb(idx, total, path, status):
        calls.append((idx, total, path.name, status))

    results = build_toc(folder, progress_callback=cb)
    # Both files reported.
    assert len(calls) == 2
    assert {c[2] for c in calls} == {"a.mp4", "b.mp4"}
    # Status is "built" on the fresh run.
    assert all(c[3] == "built" for c in calls)
    assert results[str(a)] == "built"
    assert results[str(b)] == "built"


def test_build_toc_cancel_check_aborts(tmp_path):
    folder = tmp_path / "vids"
    folder.mkdir()
    _make_clip(folder / "a.mp4")
    _make_clip(folder / "b.mp4", duration=0.5)
    _make_clip(folder / "c.mp4", duration=0.5)

    seen: list[str] = []

    def cb(idx, total, path, status):
        seen.append(path.name)

    state = {"calls": 0}

    def cancel_check():
        state["calls"] += 1
        return state["calls"] >= 2  # True on the second file's top-check

    build_toc(folder, progress_callback=cb, cancel_check=cancel_check)
    # Only the first file is processed.
    assert len(seen) == 1


# ---------- _resolve_dlc_videos_dir ----------


def _make_synthetic_dlc_project(root: Path) -> Path:
    """Create a minimal {root}/{config.yaml, videos/} layout."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text("# synthetic DLC project for test\n")
    (root / "videos").mkdir()
    return root


def test_resolve_videos_dir_from_project_root(tmp_path):
    project = _make_synthetic_dlc_project(tmp_path / "myproj")
    videos = _resolve_dlc_videos_dir(project)
    assert videos == project / "videos"


def test_resolve_videos_dir_from_config_path(tmp_path):
    project = _make_synthetic_dlc_project(tmp_path / "myproj")
    videos = _resolve_dlc_videos_dir(project / "config.yaml")
    assert videos == project / "videos"


def test_resolve_videos_dir_rejects_non_project_folder(tmp_path):
    bogus = tmp_path / "not_a_project"
    bogus.mkdir()
    with pytest.raises(FileNotFoundError, match="Not a DLC project root"):
        _resolve_dlc_videos_dir(bogus)


def test_resolve_videos_dir_rejects_project_without_videos_dir(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "config.yaml").write_text("# stub\n")
    # videos/ deliberately missing.
    with pytest.raises(FileNotFoundError, match="has no videos/ subfolder"):
        _resolve_dlc_videos_dir(project)


def test_resolve_videos_dir_missing_entirely(tmp_path):
    with pytest.raises(FileNotFoundError):
        _resolve_dlc_videos_dir(tmp_path / "ghost")


# ---------- propagate_toc_to_dlc_project ----------


def test_propagate_builds_tocs_in_project_videos(tmp_path):
    project = _make_synthetic_dlc_project(tmp_path / "myproj")
    a = _make_clip(project / "videos" / "v0.mp4")
    b = _make_clip(project / "videos" / "v1.mp4", duration=0.5)

    results = propagate_toc_to_dlc_project(project, show_progress=False)
    assert results[str(a)] == "built"
    assert results[str(b)] == "built"
    assert Path(str(a) + ".dnav-toc").exists()
    assert Path(str(b) + ".dnav-toc").exists()


def test_propagate_accepts_config_path(tmp_path):
    project = _make_synthetic_dlc_project(tmp_path / "myproj")
    a = _make_clip(project / "videos" / "v0.mp4")
    results = propagate_toc_to_dlc_project(project / "config.yaml", show_progress=False)
    assert results[str(a)] == "built"


def test_propagate_idempotent(tmp_path):
    project = _make_synthetic_dlc_project(tmp_path / "myproj")
    _make_clip(project / "videos" / "v0.mp4")
    first = propagate_toc_to_dlc_project(project, show_progress=False)
    second = propagate_toc_to_dlc_project(project, show_progress=False)
    assert all(s == "built" for s in first.values())
    assert all(s == "hit" for s in second.values())


def test_propagate_empty_videos_dir_returns_empty(tmp_path):
    project = _make_synthetic_dlc_project(tmp_path / "myproj")
    results = propagate_toc_to_dlc_project(project, show_progress=False)
    assert results == {}


def test_propagate_handles_nested_videos_subdirs(tmp_path):
    """A hand-organized project with subfolders under videos/ — propagate
    walks recursively so all clips get TOCs."""
    project = _make_synthetic_dlc_project(tmp_path / "myproj")
    sub = project / "videos" / "session_01"
    sub.mkdir()
    a = _make_clip(project / "videos" / "flat.mp4")
    b = _make_clip(sub / "nested.mp4", duration=0.5)
    results = propagate_toc_to_dlc_project(project, show_progress=False)
    assert results[str(a)] == "built"
    assert results[str(b)] == "built"


def test_propagate_raises_on_missing_project(tmp_path):
    with pytest.raises(FileNotFoundError):
        propagate_toc_to_dlc_project(tmp_path / "no_such_project", show_progress=False)
