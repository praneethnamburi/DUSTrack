"""Tests for ``dustrack.batch.build_toc``.

Covers the thin pass-through to ``datanavigator.precompute_toc_folder``
plus the callback-driven per-file iteration path (progress + cancel
hooks used by the batch-process modal).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from dustrack.batch import build_toc


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


