"""Tests for ``dustrack.convert_to_mono``.

Exercises the batch ffmpeg invocation surface: single file, directory
walk, skip-existing, skip-already-mono, output naming.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from dustrack.batch import convert_to_mono, _MONO_PIX_FMTS


FPS = 24
DURATION_S = 1


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None or Path("C:/ffmpeg/bin/ffmpeg.exe").is_file()


pytestmark = pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not available")


def _make_color_clip(out: Path) -> Path:
    """Tiny h264 yuv420p clip, the typical ultrasound capture format."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=duration={DURATION_S}:size=64x48:rate={FPS}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def _make_mono_clip(out: Path) -> Path:
    """Tiny h265 monochrome clip."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=duration={DURATION_S}:size=64x48:rate={FPS}",
        "-c:v", "libx265", "-pix_fmt", "gray", "-x265-params", "log-level=none",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def _probe_pix_fmt(path: Path) -> str:
    ffprobe = shutil.which("ffprobe") or "C:/ffmpeg/bin/ffprobe.exe"
    res = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=pix_fmt", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return res.stdout.strip()


def test_single_color_file_produces_mono(tmp_path, capsys):
    src = _make_color_clip(tmp_path / "src.mp4")
    out = convert_to_mono(src, verbose=False)
    assert len(out) == 1
    assert out[0] == src.with_name("src_mono.mp4")
    assert out[0].is_file()
    assert _probe_pix_fmt(out[0]) in _MONO_PIX_FMTS
    # Original untouched.
    assert src.is_file()


def test_skip_existing(tmp_path):
    src = _make_color_clip(tmp_path / "src.mp4")
    # First call writes.
    first = convert_to_mono(src, verbose=False)
    assert len(first) == 1
    mtime_first = first[0].stat().st_mtime
    # Second call sees existing output and skips.
    second = convert_to_mono(src, verbose=False)
    assert second == []
    assert first[0].stat().st_mtime == mtime_first


def test_skip_already_mono(tmp_path):
    src = _make_mono_clip(tmp_path / "already_mono.mp4")
    out = convert_to_mono(src, verbose=False)
    assert out == []
    # Skip-already-mono can be disabled.
    out = convert_to_mono(src, verbose=False, skip_already_mono=False)
    assert len(out) == 1


def test_directory_walk(tmp_path):
    _make_color_clip(tmp_path / "a.mp4")
    _make_color_clip(tmp_path / "b.mp4")
    out = convert_to_mono(tmp_path, verbose=False)
    names = sorted(p.name for p in out)
    assert names == ["a_mono.mp4", "b_mono.mp4"]


def test_iterable_of_paths(tmp_path):
    s1 = _make_color_clip(tmp_path / "v1.mp4")
    s2 = _make_color_clip(tmp_path / "v2.mp4")
    out = convert_to_mono([s1, s2], verbose=False)
    assert len(out) == 2


def test_missing_source_is_skipped(tmp_path):
    out = convert_to_mono(tmp_path / "does_not_exist.mp4", verbose=False)
    assert out == []


def test_empty_suffix_refuses_overwrite(tmp_path):
    """suffix='' would target the source path itself; we must refuse."""
    src = _make_color_clip(tmp_path / "src.mp4")
    out = convert_to_mono(src, suffix="", verbose=False)
    assert out == []
    # Source unchanged.
    assert src.is_file()


def test_progress_callback_fires_per_file(tmp_path):
    """``progress_callback`` is invoked once per source with its
    per-file status (``"ok"`` / ``"skip_*"`` / ``"failed"``)."""
    s1 = _make_color_clip(tmp_path / "v1.mp4")
    s2 = _make_color_clip(tmp_path / "v2.mp4")
    # Pre-encode v2's mono output so it triggers the skip-existing path.
    convert_to_mono(s2, verbose=False)

    calls: list[tuple] = []

    def cb(idx, total, src, status):
        calls.append((idx, total, src.name, status))

    convert_to_mono([s1, s2], verbose=False, progress_callback=cb)
    # One call per source, in order.
    assert [c[0] for c in calls] == [0, 1]
    assert all(c[1] == 2 for c in calls)
    statuses = [c[3] for c in calls]
    assert "ok" in statuses
    assert "skip_existing" in statuses


def test_cancel_check_aborts_between_files(tmp_path):
    """A ``cancel_check`` returning True at file N halts the loop
    before file N runs."""
    s1 = _make_color_clip(tmp_path / "v1.mp4")
    s2 = _make_color_clip(tmp_path / "v2.mp4")
    s3 = _make_color_clip(tmp_path / "v3.mp4")

    seen: list[str] = []

    def cb(idx, total, src, status):
        seen.append(src.name)

    # Cancel after the first file is processed.
    state = {"calls": 0}

    def cancel_check():
        # Returns True starting on the *second* invocation (top of file 2).
        state["calls"] += 1
        return state["calls"] >= 2

    out = convert_to_mono(
        [s1, s2, s3],
        verbose=False,
        progress_callback=cb,
        cancel_check=cancel_check,
    )
    # v1 processed; v2, v3 short-circuited.
    assert len(out) == 1
    assert seen == ["v1.mp4"]


def test_show_progress_does_not_crash(tmp_path):
    """``show_progress=True`` uses tqdm internally and must not crash
    even when tqdm output is captured."""
    s1 = _make_color_clip(tmp_path / "v1.mp4")
    out = convert_to_mono(s1, verbose=False, show_progress=True)
    assert len(out) == 1
