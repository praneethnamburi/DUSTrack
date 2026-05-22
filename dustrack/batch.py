"""Batch re-encode ultrasound videos as h265 monochrome (pix_fmt=gray).

Ultrasound is inherently grayscale, but the typical capture pipeline
stores it as yuv420p h264 with U/V planes carrying chroma noise that
gets mixed back into Y during BT.601 RGB->gray conversion downstream.
Re-encoding as h265 4:0:0 monochrome:

* drops the chroma planes entirely (parity_decoder.py 2026-05-21: chroma
  noise contribution to gray was ~1.4/255 mean per frame, DLC inference
  parity median = 0.19 px)
* unlocks dnav's ``pix_fmt='gray'`` auto-detect path (~6x sequential
  decode speedup on the LK / lk_filter paths)
* shrinks the file ~6% at CRF 22 vs the typical capture-side CRF
* speeds up TOC build ~20-40% (fewer + smaller packets per frame)

Codec choice: libx264 cannot produce true 4:0:0 in ffmpeg (silently
falls back to yuvj420p-with-constant-chroma). libx265 writes real
monochrome cleanly; cv2.VideoCapture and PyAV both handle h265 mono
without ceremony, and the 2026-05-21 cv2-vs-PyAV parity sweep confirms
they produce bit-exact identical decoded frames on this format.

Defaults (CRF 22, preset slow) match the immersionlab telemed
convention adjusted upward for h265's better compression efficiency.
On a 706x558 typical ultrasound clip these land within 6% of the
source file size.

Originals are never touched: output goes to
``<original_stem><suffix>.mp4`` next to each source. Recovery from a
bad batch = delete the ``*_mono.mp4`` outputs.

Example:

    >>> import dustrack
    >>> dustrack.convert_to_mono('S:/study/clip01.mp4')
    >>> # or a directory:
    >>> dustrack.convert_to_mono('S:/study/')
    >>> # or a list:
    >>> dustrack.convert_to_mono(['vid1.mp4', 'vid2.mp4'])
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Union


_MONO_PIX_FMTS = frozenset({
    "gray", "gray8", "gray9", "gray9le", "gray9be",
    "gray10", "gray10le", "gray10be",
    "gray12", "gray12le", "gray12be",
    "gray14", "gray14le", "gray14be",
    "gray16", "gray16le", "gray16be",
    "grayf32", "grayf32le", "grayf32be",
    "yuvj400p", "yuv400p",
})

_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v")


def _find_tool(name: str) -> str:
    """Locate an ffmpeg-family executable, accepting common Windows paths.

    ``shutil.which`` only checks PATH; on this machine ffmpeg lives at
    ``C:/ffmpeg/bin/`` and isn't always on PATH for sub-shells.
    """
    path = shutil.which(name)
    if path:
        return path
    for cand in (
        f"C:/ffmpeg/bin/{name}.exe",
        f"/c/ffmpeg/bin/{name}.exe",
    ):
        if Path(cand).is_file():
            return cand
    raise FileNotFoundError(
        f"{name} not found on PATH or at the usual Windows install location "
        f"(C:/ffmpeg/bin/{name}.exe). Install ffmpeg or pass the path "
        f"explicitly via the ``ffmpeg_bin`` kwarg."
    )


def _probe_source_pix_fmt(ffprobe: str, src: Path) -> str:
    """Return the source's encoded pix_fmt via ffprobe, or '' on failure."""
    res = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=pix_fmt", "-of", "csv=p=0", str(src)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        return ""
    return res.stdout.strip()


def _iter_sources(sources) -> list[Path]:
    """Normalise ``sources`` to a list of video file paths.

    Accepts a single path, an iterable of paths, or a directory (in
    which case all video files matching :data:`_VIDEO_EXTS` are walked
    recursively).
    """
    if isinstance(sources, (str, Path)):
        p = Path(sources)
        if p.is_dir():
            return sorted(
                fp for fp in p.rglob("*")
                if fp.is_file() and fp.suffix.lower() in _VIDEO_EXTS
            )
        return [p]
    return [Path(s) for s in sources]


def convert_to_mono(
    sources: Union[str, Path, Iterable[Union[str, Path]]],
    *,
    crf: int = 22,
    preset: str = "slow",
    suffix: str = "_mono",
    skip_existing: bool = True,
    skip_already_mono: bool = True,
    ffmpeg_bin: str | None = None,
    ffprobe_bin: str | None = None,
    verbose: bool = True,
) -> list[Path]:
    """Re-encode each source as h265 monochrome alongside the original.

    Args:
        sources: A video path, an iterable of video paths, or a directory
            (recursive walk of supported extensions). Originals are never
            modified -- each output is written as
            ``<source_stem><suffix>.mp4`` in the same folder.
        crf: libx265 quality. Lower = better quality, bigger file.
            Default 22 lands within 6% of typical ultrasound capture
            file size on 706x558 clips; DLC inference parity at CRF 22
            is sub-pixel (median 0.19 px). Pass a lower value (e.g. 18)
            for tighter pixel parity at a 50%+ file-size cost.
        preset: libx265 preset. Slower = better compression at same CRF.
            Default ``"slow"`` matches the immersionlab telemed
            convention.
        suffix: Appended to the source stem to form the output filename.
            Default ``"_mono"``.
        skip_existing: When True (default), skip sources whose output
            file already exists. Pass False to force re-encode.
        skip_already_mono: When True (default), skip sources whose
            encoded pix_fmt is already monochrome (no point re-encoding
            a mono file). Detected via ffprobe.
        ffmpeg_bin: Override ffmpeg path; default looks on PATH and at
            ``C:/ffmpeg/bin/ffmpeg.exe``.
        ffprobe_bin: Override ffprobe path; same lookup rules as ffmpeg.
        verbose: Print one status line per source.

    Returns:
        List of output paths actually written (skipped sources omitted).
    """
    ffmpeg = ffmpeg_bin or _find_tool("ffmpeg")
    ffprobe = ffprobe_bin or _find_tool("ffprobe")

    paths = _iter_sources(sources)
    if not paths:
        if verbose:
            print("convert_to_mono: no source files found.")
        return []

    written: list[Path] = []
    for src in paths:
        if not src.is_file():
            if verbose:
                print(f"  skip (missing): {src}")
            continue
        dst = src.with_name(f"{src.stem}{suffix}.mp4")
        if dst == src:
            # User chose suffix="" and source is already .mp4; refuse to
            # overwrite in-place.
            if verbose:
                print(f"  skip (would overwrite source): {src.name}")
            continue
        if skip_existing and dst.exists():
            if verbose:
                print(f"  skip (output exists): {dst.name}")
            continue
        if skip_already_mono:
            source_fmt = _probe_source_pix_fmt(ffprobe, src)
            if source_fmt in _MONO_PIX_FMTS:
                if verbose:
                    print(f"  skip (already mono, pix_fmt={source_fmt}): {src.name}")
                continue

        if verbose:
            print(f"  encoding {src.name} -> {dst.name} (crf={crf}, preset={preset})")
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "warning",
            "-i", str(src),
            "-c:v", "libx265",
            "-pix_fmt", "gray",
            "-crf", str(crf),
            "-preset", preset,
            "-fps_mode", "passthrough",
            "-an",
            "-y", str(dst),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            if verbose:
                print(
                    f"  FAILED: {src.name}\n"
                    f"  ffmpeg stderr (tail):\n  {res.stderr[-400:]}"
                )
            # Clean up a partial output if any.
            if dst.exists() and dst.stat().st_size == 0:
                dst.unlink()
            continue
        if verbose:
            src_mb = src.stat().st_size / 1e6
            dst_mb = dst.stat().st_size / 1e6
            ratio = dst_mb / src_mb if src_mb else 0.0
            print(f"  OK: {dst.name}  ({dst_mb:.1f} MB, {ratio*100:.0f}% of source)")
        written.append(dst)
    return written
