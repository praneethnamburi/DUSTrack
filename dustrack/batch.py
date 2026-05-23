"""Batch utilities for ultrasound video corpora.

Two workflows live here:

* :func:`convert_to_mono` — re-encode ultrasound clips as h265 monochrome
  (drops chroma noise; unlocks dnav's ``pix_fmt='gray'`` auto-detect for
  ~6x sequential-decode speedup). See the original module header below.

* :func:`build_toc` — pre-build the PyAV+TOC sidecar
  (``<video>.dnav-toc``) for every video in a folder. First DUSTrack
  open of a video pays the per-file TOC build cost (a full sequential
  demux to record per-packet offsets + per-frame timestamps); pre-
  building means ``dustrack.open(...)`` returns essentially instantly
  on warm folders. Delegates to :func:`datanavigator.precompute_toc_folder`
  so other portfolio consumers can hit the same code path. For DLC
  projects, point this at ``<project>/videos`` with ``recursive=False``.

Original convert_to_mono docstring follows:

Batch re-encode ultrasound videos as h265 monochrome (pix_fmt=gray).

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
from typing import Callable, Iterable, Optional, Union

import datanavigator as dnav


_MONO_PIX_FMTS = frozenset(
    {
        "gray",
        "gray8",
        "gray9",
        "gray9le",
        "gray9be",
        "gray10",
        "gray10le",
        "gray10be",
        "gray12",
        "gray12le",
        "gray12be",
        "gray14",
        "gray14le",
        "gray14be",
        "gray16",
        "gray16le",
        "gray16be",
        "grayf32",
        "grayf32le",
        "grayf32be",
        "yuvj400p",
        "yuv400p",
    }
)

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
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=pix_fmt",
            "-of",
            "csv=p=0",
            str(src),
        ],
        capture_output=True,
        text=True,
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
                fp
                for fp in p.rglob("*")
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
    show_progress: bool = False,
    progress_callback: Optional[Callable[[int, int, Path, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
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
        show_progress: If True, wrap the per-file loop in a tqdm bar
            (per-file status lines are routed through ``tqdm.write`` so
            the bar stays clean). Matches the ``show_progress`` kwarg on
            :func:`build_toc`. Off by default to preserve the historical
            print-only behaviour for shell users.
        progress_callback: Optional ``fn(idx, total, src_path, status)``
            invoked after each source. ``status`` is one of ``"ok"``,
            ``"skip_missing"``, ``"skip_overwrite"``, ``"skip_existing"``,
            ``"skip_already_mono"``, or ``"failed"``. Used by the Qt
            batch-process modal to drive its own progress UI without
            relying on tqdm.
        cancel_check: Optional zero-arg callable polled at the top of
            each source. If it returns truthy, the loop exits early.
            Used by the Qt batch-process modal's Cancel button.

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

    bar = None
    if show_progress:
        try:
            from tqdm import tqdm

            bar = tqdm(total=len(paths), desc="Converting to mono", unit="video")
        except ImportError:
            bar = None

    def _say(msg: str) -> None:
        if not verbose:
            return
        if bar is not None:
            bar.write(msg)
        else:
            print(msg)

    written: list[Path] = []
    total = len(paths)
    try:
        for idx, src in enumerate(paths):
            if cancel_check is not None and cancel_check():
                _say("convert_to_mono: cancelled.")
                break
            if not src.is_file():
                _say(f"  skip (missing): {src}")
                if progress_callback is not None:
                    progress_callback(idx, total, src, "skip_missing")
                if bar is not None:
                    bar.update(1)
                continue
            dst = src.with_name(f"{src.stem}{suffix}.mp4")
            if dst == src:
                # User chose suffix="" and source is already .mp4; refuse to
                # overwrite in-place.
                _say(f"  skip (would overwrite source): {src.name}")
                if progress_callback is not None:
                    progress_callback(idx, total, src, "skip_overwrite")
                if bar is not None:
                    bar.update(1)
                continue
            if skip_existing and dst.exists():
                _say(f"  skip (output exists): {dst.name}")
                if progress_callback is not None:
                    progress_callback(idx, total, src, "skip_existing")
                if bar is not None:
                    bar.update(1)
                continue
            if skip_already_mono:
                source_fmt = _probe_source_pix_fmt(ffprobe, src)
                if source_fmt in _MONO_PIX_FMTS:
                    _say(f"  skip (already mono, pix_fmt={source_fmt}): {src.name}")
                    if progress_callback is not None:
                        progress_callback(idx, total, src, "skip_already_mono")
                    if bar is not None:
                        bar.update(1)
                    continue

            _say(f"  encoding {src.name} -> {dst.name} (crf={crf}, preset={preset})")
            cmd = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "warning",
                "-i",
                str(src),
                "-c:v",
                "libx265",
                "-pix_fmt",
                "gray",
                "-crf",
                str(crf),
                "-preset",
                preset,
                "-fps_mode",
                "passthrough",
                "-an",
                "-y",
                str(dst),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                _say(
                    f"  FAILED: {src.name}\n"
                    f"  ffmpeg stderr (tail):\n  {res.stderr[-400:]}"
                )
                # Clean up a partial output if any.
                if dst.exists() and dst.stat().st_size == 0:
                    dst.unlink()
                if progress_callback is not None:
                    progress_callback(idx, total, src, "failed")
                if bar is not None:
                    bar.update(1)
                continue
            if verbose:
                src_mb = src.stat().st_size / 1e6
                dst_mb = dst.stat().st_size / 1e6
                ratio = dst_mb / src_mb if src_mb else 0.0
                _say(f"  OK: {dst.name}  ({dst_mb:.1f} MB, {ratio*100:.0f}% of source)")
            written.append(dst)
            if progress_callback is not None:
                progress_callback(idx, total, src, "ok")
            if bar is not None:
                bar.update(1)
    finally:
        if bar is not None:
            bar.close()
    return written


# ---------- TOC pre-build ----------


def build_toc(
    sources: Union[str, Path, Iterable[Union[str, Path]]],
    *,
    extensions: Iterable[str] = dnav.DEFAULT_VIDEO_EXTENSIONS,
    recursive: bool = True,
    force: bool = False,
    show_progress: bool = True,
    progress_callback: Optional[Callable[[int, int, Path, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> dict:
    """Pre-build the PyAV+TOC sidecar for every video under ``sources``.

    Thin pass-through to :func:`datanavigator.precompute_toc_folder` so
    DUSTrack callers can stay inside the ``dustrack`` namespace. First
    open of a video pays the per-file TOC build cost; pre-building means
    ``dustrack.open(...)`` returns essentially instantly afterward.

    Example::

        import dustrack
        dustrack.batch.build_toc("M:/us_videos_for_tracking2")

    Args:
        sources: A directory, a video file, or an iterable mixing both.
            Directories are walked for ``extensions``.
        extensions: File extensions to include (case-insensitive). Default
            matches dnav's :data:`~datanavigator.DEFAULT_VIDEO_EXTENSIONS`.
        recursive: If True (default), recurse into subdirectories.
        force: If True, rebuild even when a valid cache exists. Useful for
            upgrading pre-1.3 sidecars to schema v2 with per-frame
            timestamps.
        show_progress: If True (default), wrap iteration in a tqdm bar.
            Suppressed automatically when ``progress_callback`` is set.
        progress_callback: Optional ``fn(idx, total, video_path, status)``
            invoked after each video. ``status`` is the per-file result
            string from :func:`datanavigator.precompute_toc` (``"hit"``,
            ``"built"``, ``"built (uncached)"``, ``"error: ..."``).
            Forwarded to dnav; the path is converted to a :class:`Path`
            for consumer convenience (dnav itself emits a string path).
        cancel_check: Optional zero-arg callable polled at the top of
            each file. If truthy, the loop exits early and the partial
            result dict is returned.

    Returns:
        ``{path: status}`` per :func:`datanavigator.precompute_toc`, plus
        an ``"error: missing"`` entry for any explicitly-named path that
        doesn't exist.
    """
    # Adapt the consumer callback (which expects a Path) onto dnav's
    # (idx, total, str, status) signature so dustrack's existing
    # contract is preserved.
    if progress_callback is None:
        dnav_cb = None
    else:
        def dnav_cb(idx, total, path_str, status):
            progress_callback(idx, total, Path(path_str), status)

    return dnav.precompute_toc_folder(
        sources,
        extensions=extensions,
        recursive=recursive,
        force=force,
        show_progress=show_progress,
        progress_callback=dnav_cb,
        cancel_check=cancel_check,
    )


