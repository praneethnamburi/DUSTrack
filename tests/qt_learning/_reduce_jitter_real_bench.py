"""
Real-video reduce_jitter bench. Times ``lk_moving_average_filter`` on
the pia02_s001_011_RFA2_min1_15s production fixture
(1111 frames, 706x558, ~74 fps, 1 label, ~1075 windows at the default
0.5 s window) using the DLC inference h5 as the input -- matches the
GUI ``Reduce jitter`` button's input shape exactly
(``process_with_lk`` -> ``lk_moving_average_filter`` with
``use_parallel=True``, ``save_raw=True``).

Designed to be run against multiple code versions via
``git checkout`` -- the script itself is forward-compatible: it
detects whether ``save_raw`` is accepted by the installed
``lk_moving_average_filter`` and omits it for older versions.

Stages the input into a temp dir on each rep so the
``_lkmovavg_*.pkl`` / ``.json`` caches never short-circuit the call.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import io
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


VIDEO_SRC = Path(r"S:\_corpus\dustrack\pia02_s001_011_RFA2_min1_15s.mp4")
DLC_H5_SRC = Path(
    r"S:\_corpus\dustrack\pia02_s001_011_RFA2_min1_15s_iteration-0-x-2026-05-20"
    r"\videos\iteration-1\pia02_s001_011_RFA2_min1_15sDLC_Resnet50_pia02_s001_011_RFA2_min1_15s_iteration-0May20shuffle1_snapshot_best-100.h5"
)
WINDOW_SIZE = 0.5  # GUI default


def _stage(work_root: Path) -> tuple[Path, Path]:
    """Copy video + h5 into a clean dir; return (video, h5) paths."""
    work = Path(tempfile.mkdtemp(prefix="reduce_jitter_", dir=str(work_root)))
    vd = work / VIDEO_SRC.name
    shutil.copy(VIDEO_SRC, vd)
    # The TOC sidecar speeds up the open; ship if present.
    toc = VIDEO_SRC.with_suffix(VIDEO_SRC.suffix + ".dnav-toc")
    if toc.exists():
        shutil.copy(toc, work / toc.name)
    # The h5 needs to live next to the video so the canonical layer
    # naming resolves predictably.
    h5 = work / DLC_H5_SRC.name
    shutil.copy(DLC_H5_SRC, h5)
    return vd, h5


def _build_ann(video: Path, h5: Path):
    """Load the DLC h5 as a VideoAnnotation. Reused across reps -- the
    same ``ann`` object is passed in; ``lk_moving_average_filter``
    derives its cache filenames from ``ann.fname``'s stem so each rep
    must use a fresh stage to avoid cache hits."""
    import dustrack
    return dustrack.VideoAnnotation(fname=str(h5), vname=str(video))


def _call_filter(ann, save_raw=True):
    """Wrap the call so ``save_raw`` is only passed when supported."""
    from dustrack.postprocess import lk_moving_average_filter
    sig = inspect.signature(lk_moving_average_filter)
    kwargs = {"window_size": WINDOW_SIZE, "use_parallel": True}
    if "save_raw" in sig.parameters:
        kwargs["save_raw"] = save_raw
    return lk_moving_average_filter(ann, **kwargs)


def _one_rep(work_root: Path, save_raw: bool, quiet: bool) -> float:
    video, h5 = _stage(work_root)
    try:
        import dustrack  # noqa: F401  (import inside so reps reuse cached modules)
        ann = _build_ann(video, h5)
        # Touch the video once to land the demuxer in a known state,
        # untimed (matches `25_benchmark_lk_rstc.py` normalisation).
        ann.video[0].asnumpy()
        t0 = time.perf_counter()
        if quiet:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                _call_filter(ann, save_raw=save_raw)
        else:
            _call_filter(ann, save_raw=save_raw)
        return time.perf_counter() - t0
    finally:
        shutil.rmtree(video.parent, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="tag for this run, printed with results")
    ap.add_argument("--n-reps", type=int, default=3)
    ap.add_argument("--save-raw", choices=("true", "false", "both"), default="true",
                    help="match GUI default (True), or test the False fast-path, or both")
    ap.add_argument("--noisy", action="store_true", help="show tqdm bars + stdout")
    args = ap.parse_args()

    work_root = Path(tempfile.mkdtemp(prefix="reduce_jitter_root_"))
    try:
        import dustrack
        from datanavigator import VideoReader
        vr = VideoReader(str(VIDEO_SRC))
        n_frames = len(vr)
        shape = vr[0].asnumpy().shape
        fps = vr.get_avg_fps()
        n_window_frames = round(WINDOW_SIZE * fps)
        n_windows = n_frames - n_window_frames + 1
        print(f"dustrack {dustrack.__version__}  python {sys.version.split()[0]}")
        print(f"video: {VIDEO_SRC.name}  frames={n_frames}  shape={shape}  fps={fps:.2f}")
        print(f"window: {WINDOW_SIZE} s = {n_window_frames} frames  -> {n_windows} windows")

        modes = ["true", "false"] if args.save_raw == "both" else [args.save_raw]
        for mode in modes:
            save_raw = mode == "true"
            ts = []
            for rep in range(args.n_reps):
                dt = _one_rep(work_root, save_raw=save_raw, quiet=not args.noisy)
                ts.append(dt)
                print(f"  [{args.label}] save_raw={save_raw}  rep {rep}: {dt:.2f} s")
            ts_sorted = sorted(ts)
            med = ts_sorted[len(ts_sorted) // 2]
            print(f"[{args.label}] save_raw={save_raw}  median={med:.2f} s  min={min(ts):.2f}  max={max(ts):.2f}")
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
