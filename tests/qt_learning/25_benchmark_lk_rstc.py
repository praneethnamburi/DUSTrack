"""
Sandbox #25 -- LK / LK-RSTC / lk_moving_average_filter benchmark.

Times the three LK call sites and the reduce-jitter pipeline against
the dnav example video. Mode-minor interleaving (one rep of each bench
in turn before moving to rep n+1) per memo
feedback_thermal_confounding_benchmark_iteration.

Benchmarks
----------
* lk2_pair       -- ``lucas_kanade_2`` over a 16-frame pre-decoded grayscale
                    slice, controlled-input (no I/O). Isolates per-pair LK.
* lk_full        -- ``lucas_kanade(video, 35, 50, ..., mode='full')`` forward.
                    Includes decode + grayscale per pair.
* lk_rstc_full   -- ``lucas_kanade_rstc(video, 35, 50, start, end)``.
                    Currently decodes each frame twice.
* movavg         -- ``lk_moving_average_filter`` over the full 300-frame
                    example video at window=0.5 s, 2 labels covering all
                    frames, ``use_parallel=False`` for measurement stability.

Each step writes:
  artifacts/<step>/timings.csv  -- per-rep, per-bench wall-clock (s)
  artifacts/<step>/refs.npz     -- reference output arrays for parity diffs

With ``--compare-to <step>`` we load that step's refs.npz and assert
parity vs the current run (np.allclose with ``--atol`` / ``--rtol``).
Speedup table prints alongside the timing summary.

Usage
-----
::

    py = r"C:\\Users\\praneeth\\anaconda3\\envs\\dlc3rc14\\python.exe"
    benchmark = r"C:\\dev\\dustrack\\tests\\qt_learning\\25_benchmark_lk_rstc.py"

    # baseline (no comparison yet)
    %py %benchmark --step baseline

    # after every refactor / optimization step
    %py %benchmark --step refactor --compare-to baseline
    %py %benchmark --step pyramid  --compare-to baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from datanavigator import VideoReader
from datanavigator.examples import get_example_video

import dustrack
from dustrack.lk_opticalflow import lucas_kanade, lucas_kanade_rstc
from dustrack.lk_filter import lucas_kanade_2, lk_moving_average_filter


ARTIFACT_ROOT = Path(tempfile.gettempdir()) / "dustrack_lk_bench"

# Frame slice + points used both for parity references and the per-video benches.
# Mirrors tests/test_opticalflow.py::test_lucas_kanade_rstc so the same arrays
# exercised there are the parity reference here.
START_FRAME = 35
END_FRAME = 50  # inclusive -> 16 frames
START_POINTS = np.array([[153.81, 195.34], [231.90, 209.27]], dtype=np.float32)
END_POINTS = np.array([[166.24, 166.74], [246.63, 181.54]], dtype=np.float32)

# Moving-average bench parameters. Two synthetic labels with linear motion
# across all 300 frames so the filter has trackable data in every window.
MOVAVG_WINDOW_SIZE = 0.5  # seconds
MOVAVG_N_LABELS = 2
MOVAVG_SAVE_RAW = True  # flipped by --no-movavg-save-raw


def _gray(frame_rgb: np.ndarray) -> np.ndarray:
    return cv.cvtColor(frame_rgb, cv.COLOR_RGB2GRAY)


def _prepare_frame_slice(video: VideoReader) -> list[np.ndarray]:
    """Pre-decode + grayscale frames [START_FRAME, END_FRAME] inclusive."""
    return [_gray(video[i].asnumpy()) for i in range(START_FRAME, END_FRAME + 1)]


def _prepare_movavg_workdir(video_path: str) -> tuple[str, str]:
    """Stage a fresh annotation JSON next to a copy of the video.

    Returns (ann_fname, video_fname). Creates synthetic linear-motion
    annotations for ``MOVAVG_N_LABELS`` labels covering all video frames.
    Fresh dir per call so the filter's pkl/json cache doesn't shortcut.
    """
    work = Path(tempfile.mkdtemp(prefix="movavg_"))
    video_dst = work / Path(video_path).name
    shutil.copy(video_path, video_dst)

    vr = VideoReader(str(video_dst))
    n_frames = len(vr)
    h, w = vr[0].asnumpy().shape[:2]

    ann = dustrack.VideoAnnotation(vname=str(video_dst), n_labels=MOVAVG_N_LABELS)
    for label_idx in range(MOVAVG_N_LABELS):
        for frame in range(n_frames):
            t = frame / max(n_frames - 1, 1)
            x = 100.0 + 0.4 * t * (w - 200.0) + 30.0 * label_idx
            y = 100.0 + 0.4 * t * (h - 200.0) + 30.0 * label_idx
            ann.add(location=[x, y], label=str(label_idx), frame_number=frame)
    ann.save()
    return str(ann.fname), str(video_dst)


def _run_movavg_once(video_path: str) -> np.ndarray:
    """Stage + run lk_moving_average_filter; return its data as an array.

    Also asserts the .pkl sidecar contract: present when MOVAVG_SAVE_RAW
    is True, absent when False. Surfaces regressions in the cache
    short-circuit logic.
    """
    ann_fname, video_dst = _prepare_movavg_workdir(video_path)
    ann = dustrack.VideoAnnotation(fname=ann_fname, vname=video_dst)
    result = lk_moving_average_filter(
        ann,
        window_size=MOVAVG_WINDOW_SIZE,
        use_parallel=False,
        save_raw=MOVAVG_SAVE_RAW,
    )
    pkl_path = Path(ann_fname).parent / f"{Path(ann_fname).stem}_lkmovavg_{MOVAVG_WINDOW_SIZE:.3f}.pkl"
    if MOVAVG_SAVE_RAW:
        assert pkl_path.exists(), f"save_raw=True expected .pkl at {pkl_path}"
    else:
        assert not pkl_path.exists(), f"save_raw=False expected NO .pkl at {pkl_path}"
    # Pack into (n_labels, n_frames, 2). NaN where the filter couldn't fit
    # (the trailing window-1 frames). Order labels deterministically.
    labels = sorted(result.data.keys(), key=lambda s: (len(s), s))
    n_frames = result.n_frames
    arr = np.full((len(labels), n_frames, 2), np.nan, dtype=np.float64)
    for li, lbl in enumerate(labels):
        for f, xy in result.data[lbl].items():
            arr[li, f] = np.asarray(xy)
    # Workdir cleanup -- we keep nothing.
    shutil.rmtree(Path(ann_fname).parent, ignore_errors=True)
    return arr


def _bench_lk2_pair(state) -> float:
    t0 = time.perf_counter()
    out = lucas_kanade_2(state["frame_slice"], START_POINTS)
    t1 = time.perf_counter()
    state["ref_lk2_pair"] = np.asarray(out)
    return t1 - t0


def _bench_lk_full(state) -> float:
    t0 = time.perf_counter()
    out = lucas_kanade(
        state["video"], START_FRAME, END_FRAME, START_POINTS, mode="full"
    )
    t1 = time.perf_counter()
    state["ref_lk_full"] = np.asarray(out)
    return t1 - t0


def _bench_lk_rstc_full(state) -> float:
    t0 = time.perf_counter()
    out = lucas_kanade_rstc(
        state["video"], START_FRAME, END_FRAME, START_POINTS, END_POINTS
    )
    t1 = time.perf_counter()
    state["ref_lk_rstc_full"] = np.asarray(out)
    return t1 - t0


def _bench_movavg(state) -> float:
    t0 = time.perf_counter()
    out = _run_movavg_once(state["video_path"])
    t1 = time.perf_counter()
    state["ref_movavg"] = np.asarray(out)
    return t1 - t0


BENCH_TABLE = [
    ("lk2_pair", _bench_lk2_pair),
    ("lk_full", _bench_lk_full),
    ("lk_rstc_full", _bench_lk_rstc_full),
    ("movavg", _bench_movavg),
]


def _normalize_video_state(state) -> None:
    """Pin the VideoReader demuxer to frame 0 before each bench.

    Without this, lk_full's measured cost depends on where the
    previous bench left the demuxer -- a reverse-seek to ``start_frame``
    runs ~60 ms slower than a forward-seek. The baseline run showed
    the artifact: lk_rstc_full's reverse pass happened to leave the
    demuxer right at lk_full's start_frame, so lk_full got
    artificially fast. Read frame 0 (untimed) so every bench starts
    from the same demuxer state.
    """
    state["video"][0].asnumpy()


def _interleaved_bench(state, n_reps: int, n_warmup: int) -> dict[str, list[float]]:
    """Mode-minor: rep r runs benches A, B, C, ... in turn before rep r+1.

    Demuxer state is normalized between benches so per-bench timings
    don't depend on what the previous bench left the demuxer doing.
    """
    timings: dict[str, list[float]] = {name: [] for name, _ in BENCH_TABLE}
    for rep in range(-n_warmup, n_reps):
        for name, fn in BENCH_TABLE:
            _normalize_video_state(state)
            dt = fn(state)
            if rep >= 0:
                timings[name].append(dt)
    return timings


def _summarise(timings: dict[str, list[float]]) -> dict[str, dict]:
    summary = {}
    for name, ts in timings.items():
        ts_sorted = sorted(ts)
        summary[name] = {
            "median_s": statistics.median(ts_sorted),
            "min_s": min(ts_sorted),
            "max_s": max(ts_sorted),
            "n": len(ts_sorted),
        }
    return summary


def _write_artifacts(step: str, timings: dict[str, list[float]], refs: dict) -> Path:
    artdir = ARTIFACT_ROOT / step
    artdir.mkdir(parents=True, exist_ok=True)

    # Per-rep timings CSV.
    with (artdir / "timings.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bench", "rep", "wall_s"])
        for name, ts in timings.items():
            for i, t in enumerate(ts):
                w.writerow([name, i, f"{t:.6f}"])

    # Reference arrays.
    np.savez(artdir / "refs.npz", **refs)

    # Summary JSON.
    (artdir / "summary.json").write_text(json.dumps(_summarise(timings), indent=2))
    return artdir


def _print_summary(step: str, timings: dict[str, list[float]], compare_to: str | None) -> None:
    summary = _summarise(timings)
    print(f"\n== step={step} ==")
    print(f"{'bench':<16} {'median (ms)':>12} {'min (ms)':>10} {'max (ms)':>10} {'n':>4}")
    for name, s in summary.items():
        print(
            f"{name:<16} "
            f"{1000*s['median_s']:>12.3f} "
            f"{1000*s['min_s']:>10.3f} "
            f"{1000*s['max_s']:>10.3f} "
            f"{s['n']:>4}"
        )

    if compare_to:
        prev_path = ARTIFACT_ROOT / compare_to / "summary.json"
        if not prev_path.exists():
            print(f"  (no baseline at {prev_path}; skipping speedup table)")
            return
        prev = json.loads(prev_path.read_text())
        print(f"\n  speedup vs {compare_to}:")
        for name, s in summary.items():
            if name not in prev:
                continue
            base = prev[name]["median_s"]
            now = s["median_s"]
            if now > 0:
                print(f"    {name:<16}  {base / now:>5.2f}x  ({1000*base:.2f} -> {1000*now:.2f} ms)")


def _parity_check(step: str, refs: dict, compare_to: str, atol: float, rtol: float) -> bool:
    """Return True iff all reference arrays match within tolerance. Prints diffs."""
    prev_npz = ARTIFACT_ROOT / compare_to / "refs.npz"
    if not prev_npz.exists():
        print(f"  (no parity reference at {prev_npz}; skipping parity check)")
        return True

    prev = np.load(prev_npz)
    ok = True
    print(f"\n  parity vs {compare_to} (atol={atol:.2e}, rtol={rtol:.2e}):")
    for name in refs:
        if name not in prev.files:
            continue
        a = np.asarray(refs[name])
        b = np.asarray(prev[name])
        if a.shape != b.shape:
            print(f"    {name:<24}  SHAPE MISMATCH {a.shape} vs {b.shape}")
            ok = False
            continue
        # Treat NaN-aligned positions as equal (movavg has trailing NaN tail).
        nan_mask = np.isnan(a) & np.isnan(b)
        diff = np.where(nan_mask, 0.0, np.abs(a - b))
        mismatched_nan = np.isnan(a) ^ np.isnan(b)
        max_abs = float(np.nanmax(diff)) if diff.size else 0.0
        if mismatched_nan.any():
            n_mis = int(mismatched_nan.sum())
            print(f"    {name:<24}  NAN MISMATCH at {n_mis} positions")
            ok = False
            continue
        passed = np.allclose(
            np.where(nan_mask, 0.0, a),
            np.where(nan_mask, 0.0, b),
            atol=atol,
            rtol=rtol,
            equal_nan=False,
        )
        status = "ok" if passed else "FAIL"
        print(f"    {name:<24}  {status}  max|delta|={max_abs:.3e}")
        ok = ok and passed
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, help="label for this run's artifacts dir")
    ap.add_argument("--compare-to", default=None, help="prior step to diff vs")
    ap.add_argument("--n-reps", type=int, default=5)
    ap.add_argument("--n-warmup", type=int, default=1)
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("--skip-movavg", action="store_true", help="omit the slow end-to-end bench")
    ap.add_argument("--no-movavg-save-raw", action="store_true",
                    help="run lk_moving_average_filter with save_raw=False (streaming sum+count, no .pkl)")
    args = ap.parse_args()

    if args.skip_movavg:
        global BENCH_TABLE
        BENCH_TABLE = [(n, f) for (n, f) in BENCH_TABLE if n != "movavg"]
    if args.no_movavg_save_raw:
        global MOVAVG_SAVE_RAW
        MOVAVG_SAVE_RAW = False
        print("[bench] movavg save_raw=False -- skipping .pkl sidecar")

    print(f"dustrack {dustrack.__version__}  python {sys.version.split()[0]}")
    video_path = get_example_video()
    print(f"video: {video_path}")
    video = VideoReader(video_path)
    print(f"  frames={len(video)}  shape={video[0].asnumpy().shape}  fps={video.get_avg_fps():.2f}")

    state = {
        "video": video,
        "video_path": video_path,
        "frame_slice": _prepare_frame_slice(video),
    }

    timings = _interleaved_bench(state, n_reps=args.n_reps, n_warmup=args.n_warmup)

    refs = {k.replace("ref_", ""): v for k, v in state.items() if k.startswith("ref_")}
    artdir = _write_artifacts(args.step, timings, refs)
    print(f"\nartifacts: {artdir}")

    _print_summary(args.step, timings, args.compare_to)

    if args.compare_to:
        ok = _parity_check(args.step, refs, args.compare_to, args.atol, args.rtol)
        if not ok:
            print("\nPARITY FAILED. Inspect refs.npz before proceeding.")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
