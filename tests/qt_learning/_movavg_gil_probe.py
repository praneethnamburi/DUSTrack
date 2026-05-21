"""
GIL-breaking-path probe for ``lk_moving_average_filter`` parallelism.

Stand-alone (not a pytest test); run with the dlc3rc14 env. Measures
three parallel configurations on the dnav example video:

  thread_pool  -- the current production path (ThreadPoolExecutor +
                  cv.setNumThreads(1)).
  process_pool -- ProcessPoolExecutor with frames pickled on each
                  submission. The first plausible GIL-breaking path;
                  cost is per-submission pickle overhead.
  shared_mem   -- ProcessPoolExecutor backed by ``multiprocessing.
                  shared_memory``. All frames pre-decoded into one
                  shared block; workers index into it instead of
                  receiving frame bytes. Avoids pickle bytes for frames.

Each config runs the *same* 286-window movavg over the 300-frame
example video, W=15 (0.5 s), L=2, and skips disk I/O. Output is wall
time + parity vs the thread_pool reference.

Goal of the probe: decide whether the multiprocessing detour can move
the parallel ceiling meaningfully below the GIL-bound 5.75 s.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from multiprocessing import shared_memory
from pathlib import Path

import cv2 as cv
import numpy as np

import dustrack
from datanavigator import VideoReader
from datanavigator.examples import get_example_video
from dustrack.postprocess import (
    compute_sigmoid_weights,
    gray,
    lucas_kanade_rstc_2,
)


WINDOW_SIZE = 0.5
N_LABELS = 2


# ----------------------------------------------------------------------
# Stage the synthetic annotation (mirrors the bench harness exactly so
# numbers are directly comparable).
# ----------------------------------------------------------------------

def _stage_ann():
    vp = get_example_video()
    work = Path(tempfile.mkdtemp(prefix="movavg_probe_"))
    vd = work / Path(vp).name
    shutil.copy(vp, vd)
    vr = VideoReader(str(vd))
    n, h, w = len(vr), *vr[0].asnumpy().shape[:2]
    ann = dustrack.VideoAnnotation(vname=str(vd), n_labels=N_LABELS)
    for li in range(N_LABELS):
        for f in range(n):
            t = f / max(n - 1, 1)
            ann.add(
                location=[100.0 + 0.4 * t * (w - 200) + 30 * li,
                          100.0 + 0.4 * t * (h - 200) + 30 * li],
                label=str(li), frame_number=f,
            )
    ann.save()
    return work, ann


def _build_inputs(ann):
    """Pre-decode every frame to grayscale and gather per-window inputs.

    Returns ``(frames_gray, n_window_frames, windows, sigmoid_forward,
    sigmoid_reverse)`` where ``windows`` is a list of ``(cnt, start_frame,
    end_frame, start_points, end_points)`` tuples.
    """
    video = ann.video
    n_window_frames = round(WINDOW_SIZE * video.get_avg_fps())
    n_frames = ann.n_frames
    frames_gray = np.stack([gray(video[i].asnumpy()) for i in range(n_frames)])
    sigmoid_forward, sigmoid_reverse = compute_sigmoid_weights(n_window_frames)
    label_list = sorted(ann.data.keys())
    windows = []
    for cnt in range(n_frames - n_window_frames + 1):
        start_frame = cnt
        end_frame = cnt + n_window_frames - 1
        start_points = [ann.data[label][start_frame] for label in label_list]
        end_points = [ann.data[label][end_frame] for label in label_list]
        windows.append((cnt, start_frame, end_frame, start_points, end_points))
    return frames_gray, n_window_frames, windows, sigmoid_forward, sigmoid_reverse, label_list


# ----------------------------------------------------------------------
# Worker functions. Must be module-level so ProcessPoolExecutor can
# pickle the reference.
# ----------------------------------------------------------------------

def _worker_pickled(args):
    """Receives frame ARRAYS by pickle. The expensive variant."""
    frames_list, start_points, end_points, sf, sr = args
    return lucas_kanade_rstc_2(frames_list, start_points, end_points,
                                sigmoid_forward=sf, sigmoid_reverse=sr)


# Shared-memory worker state -- attached lazily per worker process.
_SHM_STATE = {"shm": None, "frames": None}


def _worker_shm_init(shm_name, shape, dtype_name):
    """Pool-init hook: attach the worker to the shared-memory block."""
    shm = shared_memory.SharedMemory(name=shm_name)
    frames = np.ndarray(shape, dtype=np.dtype(dtype_name), buffer=shm.buf)
    # Pin so the SharedMemory object survives across calls in this worker.
    _SHM_STATE["shm"] = shm
    _SHM_STATE["frames"] = frames
    # cv inside worker also wants no internal threading.
    cv.setNumThreads(1)


def _worker_shm(args):
    """Receives indices, slices into the shared frames array."""
    start_frame, end_frame, start_points, end_points, sf, sr = args
    frames = _SHM_STATE["frames"]
    # Slice forward window. ascontiguousarray() because LK wants C-contig.
    window = [np.ascontiguousarray(frames[i]) for i in range(start_frame, end_frame + 1)]
    return lucas_kanade_rstc_2(window, start_points, end_points,
                                sigmoid_forward=sf, sigmoid_reverse=sr)


# ----------------------------------------------------------------------
# Three runs.
# ----------------------------------------------------------------------

def _make_avg(n_frames, n_labels):
    return (np.zeros((n_frames, n_labels, 2), dtype=np.float64),
            np.zeros((n_frames, n_labels), dtype=np.int32))


def _accumulate(sum_paths, count_paths, start_frame, rstc_path):
    n = rstc_path.shape[0]
    sum_paths[start_frame:start_frame + n] += rstc_path
    count_paths[start_frame:start_frame + n] += 1


def run_thread_pool(frames_gray, n_window_frames, windows, sf, sr, n_frames, n_labels):
    """Reference path: matches the production loop (sans .pkl, sans tqdm)."""
    sum_paths, count_paths = _make_avg(n_frames, n_labels)
    saved = cv.getNumThreads()
    cv.setNumThreads(1)
    try:
        with ThreadPoolExecutor() as executor:
            max_inflight = max(2 * (executor._max_workers or 4), 8)
            inflight = {}
            it = iter(windows)
            n_done = 0
            while n_done < len(windows):
                while len(inflight) < max_inflight:
                    try:
                        cnt, sf_idx, ef_idx, sp, ep = next(it)
                    except StopIteration:
                        break
                    window = [frames_gray[i] for i in range(sf_idx, ef_idx + 1)]
                    fut = executor.submit(lucas_kanade_rstc_2, window, sp, ep,
                                           sigmoid_forward=sf, sigmoid_reverse=sr)
                    inflight[fut] = (cnt, sf_idx)
                if not inflight:
                    break
                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for fut in done:
                    cnt, sf_idx = inflight.pop(fut)
                    rstc_path = fut.result()
                    _accumulate(sum_paths, count_paths, sf_idx, rstc_path)
                    n_done += 1
    finally:
        cv.setNumThreads(saved)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sum_paths / count_paths[..., np.newaxis]


def run_process_pool_pickled(frames_gray, n_window_frames, windows, sf, sr, n_frames, n_labels):
    """ProcessPoolExecutor with frames pickled per submission. Baseline IPC cost."""
    sum_paths, count_paths = _make_avg(n_frames, n_labels)
    with ProcessPoolExecutor() as executor:
        max_inflight = max(2 * (executor._max_workers or 4), 8)
        inflight = {}
        it = iter(windows)
        n_done = 0
        while n_done < len(windows):
            while len(inflight) < max_inflight:
                try:
                    cnt, sf_idx, ef_idx, sp, ep = next(it)
                except StopIteration:
                    break
                window = [frames_gray[i] for i in range(sf_idx, ef_idx + 1)]
                fut = executor.submit(_worker_pickled,
                                       (window, sp, ep, sf, sr))
                inflight[fut] = (cnt, sf_idx)
            if not inflight:
                break
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:
                cnt, sf_idx = inflight.pop(fut)
                rstc_path = fut.result()
                _accumulate(sum_paths, count_paths, sf_idx, rstc_path)
                n_done += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return sum_paths / count_paths[..., np.newaxis]


def run_process_pool_shm(frames_gray, n_window_frames, windows, sf, sr, n_frames, n_labels):
    """ProcessPoolExecutor + shared-memory frames. Frames pickled ONCE."""
    sum_paths, count_paths = _make_avg(n_frames, n_labels)
    shm = shared_memory.SharedMemory(create=True, size=frames_gray.nbytes)
    shm_arr = np.ndarray(frames_gray.shape, dtype=frames_gray.dtype, buffer=shm.buf)
    shm_arr[:] = frames_gray
    try:
        with ProcessPoolExecutor(
            initializer=_worker_shm_init,
            initargs=(shm.name, frames_gray.shape, frames_gray.dtype.name),
        ) as executor:
            max_inflight = max(2 * (executor._max_workers or 4), 8)
            inflight = {}
            it = iter(windows)
            n_done = 0
            while n_done < len(windows):
                while len(inflight) < max_inflight:
                    try:
                        cnt, sf_idx, ef_idx, sp, ep = next(it)
                    except StopIteration:
                        break
                    fut = executor.submit(_worker_shm,
                                           (sf_idx, ef_idx, sp, ep, sf, sr))
                    inflight[fut] = (cnt, sf_idx)
                if not inflight:
                    break
                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for fut in done:
                    cnt, sf_idx = inflight.pop(fut)
                    rstc_path = fut.result()
                    _accumulate(sum_paths, count_paths, sf_idx, rstc_path)
                    n_done += 1
    finally:
        shm.close()
        shm.unlink()
    with np.errstate(invalid="ignore", divide="ignore"):
        return sum_paths / count_paths[..., np.newaxis]


def _time(label, fn, *args, n_reps=3):
    ts = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        out = fn(*args)
        ts.append(time.perf_counter() - t0)
    return out, sorted(ts)


def main():
    work, ann = _stage_ann()
    try:
        frames_gray, n_window_frames, windows, sf, sr, label_list = _build_inputs(ann)
        n_frames = ann.n_frames
        n_labels = len(label_list)

        print(f"video: {ann.video.fname}")
        print(f"  frames={n_frames}  shape={frames_gray.shape[1:]}  win={n_window_frames}")
        print(f"  shared-mem size: {frames_gray.nbytes / 1e6:.1f} MB")
        print(f"  windows={len(windows)}, labels={n_labels}")
        print()

        # warmup each path once (untimed)
        run_thread_pool(frames_gray, n_window_frames, windows, sf, sr, n_frames, n_labels)
        ref, ts_thr = _time("thread_pool", run_thread_pool,
                            frames_gray, n_window_frames, windows, sf, sr, n_frames, n_labels)
        out_p, ts_pp = _time("process_pool_pickled", run_process_pool_pickled,
                              frames_gray, n_window_frames, windows, sf, sr, n_frames, n_labels)
        out_s, ts_shm = _time("process_pool_shm", run_process_pool_shm,
                                frames_gray, n_window_frames, windows, sf, sr, n_frames, n_labels)

        print(f"{'path':<24} {'min (s)':>10} {'median':>10} {'max':>10}    reps")
        for label, ts in [
            ("thread_pool", ts_thr),
            ("process_pool_pickled", ts_pp),
            ("process_pool_shm", ts_shm),
        ]:
            med = ts[len(ts) // 2]
            reps = " ".join(f"{t:.2f}" for t in ts)
            print(f"{label:<24} {ts[0]:>10.2f} {med:>10.2f} {ts[-1]:>10.2f}    {reps}")

        # Parity (vs thread_pool ref).
        d_p = float(np.nanmax(np.where(np.isnan(ref) & np.isnan(out_p), 0, np.abs(ref - out_p))))
        d_s = float(np.nanmax(np.where(np.isnan(ref) & np.isnan(out_s), 0, np.abs(ref - out_s))))
        print(f"\nparity vs thread_pool: pickled max|delta|={d_p:.2e}  shm max|delta|={d_s:.2e}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
