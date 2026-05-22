"""Sandbox #28 -- multi-video DUSTrack per-frame update() benchmark.

Mirrors the probe-09 / probe-14 methodology (continuous frame walk,
warmup discarded, ``app.processEvents()`` inside the timing) but
exercises the 1.2.0a3 multi-video session shape -- the figure now
hosts artists for every bundle's annotations (~220 Line2D on the
trace axes for a 12-video pia02 session), not just the active one,
so the per-update cost is the relevant production number after
multi-video lands.

What's measured:

- Steady-state per-frame ``update()`` cost on the ACTIVE bundle
  (sequential ``_current_idx = i % n_frames``), AFTER background
  hydration has completed and the canvas has been warmed (so the
  counter-gated force-paint in ``DUSTrack.update`` has already
  fired its one-shot for this bundle).
- N=200, n_warmup=15, dropping the warmup window.

Compared to probe 14 (single-video, same project's video 0):
  1.5.0 fast_render Tier 2 baseline: 36 ms / 28 fps.

Run via::

    python tests/qt_learning/28_benchmark_multi_video_update.py [--record LABEL]
"""
from __future__ import annotations

import argparse
import logging
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")
logging.getLogger("numexpr.utils").setLevel(logging.WARNING)

import matplotlib  # noqa: E402
matplotlib.use("QtAgg")
from matplotlib import pyplot as plt  # noqa: E402

from qtpy.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROJECT = Path(
    "M:/DLC_MODELS/participant_models_general/s006/RFA/"
    "interosseous_pn24-x-2025-10-24"
)


def _git_describe(repo_root: str) -> str:
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root, stderr=subprocess.DEVNULL, text=True,
        ).strip()
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root, stderr=subprocess.DEVNULL, text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root, stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return f"{branch}@{sha}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def _compute_stats(times_ms):
    s = sorted(times_ms)
    n = len(s)
    return {
        "n": n,
        "min": min(times_ms),
        "median": statistics.median(times_ms),
        "mean": statistics.mean(times_ms),
        "p95": s[int(n * 0.95)],
        "p99": s[int(n * 0.99)] if n > 100 else s[-1],
        "max": max(times_ms),
    }


def _wait_for_hydration(tracker, timeout=300.0):
    from qtpy.QtCore import QCoreApplication
    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(b.is_terminal for b in tracker._bundles):
            return
        QCoreApplication.processEvents()
        time.sleep(0.05)
    raise TimeoutError("hydration timed out")


def _append_to_benchmarking_md(label, stats, n_warmup,
                                dustrack_root, project_root,
                                n_bundles, n_lines):
    md_path = os.path.join(dustrack_root, "BENCHMARKING.md")
    if not os.path.exists(md_path):
        sys.stderr.write(f"--record: {md_path} not found.\n")
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    import datanavigator as _dnav
    import dustrack as _dust
    dnav_root = os.path.dirname(os.path.dirname(os.path.abspath(_dnav.__file__)))
    block = [
        "",
        f"### DUSTrack multi-video UI -- {label} -- {ts}",
        "",
        f"- datanavigator: `{_git_describe(dnav_root)}` source `{_dnav.__file__}`",
        f"- dustrack: `{_git_describe(dustrack_root)}` source `{_dust.__file__}`",
        f"- backend: {matplotlib.get_backend()}, qt_api={os.environ.get('QT_API', '?')}",
        f"- project: {project_root} ({n_bundles} bundles)",
        f"- trace axes: {n_lines} Line2D on `_ax_trace_x`",
        f"- N={stats['n']} ({n_warmup} warmup discarded), continuous frame walk",
        "",
        "| min | median | mean | p95 | p99 | max | fps (median) |",
        "|---|---|---|---|---|---|---|",
        f"| {stats['min']:.2f} | {stats['median']:.2f} | {stats['mean']:.2f} "
        f"| {stats['p95']:.2f} | {stats['p99']:.2f} | {stats['max']:.2f} "
        f"| {1000.0 / stats['median']:.1f} |",
    ]
    with open(md_path, "a", encoding="utf-8") as f:
        f.write("\n".join(block) + "\n")
    print(f"appended block to {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(DEFAULT_PROJECT),
                        help="DLC project root for the multi-video session")
    parser.add_argument("--n-frames", type=int, default=200,
                        help="Total frames to render (default 200)")
    parser.add_argument("--n-warmup", type=int, default=15,
                        help="Frames discarded as warmup (default 15)")
    parser.add_argument("--record", metavar="LABEL",
                        help="Append result block to BENCHMARKING.md")
    args = parser.parse_args()

    project_root = Path(args.project)
    if not project_root.exists():
        sys.stderr.write(f"project not found: {project_root}\n")
        return 2

    import dustrack
    import datanavigator
    dustrack_root = os.path.dirname(os.path.dirname(os.path.abspath(dustrack.__file__)))
    dnav_root = os.path.dirname(os.path.dirname(os.path.abspath(datanavigator.__file__)))

    print("=" * 60)
    print("DUSTrack MULTI-VIDEO UI frame-update benchmark")
    print(f"  datanavigator: {_git_describe(dnav_root)}  ({datanavigator.__file__})")
    print(f"  dustrack     : {_git_describe(dustrack_root)}  ({dustrack.__file__})")
    print(f"  backend      : {matplotlib.get_backend()}, qt_api={os.environ.get('QT_API')}")
    print(f"  project      : {project_root}")
    print(f"  frames       : {args.n_frames} ({args.n_warmup} warmup, "
          f"{args.n_frames - args.n_warmup} measured)")
    print("=" * 60)

    print("opening multi-video session...")
    t0 = time.time()
    tracker = dustrack.open(project_root)
    print(f"  active bundle constructed in {time.time() - t0:.2f}s")
    print(f"  {len(tracker._bundles)} bundles total")

    print("waiting for background hydration...")
    t0 = time.time()
    _wait_for_hydration(tracker)
    print(f"  all bundles hydrated in {time.time() - t0:.2f}s")

    # Drain any pending paint events after hydration completes so the
    # measured loop starts from a clean canvas state.
    for _ in range(20):
        app.processEvents()

    n_bundles = len(tracker._bundles)
    n_lines_x = len(tracker._ax_trace_x.lines)
    print(f"  trace_x lines: {n_lines_x} (across {n_bundles} bundles)")

    n_total_frames = len(tracker.data)
    n_frames = min(args.n_frames, n_total_frames)
    print(f"running {n_frames} update() calls on active bundle...")

    times_ms = []
    for i in range(n_frames):
        tracker._current_idx = i % n_total_frames
        t0 = time.perf_counter()
        tracker.update()
        app.processEvents()
        t1 = time.perf_counter()
        if i >= args.n_warmup:
            times_ms.append((t1 - t0) * 1000)

    stats = _compute_stats(times_ms)
    print(f"N measured  = {stats['n']}")
    print(f"min         = {stats['min']:7.2f} ms")
    print(f"median      = {stats['median']:7.2f} ms")
    print(f"mean        = {stats['mean']:7.2f} ms")
    print(f"p95         = {stats['p95']:7.2f} ms")
    print(f"p99         = {stats['p99']:7.2f} ms")
    print(f"max         = {stats['max']:7.2f} ms")
    print(f"fps (median) = {1000.0 / stats['median']:6.1f}")
    print(f"fps (p95)    = {1000.0 / stats['p95']:6.1f}")
    print("=" * 60)

    if args.record:
        _append_to_benchmarking_md(
            args.record, stats, args.n_warmup,
            dustrack_root, project_root, n_bundles, n_lines_x,
        )

    plt.close(tracker.figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
