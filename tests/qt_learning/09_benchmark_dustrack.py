"""
Sandbox #9 -- real DUSTrack UI responsiveness benchmark.

The synthetic-frame harness in 08_benchmark.py measures the per-frame
update() cost of a GenericBrowser with DUSTrack-shaped widget load,
but the actual plot is one imshow. Real DUSTrack browsers stack the
imshow + multiple annotation traces + 1-D signal plots, so per-frame
cost is higher. This script measures the live DUSTrack instance.

Methodology mirrors 08_benchmark.py: each iteration sets _current_idx,
calls update(), then processEvents() to drain Qt's paint queue. Times
the pair. First N_WARMUP iterations discarded.

Requires:
  - dlc-family env with both dustrack and deeplabcut installed
    (e.g. `dlc` env, with `pip install -e C:/dev/DUSTrack` first)
  - Network access to M:\\DLC_MODELS for the DLCProject config
  - dustrack and datanavigator on matching branches (the
    add_separator() API is required on 1.4.0-qt; absent on master,
    so DUSTrack 1.4.0-qt branch can only run against datanavigator
    1.4.0-qt branch).

Usage:
    C:\\Users\\praneeth\\anaconda3\\envs\\dlc\\python.exe \\
        C:\\dev\\datanavigator\\tests\\qt_learning\\09_benchmark_dustrack.py \\
        [--record LABEL] [--n-frames N] [--n-warmup N]
"""

import argparse
import logging
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime

# Force QtAgg before any pyplot import. Don't use offscreen for DUSTrack:
# the UI is large enough that some realization paths assume a real
# windowing system; offscreen has occasionally bitten me on complex
# matplotlib gridspec setups. Set QT_QPA_PLATFORM externally if needed.
os.environ.setdefault("QT_API", "pyside6")

logging.getLogger("numexpr.utils").setLevel(logging.WARNING)

import matplotlib  # noqa: E402
matplotlib.use("QtAgg")
from matplotlib import pyplot as plt  # noqa: E402

from qtpy.QtCore import QtMsgType, qInstallMessageHandler  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402


def _silence_known_qt_warnings(msg_type, _context, message):
    if msg_type == QtMsgType.QtWarningMsg:
        for needle in (
            "Cannot find font directory",
            "does not support propagateSizeHints",
            "does not support raise",
        ):
            if needle in message:
                return
    sys.stderr.write(message + "\n")


qInstallMessageHandler(_silence_known_qt_warnings)
app = QApplication.instance() or QApplication([])


HERE = os.path.dirname(os.path.abspath(__file__))
DNAV_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_CONFIG = r"M:\DLC_MODELS\general\interosseous_pn24-x-2025-10-24\config.yaml"


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


def _append_to_benchmarking_md(label, stats, n_warmup, dustrack_root):
    md_path = os.path.join(DNAV_ROOT, "BENCHMARKING.md")
    if not os.path.exists(md_path):
        sys.stderr.write(f"--record: {md_path} not found.\n")
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    import datanavigator as _dnav
    import dustrack as _dust
    block = [
        "",
        f"### DUSTrack UI -- {label} -- {ts}",
        "",
        f"- datanavigator: `{_git_describe(DNAV_ROOT)}` source `{_dnav.__file__}`",
        f"- dustrack: `{_git_describe(dustrack_root)}` source `{_dust.__file__}`",
        f"- backend: {matplotlib.get_backend()}, qt_api={os.environ.get('QT_API', '?')}, "
        f"qt_plat={os.environ.get('QT_QPA_PLATFORM', 'default')}",
        f"- N={stats['n']} ({n_warmup} warmup discarded)",
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Path to DLC config.yaml")
    parser.add_argument("--video-index", type=int, default=0,
                        help="Index into DLCProject.video_list")
    parser.add_argument("--n-frames", type=int, default=200,
                        help="Total frames to render (default 200)")
    parser.add_argument("--n-warmup", type=int, default=15,
                        help="Frames discarded as warmup (default 15)")
    parser.add_argument("--record", metavar="LABEL",
                        help="Append result block to BENCHMARKING.md")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.stderr.write(f"config not found: {args.config}\n")
        return 2

    # Import dustrack *after* QtAgg / QApplication are set up, so its
    # matplotlib import / first figure construction goes through the
    # correct backend.
    from dustrack import DLCProject
    import dustrack
    import datanavigator
    dustrack_root = os.path.dirname(os.path.dirname(os.path.abspath(dustrack.__file__)))

    print("=" * 60)
    print("DUSTrack UI frame-update benchmark")
    print(f"  datanavigator: {_git_describe(DNAV_ROOT)}  ({datanavigator.__file__})")
    print(f"  dustrack     : {_git_describe(dustrack_root)}  ({dustrack.__file__})")
    print(f"  backend      : {matplotlib.get_backend()}, qt_api={os.environ.get('QT_API')}")
    print(f"  config       : {args.config}")
    print(f"  video index  : {args.video_index}")
    print(f"  frames       : {args.n_frames} ({args.n_warmup} warmup, "
          f"{args.n_frames - args.n_warmup} measured)")
    print("=" * 60)

    print("opening DLCProject...")
    g = DLCProject(args.config)
    print(f"  {len(g.video_list)} videos in project")
    print(f"  using: {g.video_list[args.video_index]}")

    print("calling annotate() (this opens the UI window)...")
    ret = g.annotate(video_index=args.video_index)
    n_total_frames = len(ret)
    print(f"DUSTrack instance has {n_total_frames} frames")

    # Drain initial paints + asset realization.
    for _ in range(10):
        app.processEvents()

    # If the video has fewer frames than we want to time, loop the index.
    n_frames = min(args.n_frames, n_total_frames)
    print(f"running {n_frames} update() calls...")

    times_ms = []
    for i in range(n_frames):
        ret._current_idx = i % n_total_frames
        t0 = time.perf_counter()
        ret.update()
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
        _append_to_benchmarking_md(args.record, stats, args.n_warmup, dustrack_root)

    plt.close(ret.figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
