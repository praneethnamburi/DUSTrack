"""
Sandbox #14 -- Tier 2 (fast_render) DUSTrack UI responsiveness benchmark.

Mirrors probe 09's harness against the same real DUSTrack instance on
the same `interosseous_pn24-x` video, but constructs the DUSTrack
browser with ``fast_render=True`` so the image region and annotation
scatter render Qt-native (QGraphicsView + QPixmapItem +
QGraphicsItemGroup) instead of through matplotlib's full-figure raster
path. The architectural rationale and probe 13's component breakdown
live in BENCHMARKING.md; this probe nails the actual per-frame total.

Acceptance threshold (per the 1.5.0 plan): median ≤ 25 ms / frame,
i.e. ≥ 40 fps and ≥ 3.7x over the 1.4.0-qt + cache_quick_wins
baseline of 93.8 ms. Probe 13's prediction was ~17-22 ms.

Constructs DUSTrack directly rather than through
``DLCProject.annotate()`` so the ``fast_render`` kwarg can be
threaded through without depending on a DUSTrack API change.

Usage:
    C:\\Users\\praneeth\\anaconda3\\envs\\dlc\\python.exe \\
        C:\\dev\\datanavigator\\tests\\qt_learning\\14_benchmark_fast_render.py \\
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
        f"- fast_render: True",
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
    parser.add_argument("--profile", action="store_true",
                        help="Print per-segment timing breakdown")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.stderr.write(f"config not found: {args.config}\n")
        return 2

    from dustrack import DLCProject, DUSTrack
    from dustrack.dlcinterface import VideoFileManager
    import dustrack
    import datanavigator
    dustrack_root = os.path.dirname(os.path.dirname(os.path.abspath(dustrack.__file__)))

    print("=" * 60)
    print("DUSTrack UI frame-update benchmark (fast_render Tier 2)")
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

    # Mirror DLCProject.annotate's annotation-layer discovery so the
    # measured DUSTrack instance has the same load as probe 09 but is
    # constructed directly (so we can pass fast_render=True).
    video_index = args.video_index
    if video_index < 0:
        video_index = len(g.video_list) + video_index
    if g.latest_iteration_is_trained():
        new_iteration_num = g.latest_iteration + 1
    else:
        new_iteration_num = g.latest_iteration
    new_annotation_suffix = f"iteration-{new_iteration_num}"
    fm_annotations = VideoFileManager(g, video_index)
    annotation_names = fm_annotations.get_all_annotation_layers(new_annotation_suffix)
    annotation_names["buffer"] = fm_annotations.get_new_json("buffer")

    print("constructing DUSTrack(fast_render=True) directly...")
    ret = DUSTrack(
        g.video_list[video_index],
        annotation_names,
        height_ratios=(3, 1, 1),
        fast_render=True,
    )
    for ann in ret.annotations:
        if "dlc_" in ann.name:
            ann.set_plot_type("line")
    ann_names = [x.name for x in ret.annotations if "dlc_" in x.name]
    if ann_names:
        ret.statevariables["annotation_overlay"].set_state(ann_names[-1])
    ret.update()

    n_total_frames = len(ret)
    print(f"DUSTrack instance has {n_total_frames} frames")

    # Drain initial paints + asset realization.
    for _ in range(10):
        app.processEvents()

    n_frames = min(args.n_frames, n_total_frames)
    print(f"running {n_frames} update() calls...")

    # Per-segment instrumentation, mirroring probe 11's breakdown so
    # we can attribute the remaining cost when the total exceeds the
    # 25 ms target. Activated when --profile is passed.
    segment_acc = {k: [] for k in (
        "update", "process_events",
    )}

    times_ms = []
    for i in range(n_frames):
        ret._current_idx = i % n_total_frames
        t0 = time.perf_counter()
        ret.update()
        t_update = time.perf_counter()
        app.processEvents()
        t_pe = time.perf_counter()
        if i >= args.n_warmup:
            times_ms.append((t_pe - t0) * 1000)
            segment_acc["update"].append((t_update - t0) * 1000)
            segment_acc["process_events"].append((t_pe - t_update) * 1000)

    if args.profile:
        import statistics as _st
        print()
        print("Per-segment median (ms):")
        for k, vs in segment_acc.items():
            if vs:
                print(f"  {k:20s}  median={_st.median(vs):7.2f}  "
                      f"mean={_st.mean(vs):7.2f}  n={len(vs)}")
        print()

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

    # 1.5.0 acceptance: median <= 25 ms (>= 40 fps).
    THRESHOLD_MS = 25.0
    if stats["median"] > THRESHOLD_MS:
        print(f"WARNING: median {stats['median']:.2f} ms exceeds "
              f"acceptance threshold ({THRESHOLD_MS} ms / 40 fps).")
    else:
        baseline_ms = 93.8
        print(f"PASS: median {stats['median']:.2f} ms, "
              f"speedup {baseline_ms / stats['median']:.2f}x over 1.4.0-qt + cache.")

    if args.record:
        _append_to_benchmarking_md(args.record, stats, args.n_warmup, dustrack_root)

    plt.close(ret.figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
