"""
Sandbox #24 -- DUSTrack cold-open benchmark (pia02-shaped sessions).

Times the path from ``DLCProject(config)`` through
``g.annotate(video_index=...)`` to first painted frame, broken down
into segments. Cold-open feels slow on pia02-scale sessions
(many annotation layers x long videos); this probe identifies the
actual hot path before any optimization work happens (per the
1.2.0 roadmap: "profile-first; optimize the surfaced hot path, not
the assumed one").

What gets timed (wall-clock, perf_counter):

  * DLCProject(config)              -- project meta load
  * VideoFileManager(...)           -- file walk
  * get_all_annotation_layers(...)  -- enumerate JSONs + DLC traces
  * DUSTrack.__init__               -- heavy constructor (sub-segmented)
      - VideoBrowser.__init__       -- PyAV TOC + image pane mount
      - add_annotation_layers       -- per-layer load + setup_display
      - statevariables.show         -- Qt sidebar mount
      - add_events / key bindings / buttons
      - first update() + plt.draw()
  * _normalize_dlc_layer_display
  * _restructure_annotation_order
  * final ret.update()

Plus a cProfile pass over the whole annotate() call, with the top
20 cumulative-time entries reported.

Usage::

    C:\\Users\\praneeth\\anaconda3\\envs\\dlc3rc14\\python.exe \\
        C:\\dev\\DUSTrack\\tests\\qt_learning\\24_benchmark_cold_open.py \\
        [--config CFG] [--video-index N] [--max-layers]

``--max-layers`` overrides ``--video-index`` and picks the video in
the project with the most annotation JSONs on disk -- the
pia02-shaped worst case.
"""

import argparse
import cProfile
import functools
import io
import logging
import os
import pstats
import sys
import time
from pathlib import Path

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


DEFAULT_CONFIG = r"M:\DLC_MODELS\general\interosseous_pn24-x-2025-10-24\config.yaml"


# ---------------------------------------------------------------------------
# Segment-timing helper.
# ---------------------------------------------------------------------------

class _Timer:
    def __init__(self):
        self.segments: list[tuple[str, float]] = []

    def add(self, label: str, dt: float) -> None:
        self.segments.append((label, dt))

    def block(self, label: str):
        timer_self = self

        class _CM:
            def __enter__(self):
                self._t0 = time.perf_counter()
                return self

            def __exit__(self, *exc):
                timer_self.add(label, time.perf_counter() - self._t0)
                return False

        return _CM()

    def report(self) -> None:
        if not self.segments:
            return
        total = sum(dt for _, dt in self.segments)
        max_label = max(len(l) for l, _ in self.segments)
        print("-" * 72)
        print(f"  {'segment':<{max_label}}  {'ms':>10}  {'% of total':>10}")
        print("-" * 72)
        for label, dt in self.segments:
            ms = dt * 1000.0
            pct = (dt / total * 100.0) if total > 0 else 0.0
            print(f"  {label:<{max_label}}  {ms:10.2f}  {pct:9.1f}%")
        print("-" * 72)
        print(f"  {'TOTAL':<{max_label}}  {total*1000.0:10.2f}  {100.0:9.1f}%")
        print("-" * 72)


# ---------------------------------------------------------------------------
# Patching helpers -- wrap entry points to record per-method wall-clock.
# ---------------------------------------------------------------------------

def _wrap(timer: _Timer, label: str, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            timer.add(label, time.perf_counter() - t0)
    return wrapper


def install_segment_timers(timer: _Timer):
    """Patch dustrack + datanavigator entry points to record wall-clock
    on each segment of the annotate() call.
    """
    import dustrack
    from dustrack import gui
    import datanavigator
    from datanavigator import videos as dnav_videos

    # Heavy methods called inside DUSTrack.__init__ (1.2.0rc1 merged
    # the former _DUSTrackBase methods into the DUSTrack class itself).
    gui.DUSTrack.add_annotation_layers = _wrap(
        timer, "  add_annotation_layers",
        gui.DUSTrack.add_annotation_layers,
    )
    gui.DUSTrack.add_events = _wrap(
        timer, "  add_events",
        gui.DUSTrack.add_events,
    )
    gui.DUSTrack.set_key_bindings = _wrap(
        timer, "  set_key_bindings",
        gui.DUSTrack.set_key_bindings,
    )

    # statevariables.show is one of the slow Qt-sidebar mounts.
    from datanavigator.assets import StateVariables
    StateVariables.show = _wrap(
        timer, "  statevariables.show", StateVariables.show,
    )

    # VideoBrowser.__init__ wraps the PyAV TOC build + image pane.
    dnav_videos.VideoBrowser.__init__ = _wrap(
        timer, "  VideoBrowser.__init__",
        dnav_videos.VideoBrowser.__init__,
    )

    # DUSTrack-side: enhance widget + close guard.
    gui.DUSTrack._add_enhance_widget = _wrap(
        timer, "  _add_enhance_widget",
        gui.DUSTrack._add_enhance_widget,
    )
    gui.DUSTrack._install_close_guard = _wrap(
        timer, "  _install_close_guard",
        gui.DUSTrack._install_close_guard,
    )

    # Post-construct helpers.
    gui.DUSTrack._normalize_dlc_layer_display = _wrap(
        timer, "_normalize_dlc_layer_display",
        gui.DUSTrack._normalize_dlc_layer_display,
    )
    gui.DUSTrack._restructure_annotation_order = _wrap(
        timer, "_restructure_annotation_order",
        gui.DUSTrack._restructure_annotation_order,
    )

    # update() = the final paint pump.
    gui.DUSTrack.update = _wrap(
        timer, "ret.update() (final)",
        gui.DUSTrack.update,
    )


# ---------------------------------------------------------------------------
# Layer counting / video selection
# ---------------------------------------------------------------------------

def _count_layers_per_video(g) -> list[tuple[int, int, int]]:
    """Return [(video_index, n_annotation_jsons, n_dlc_traces), ...].

    Walks the DLC project root ONCE (not per-video) and assigns each
    annotation JSON / DLC h5 to its owning video by stem matching.
    The naive ``VideoFileManager(g, i) for i in ...`` would re-walk
    the project root 120x on a 120-video pia02 config, which
    deadlocks on a network drive in practice -- a finding worth
    keeping in mind for the 1.2.0 loading-time work.
    """
    project_root = Path(g.paths['project'])
    json_buckets: dict[str, int] = {}
    h5_buckets: dict[str, int] = {}
    # Single os.walk over the whole project tree.
    for root, _dirs, files in os.walk(project_root):
        for name in files:
            if name.endswith(".json") and "_annotations" in name:
                stem = name.split("_annotations")[0]
                json_buckets[stem] = json_buckets.get(stem, 0) + 1
            elif name.endswith(".h5") and "DLC" in name:
                stem = name.split("DLC")[0]
                h5_buckets[stem] = h5_buckets.get(stem, 0) + 1

    counts: list[tuple[int, int, int]] = []
    for i, video_path in enumerate(g.video_list):
        stem = Path(video_path).stem
        counts.append(
            (i, json_buckets.get(stem, 0), h5_buckets.get(stem, 0))
        )
    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--video-index", type=int, default=0)
    parser.add_argument(
        "--max-layers",
        action="store_true",
        help="Override --video-index; pick the video with the most "
             "annotation JSONs on disk.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="cProfile top-N to display by cumulative time.",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Don't open DUSTrack; just print the layer-count table.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.stderr.write(f"config not found: {args.config}\n")
        return 2

    from dustrack import DLCProject
    import dustrack
    import datanavigator

    print("=" * 72)
    print("DUSTrack cold-open benchmark")
    print(f"  datanavigator: {datanavigator.__file__}")
    print(f"  dustrack     : {dustrack.__file__}")
    print(f"  backend      : {matplotlib.get_backend()}, "
          f"qt_api={os.environ.get('QT_API')}")
    print(f"  config       : {args.config}")
    print("=" * 72)

    # ---- DLCProject construction ----
    timer = _Timer()
    with timer.block("DLCProject(config)"):
        g = DLCProject(args.config)
    print(f"  {len(g.video_list)} videos in project")

    # ---- Video selection ----
    with timer.block("scan video layer counts"):
        layer_counts = _count_layers_per_video(g)
    layer_counts_sorted = sorted(layer_counts, key=lambda x: x[1], reverse=True)
    print()
    print(f"Layer-count distribution across {len(layer_counts)} videos:")
    print(f"  top 10 by # of annotation JSON files:")
    for idx, n_json, n_h5 in layer_counts_sorted[:10]:
        name = Path(g.video_list[idx]).name
        print(f"    [{idx:>3}] {n_json:>3} JSONs, {n_h5:>3} DLC traces  -- {name}")
    if len(layer_counts_sorted) > 10:
        print("  ...")
        print(f"  bottom 3:")
        for idx, n_json, n_h5 in layer_counts_sorted[-3:]:
            name = Path(g.video_list[idx]).name
            print(f"    [{idx:>3}] {n_json:>3} JSONs, {n_h5:>3} DLC traces  -- {name}")
    if args.scan_only:
        return 0

    if args.max_layers:
        if layer_counts_sorted:
            args.video_index = layer_counts_sorted[0][0]
            print(f"\n  --max-layers => video_index={args.video_index}")

    video_fname = g.video_list[args.video_index]
    n_json = next((j for i, j, _ in layer_counts if i == args.video_index), 0)
    n_h5 = next((h for i, _, h in layer_counts if i == args.video_index), 0)
    print()
    print(f"Selected video: index={args.video_index}")
    print(f"  path:    {video_fname}")
    print(f"  layers:  {n_json} JSONs, {n_h5} DLC traces")
    print()

    # ---- Install segment timers ----
    install_segment_timers(timer)

    # ---- Profiled annotate() ----
    print("-" * 72)
    print("Calling g.annotate() under cProfile + segment timers...")
    print("-" * 72)

    profiler = cProfile.Profile()
    profiler.enable()
    with timer.block("g.annotate(...) TOTAL"):
        ret = g.annotate(video_index=args.video_index)
    profiler.disable()

    # ---- Drain initial paint events so first-frame timing is recorded ----
    with timer.block("first paint drain (processEvents x20)"):
        for _ in range(20):
            app.processEvents()

    # ---- Report segments ----
    print()
    print("=" * 72)
    print("Segment wall-clock breakdown")
    print("=" * 72)
    timer.report()

    # ---- Report top cumulative time ----
    print()
    print("=" * 72)
    print(f"cProfile -- top {args.top} cumulative time (excluding builtins)")
    print("=" * 72)
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    stats.print_stats(args.top)
    # Filter out the rate-limit line + headers; print the body.
    print(s.getvalue())

    print("=" * 72)
    print(f"cProfile -- top {args.top} TOTAL time (self-time excluding callees)")
    print("=" * 72)
    s2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=s2).sort_stats("tottime")
    stats2.print_stats(args.top)
    print(s2.getvalue())

    plt.close(ret.figure)
    for _ in range(5):
        app.processEvents()
    return 0


if __name__ == "__main__":
    sys.exit(main())
