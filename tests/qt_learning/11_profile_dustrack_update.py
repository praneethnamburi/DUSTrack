"""
Profile probe: attribute the ~128.6 ms / frame real-DUSTrack budget
across the segments of VideoPointAnnotator.update().

Goal: before committing to a blit-mode refactor (TODO.md "Further perf
wins, item 1"), confirm that the canvas-raster step (the part blit
would replace) is actually a large fraction of per-frame cost. If
processEvents() alone is small, the wins live elsewhere (PyAV decode,
trace updates) and blit isn't the next move.

Methodology
-----------
Mirrors `09_benchmark_dustrack.py`: opens DLCProject + DUSTrack UI on
the interosseous_pn24-x video, drives `_current_idx` through N frames,
times each. Replaces `ret.update` with a hand-instrumented version
that runs the same sequence of calls as VideoPointAnnotator.update +
VideoBrowser.update but with time.perf_counter() spans around each
named segment. `processEvents()` is timed separately in the outer
loop (it's where the actual canvas rasterization happens — what
blit-mode would short-circuit).

Segments (in call order):
  1. annotation_visibility  -- per-annotation scatter offsets + trace ydata
  2. statevars_display      -- Qt overlay text push (cheap on 1.4.0-qt)
  3. frame_marker           -- frame_marker_{x,y} set_data + FOI + ylim recompute
  4. decode                 -- self.data[idx].asnumpy()  (PyAV TOC + decode)
  5. image_process          -- DUSTrack's enhance_ultrasound_image()
  6. imshow_set_data        -- self._im.set_data(processed)
  7. title                  -- self._ax.set_title(self.titlefunc(self))
  8. update_assets          -- super().update_assets() (button / memslot / etc)
  9. plt_draw               -- two plt.draw() calls (schedule only -- no raster)

Then, outside the timed update():
 10. process_events         -- app.processEvents() drains Qt's paint queue
                               (this is where matplotlib actually rasters)

The sum of (1..10) should roughly equal one full iteration of the
existing 09_benchmark_dustrack.py loop body.

Reads
-----
- 1.3.0 baseline median: 141.6 ms
- 1.4.0-qt median:       128.6 ms  (Qt widget swap shaved ~13 ms)
- enhance_ultrasound_image microbench:  ~1.4 ms  (probe 10)

Usage
-----
    C:\\Users\\praneeth\\anaconda3\\envs\\dlc\\python.exe \\
        C:\\dev\\datanavigator\\tests\\qt_learning\\11_profile_dustrack_update.py \\
        [--n-frames N] [--n-warmup N]
"""

import argparse
import collections
import logging
import os
import statistics
import sys
import time

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


# --- profiled update --------------------------------------------------

# Segment names in the order they fire.
SEGMENTS = [
    "annotation_visibility",
    "statevars_display",
    "frame_marker",
    "decode",
    "image_process",
    "imshow_set_data",
    "title",
    "update_assets",
    "plt_draw",
]


def install_profiled_update(ret, tracker):
    """Replace ret.update with an instrumented version.

    The replacement mirrors VideoPointAnnotator.update() and
    VideoBrowser.update() exactly -- same call order, same kwargs --
    but wraps each segment in a time.perf_counter() span that appends
    into tracker[name].
    """

    def _now():
        return time.perf_counter()

    # Stash original entry points so we can call into them.
    _ann_vis = ret.update_annotation_visibility
    _sv_disp = ret.statevariables.update_display
    _frame_marker = ret.update_frame_marker
    _video_data = ret.data
    _image_process = ret.image_process_func
    _im = ret._im
    _ax = ret._ax
    _titlefunc = ret.titlefunc
    _update_assets = ret.update_assets

    def _profiled_update(self=None, event=None):
        t0 = _now(); _ann_vis(draw=False)
        t1 = _now(); _sv_disp(draw=False)
        t2 = _now(); _frame_marker(draw=False)

        # Inline of VideoBrowser.update body so we can time decode vs.
        # image_process vs. imshow_set_data separately. Equivalent call
        # sequence; no behavior change.
        t3 = _now(); raw = _video_data[ret._current_idx].asnumpy()
        t4 = _now(); processed = _image_process(raw)
        t5 = _now(); _im.set_data(processed)
        t6 = _now(); _ax.set_title(_titlefunc(ret))
        t7 = _now(); _update_assets()
        t8 = _now(); plt.draw(); plt.draw()
        t9 = _now()

        tracker["annotation_visibility"].append((t1 - t0) * 1000)
        tracker["statevars_display"].append((t2 - t1) * 1000)
        tracker["frame_marker"].append((t3 - t2) * 1000)
        tracker["decode"].append((t4 - t3) * 1000)
        tracker["image_process"].append((t5 - t4) * 1000)
        tracker["imshow_set_data"].append((t6 - t5) * 1000)
        tracker["title"].append((t7 - t6) * 1000)
        tracker["update_assets"].append((t8 - t7) * 1000)
        tracker["plt_draw"].append((t9 - t8) * 1000)

    ret.update = _profiled_update


# --- stats helpers ----------------------------------------------------

def _stats(times_ms):
    if not times_ms:
        return None
    s = sorted(times_ms)
    n = len(s)
    return {
        "n": n,
        "min": min(times_ms),
        "median": statistics.median(times_ms),
        "mean": statistics.mean(times_ms),
        "p95": s[int(n * 0.95)] if n >= 20 else s[-1],
        "max": max(times_ms),
    }


def _fmt_row(label, st):
    if st is None:
        return f"  {label:26s} --"
    return (
        f"  {label:26s} "
        f"median={st['median']:7.2f}  "
        f"mean={st['mean']:7.2f}  "
        f"p95={st['p95']:7.2f}  "
        f"min={st['min']:7.2f}  "
        f"max={st['max']:7.2f}"
    )


# --- main -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--video-index", type=int, default=0)
    parser.add_argument("--n-frames", type=int, default=200)
    parser.add_argument("--n-warmup", type=int, default=15)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.stderr.write(f"config not found: {args.config}\n")
        return 2

    from dustrack import DLCProject
    import dustrack
    import datanavigator

    print("=" * 72)
    print("DUSTrack update() segment profile")
    print(f"  datanavigator: {datanavigator.__file__}")
    print(f"  dustrack     : {dustrack.__file__}")
    print(f"  backend      : {matplotlib.get_backend()}, qt_api={os.environ.get('QT_API')}")
    print(f"  config       : {args.config}")
    print(f"  frames       : {args.n_frames}  ({args.n_warmup} warmup discarded)")
    print("=" * 72)

    print("opening DLCProject...")
    g = DLCProject(args.config)
    print(f"  using: {g.video_list[args.video_index]}")

    print("calling annotate() (opens UI window)...")
    ret = g.annotate(video_index=args.video_index)
    n_total_frames = len(ret)
    print(f"  {n_total_frames} frames")

    # Drain initial paints.
    for _ in range(10):
        app.processEvents()

    tracker = collections.defaultdict(list)
    install_profiled_update(ret, tracker)

    process_events_ms = []
    total_iter_ms = []

    n_frames = min(args.n_frames, n_total_frames)
    print(f"running {n_frames} iterations...")
    for i in range(n_frames):
        ret._current_idx = i % n_total_frames
        t0 = time.perf_counter()
        ret.update()
        t1 = time.perf_counter()
        app.processEvents()
        t2 = time.perf_counter()
        if i >= args.n_warmup:
            process_events_ms.append((t2 - t1) * 1000)
            total_iter_ms.append((t2 - t0) * 1000)

    # Trim warmup from per-segment trackers (the install_profiled_update
    # callback records every call; warmup needs the same slice).
    for k in tracker:
        tracker[k] = tracker[k][args.n_warmup:]

    print()
    print("=" * 72)
    print("Per-segment timings (ms)")
    print("=" * 72)

    seg_medians_sum = 0.0
    for name in SEGMENTS:
        st = _stats(tracker.get(name, []))
        if st is not None:
            seg_medians_sum += st["median"]
        print(_fmt_row(name, st))

    pe_st = _stats(process_events_ms)
    print(_fmt_row("process_events (raster)", pe_st))

    print("-" * 72)
    total_st = _stats(total_iter_ms)
    print(_fmt_row("TOTAL (update + raster)", total_st))
    if pe_st is not None:
        print(f"  (sum of segment medians: {seg_medians_sum:.2f} ms, "
              f"raster median: {pe_st['median']:.2f} ms, "
              f"total median: {total_st['median']:.2f} ms)")

    print("=" * 72)
    if pe_st is not None and total_st is not None:
        raster_frac = pe_st["median"] / total_st["median"] * 100
        print(
            f"Canvas raster (process_events) is {raster_frac:.1f}% of total."
        )
        print(
            "Blit-mode would target this segment + any draw_idle scheduled inside"
        )
        print("update() that hasn't fired yet at the t1 boundary.")
    print("=" * 72)

    plt.close(ret.figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
