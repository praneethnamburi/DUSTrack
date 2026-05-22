"""Sandbox 14b -- single-video probe 14 WITH paintEvent counter.

Diagnostic counterpart to 28b. Single-video sessions were never
affected by the multi-video stale-paint bug; canvas.update() is
pure cost-add in single-video. This probe measures the per-update
paintEvent count + the median update cost in single-video to
confirm.

Hypothesis: flush_events alone delivers ~200 paintEvents in
single-video (so it's working correctly there). If true,
canvas.update should be gated on multi-video-only to preserve
single-video perf.
"""
from __future__ import annotations

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

from qtpy.QtCore import QEvent  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

DEFAULT_CONFIG = r"M:\DLC_MODELS\general\interosseous_pn24-x-2025-10-24\config.yaml"


def main() -> int:
    import dustrack
    from dustrack import DLCProject, DUSTrack
    from dustrack.dlcinterface import VideoFileManager

    print("=" * 60)
    print("DUSTrack single-video bench WITH PaintEvent counter")
    print("=" * 60)

    g = DLCProject(DEFAULT_CONFIG)
    print(f"  {len(g.video_list)} videos in project")
    video_index = 0
    video = g.video_list[video_index]
    print(f"  using: {video}")

    new_iteration_num = g.latest_iteration
    if g.latest_iteration_is_trained:
        new_iteration_num += 1
    new_annotation_suffix = f"iteration-{new_iteration_num}"
    fm = VideoFileManager(g, video_index)
    annotation_names = fm.get_all_annotation_layers(new_annotation_suffix)
    annotation_names["buffer"] = fm.get_new_json("buffer")

    tracker = DUSTrack(
        g.video_list[video_index],
        annotation_names,
        height_ratios=(3, 1, 1),
        fast_render=True,
    )
    tracker.update()
    print(f"DUSTrack instance has {len(tracker.data)} frames")
    print(f"  bundle count: {len(tracker._bundles)}")

    for _ in range(20):
        app.processEvents()

    canvas = tracker.figure.canvas
    paint_count = [0]
    original_event = canvas.event

    def counting_event(ev):
        try:
            if ev.type() == QEvent.Paint:
                paint_count[0] += 1
        except Exception:
            pass
        return original_event(ev)

    canvas.event = counting_event

    n_total_frames = len(tracker.data)
    n_frames = 200
    n_warmup = 15
    print(f"running {n_frames} update() calls...")

    paint_count[0] = 0
    times_ms = []
    for i in range(n_frames):
        tracker._current_idx = i % n_total_frames
        t0 = time.perf_counter()
        tracker.update()
        app.processEvents()
        t1 = time.perf_counter()
        if i >= n_warmup:
            times_ms.append((t1 - t0) * 1000)

    paint_after = paint_count[0]

    s = sorted(times_ms)
    n = len(s)
    median = statistics.median(times_ms)
    p95 = s[int(n * 0.95)]
    print(f"N measured       = {n}")
    print(f"median           = {median:7.2f} ms")
    print(f"p95              = {p95:7.2f} ms")
    print(f"fps (median)     = {1000.0 / median:6.1f}")
    print()
    print(f"paintEvents delivered during {n_frames} update calls: "
          f"{paint_after}")
    print("=" * 60)

    plt.close(tracker.figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
