"""Sandbox 28b -- multi-video update() bench WITH paintEvent counter.

Same shape as probe 28 (continuous frame walk on active bundle, 200
iterations, 15 warmup discarded) but installs a QEvent.Paint
counter on the canvas widget BEFORE the timing loop starts. After
the loop, reports how many paintEvents the canvas actually
received.

Diagnostic intent: at c1dc63b (flush_events only) the bench measures
22 ms / 45 fps, vs HEAD (canvas.update + flush_events) at 51 ms / 19
fps. The hypothesis is that c1dc63b's flush_events alone fails to
trigger actual painting under multi-video (the "stale trace" bug)
and the 22 ms measures processEvents draining an empty queue. If
true, paint_count << 200; if false, paint_count >= 200 and the
bench-artifact theory is wrong.
"""
from __future__ import annotations

import logging
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")
logging.getLogger("numexpr.utils").setLevel(logging.WARNING)

import matplotlib  # noqa: E402
matplotlib.use("QtAgg")
from matplotlib import pyplot as plt  # noqa: E402

from qtpy.QtCore import QCoreApplication, QEvent  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

DEFAULT_PROJECT = Path(
    "M:/DLC_MODELS/participant_models_general/s006/RFA/"
    "interosseous_pn24-x-2025-10-24"
)


def _wait_for_hydration(tracker, timeout=300.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(b.is_terminal for b in tracker._bundles):
            return
        QCoreApplication.processEvents()
        time.sleep(0.05)
    raise TimeoutError("hydration timed out")


def main() -> int:
    import dustrack

    print("=" * 60)
    print("DUSTrack multi-video bench WITH PaintEvent counter")
    print(f"  dustrack: {dustrack.__file__}")
    print(f"  project : {DEFAULT_PROJECT}")
    print("=" * 60)

    print("opening multi-video session...")
    t0 = time.time()
    tracker = dustrack.open(DEFAULT_PROJECT)
    print(f"  active bundle constructed in {time.time() - t0:.2f}s")
    print(f"  {len(tracker._bundles)} bundles total")

    print("waiting for background hydration...")
    t0 = time.time()
    _wait_for_hydration(tracker)
    print(f"  all bundles hydrated in {time.time() - t0:.2f}s")

    for _ in range(20):
        app.processEvents()

    canvas = tracker.figure.canvas
    print(f"  canvas widget: {type(canvas).__name__}")

    # Install paint-event counter on the canvas widget.
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
    print(f"running {n_frames} update() calls on active bundle...")

    paint_count[0] = 0  # reset just before the timing loop
    paint_before = paint_count[0]

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
          f"{paint_after - paint_before}")
    print(f"  (interpretation: if {n_frames}-ish, painting is happening; "
          f"if 0 or 1, painting is NOT happening)")
    print("=" * 60)

    plt.close(tracker.figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
