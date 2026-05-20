"""
Feasibility probe 13: hybrid canvas (Qt video pane + matplotlib overlay).

Decides whether VideoBrowser / VideoPointAnnotator / VideoPlotBrowser
should grow a Qt-side video display path to bypass the ~82 ms canvas
pixmap upload identified in probe 12 (BENCHMARKING.md).

Two candidate architectures
---------------------------

  Option B  (under-the-hood; annotation overlay unchanged)
      A full-figure transparent matplotlib canvas sits on top of a
      QLabel that holds the decoded frame as a QPixmap. Scatter
      points and trace axes stay matplotlib-drawn. Savings come
      from skipping Agg's imshow rasterization (~13 ms per probe
      12). The QtAgg widget pixmap upload area is unchanged (full
      figure), so the dominant Qt-side cost persists.

  Option A  (annotation scatter moves to Qt-native overlay)
      The matplotlib canvas covers only the trace region (smaller
      area, smaller upload). The video frame is delivered via
      QLabel(QPixmap). The annotation scatter overlay is no longer
      a matplotlib PathCollection on the image Axes; it must be
      re-rendered via Qt (QGraphicsScene item, custom paint, or
      similar). Not under-the-hood for the scatter render path.

Method
------
Open the real DUSTrack UI (interosseous_pn24-x, same video as
probes 09/11/12). Then measure:

  1. Baseline           -- current update + processEvents
  2. canvas.draw (imshow visible)   -- reference, synchronous
  3. canvas.draw (imshow hidden)    -- approximates Option B's mpl
                                       cost (Agg skips imshow raster,
                                       upload area unchanged)
  4. small canvas.draw  -- separate Figure sized to the trace region
                           only, two synthetic line plots of similar
                           density; approximates Option A's mpl cost
  5. QLabel pixmap upload -- decode a frame, QImage(rgb)->QPixmap->
                             label.setPixmap + processEvents

Then predict per-frame totals:

  Option B  = (update body residual) + (canvas.draw imshow-hidden) +
              (qlabel upload)
  Option A  = (update body residual minus large mpl axes cost) +
              (small canvas.draw) + (qlabel upload)

The "update body residual" after cache_quick_wins is ~11 ms
(BENCHMARKING.md probe 11 +cache column subtotal). For Option A
we further subtract the per-frame ydata work that targets the
*image* axes (small scatter offsets), if any -- conservatively
keep all 11 ms.

Usage
-----
    C:\\Users\\praneeth\\anaconda3\\envs\\dlc\\python.exe \\
        C:\\dev\\datanavigator\\tests\\qt_learning\\13_benchmark_hybrid_canvas_feasibility.py \\
        [--n-frames N] [--n-warmup N]

Reads off
---------
- 1.4.0-qt + cache_quick_wins baseline:  93.8 ms median (BENCHMARKING.md)
- Probe 12: full canvas.draw() ~80 ms, of which ~13 ms is Agg
  imshow raster and ~83 ms is canvas pixmap upload
"""

import argparse
import logging
import os
import statistics
import sys
import time

os.environ.setdefault("QT_API", "pyside6")
logging.getLogger("numexpr.utils").setLevel(logging.WARNING)

import numpy as np  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("QtAgg")
from matplotlib import pyplot as plt  # noqa: E402

from qtpy.QtCore import Qt, QtMsgType, qInstallMessageHandler  # noqa: E402
from qtpy.QtGui import QImage, QPixmap  # noqa: E402
from qtpy.QtWidgets import QApplication, QLabel  # noqa: E402


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


def _stats(times_ms):
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


def _print_stats(label, st):
    print(
        f"  {label:38s} "
        f"median={st['median']:7.2f}  "
        f"mean={st['mean']:7.2f}  "
        f"p95={st['p95']:7.2f}  "
        f"min={st['min']:7.2f}  "
        f"max={st['max']:7.2f}  "
        f"n={st['n']}"
    )


def _as_rgb_uint8(arr):
    """Coerce a decoded frame (ndarray) into contiguous H x W x 3 uint8."""
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)


def _ndarray_to_qpixmap(arr):
    """Build a QPixmap from an H x W x 3 uint8 ndarray. The QImage holds a
    view of `arr`'s buffer, so callers must keep `arr` alive until the
    QPixmap.fromImage copy completes (it does so synchronously)."""
    h, w, _ = arr.shape
    img = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(img)


def measure_qlabel_upload(decoded_frames, label, n_warmup):
    times_ms = []
    for i, frame in enumerate(decoded_frames):
        rgb = _as_rgb_uint8(frame)
        t0 = time.perf_counter()
        pixmap = _ndarray_to_qpixmap(rgb)
        label.setPixmap(pixmap)
        app.processEvents()
        t1 = time.perf_counter()
        if i >= n_warmup:
            times_ms.append((t1 - t0) * 1000)
    return times_ms


def measure_canvas_draw(figure, n_iters, n_warmup):
    canvas = figure.canvas
    times_ms = []
    for i in range(n_iters):
        t0 = time.perf_counter()
        canvas.draw()
        t1 = time.perf_counter()
        if i >= n_warmup:
            times_ms.append((t1 - t0) * 1000)
    return times_ms


def build_trace_only_figure(width_in, height_in, n_points):
    """Synthetic 'small canvas' mirroring the trace-region layout of
    VideoPointAnnotator: two stacked axes filling the full width, two
    line plots per axis (trace + frame_marker / FOI), no imshow."""
    fig = plt.figure(figsize=(width_in, height_in), constrained_layout=True)
    gs = fig.add_gridspec(2, 1)
    ax_x = fig.add_subplot(gs[0, 0])
    ax_y = fig.add_subplot(gs[1, 0])
    rng = np.random.default_rng(0)
    t = np.arange(n_points)
    for ax in (ax_x, ax_y):
        ax.plot(t, rng.standard_normal(n_points) * 100, lw=0.8)
        ax.plot(t, rng.standard_normal(n_points) * 100, lw=0.8)
        ax.set_xlim(0, n_points)
    fig.canvas.draw()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--video-index", type=int, default=0)
    parser.add_argument("--n-frames", type=int, default=120)
    parser.add_argument("--n-warmup", type=int, default=15)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.stderr.write(f"config not found: {args.config}\n")
        return 2

    from dustrack import DLCProject

    print("=" * 78)
    print("Hybrid-canvas feasibility probe")
    print(f"  config       : {args.config}")
    print(f"  frames       : {args.n_frames} ({args.n_warmup} warmup discarded)")
    print("=" * 78)

    g = DLCProject(args.config)
    ret = g.annotate(video_index=args.video_index)
    n_total = len(ret)
    print(f"DUSTrack instance has {n_total} frames")

    for _ in range(10):
        app.processEvents()

    n_frames = min(args.n_frames, n_total)

    # -- 1. Baseline (current update + processEvents) ---------------------
    print("\n[1/5] Baseline (current update + processEvents) ...")
    baseline_ms = []
    for i in range(n_frames):
        ret._current_idx = i % n_total
        t0 = time.perf_counter()
        ret.update()
        app.processEvents()
        t1 = time.perf_counter()
        if i >= args.n_warmup:
            baseline_ms.append((t1 - t0) * 1000)

    # -- 2. Reference: full canvas.draw() (imshow visible) -----------------
    print("[2/5] Reference canvas.draw() (imshow visible) ...")
    canvas_with_imshow_ms = measure_canvas_draw(
        ret.figure, n_frames, args.n_warmup,
    )

    # -- 3. canvas.draw() with imshow hidden  (Option B's mpl cost) --------
    print("[3/5] canvas.draw() with imshow hidden (Option B mpl cost) ...")
    ret._im.set_visible(False)
    ret.figure.canvas.draw()  # force first draw to settle with new visibility
    canvas_no_imshow_ms = measure_canvas_draw(
        ret.figure, n_frames, args.n_warmup,
    )
    ret._im.set_visible(True)
    ret.figure.canvas.draw()

    # -- 4. Small (trace-only) canvas  (Option A's mpl cost) ---------------
    print("[4/5] Small trace-only canvas.draw() (Option A mpl cost) ...")
    # Trace region: full figure width (~12 in), trace height ~= 2/3 of 8 in
    # for a 3-row gridspec where row 0 is the image strip. Approximate.
    fig_w_in, fig_h_in = ret.figure.get_size_inches()
    trace_h_in = fig_h_in * (2.0 / 3.0)
    small_fig = build_trace_only_figure(fig_w_in, trace_h_in, n_points=n_total)
    small_canvas_ms = measure_canvas_draw(small_fig, n_frames, args.n_warmup)

    # -- 5. QLabel pixmap upload  (Qt-side video pane cost) ----------------
    print("[5/5] QLabel pixmap upload (Qt-side video pane cost) ...")
    label = QLabel()
    sample_frame = _as_rgb_uint8(ret.data[0].asnumpy())
    h, w, _ = sample_frame.shape
    label.resize(w, h)
    label.show()
    for _ in range(5):
        app.processEvents()
    decoded = [ret.data[i % n_total].asnumpy() for i in range(n_frames)]
    qlabel_ms = measure_qlabel_upload(decoded, label, args.n_warmup)
    label.hide()
    label.deleteLater()

    # -- Stats ------------------------------------------------------------
    base = _stats(baseline_ms)
    canvas_yes = _stats(canvas_with_imshow_ms)
    canvas_no = _stats(canvas_no_imshow_ms)
    small = _stats(small_canvas_ms)
    qlabel = _stats(qlabel_ms)

    # Predicted hybrids
    # Update-body residual after cache_quick_wins ~11 ms (BENCHMARKING.md
    # probe 11 +cache column subtotal). Treat as a constant.
    UPDATE_BODY_RESIDUAL_MS = 11.0
    predicted_B = UPDATE_BODY_RESIDUAL_MS + canvas_no["median"] + qlabel["median"]
    predicted_A = UPDATE_BODY_RESIDUAL_MS + small["median"] + qlabel["median"]

    print()
    print("=" * 78)
    print("Per-frame timings (medians, ms unless noted)")
    print("-" * 78)
    _print_stats("baseline (today's path)", base)
    _print_stats("canvas.draw() w/ imshow", canvas_yes)
    _print_stats("canvas.draw() w/o imshow", canvas_no)
    _print_stats("small canvas.draw() (trace-only)", small)
    _print_stats("QLabel.setPixmap + processEvents", qlabel)
    print("-" * 78)
    print(f"Implied components (medians, ms)")
    print(f"  Agg imshow raster (diff)              "
          f"{canvas_yes['median'] - canvas_no['median']:+7.2f}")
    print(f"  upload-cost reduction (small canvas)  "
          f"{canvas_yes['median'] - small['median']:+7.2f}")
    print()
    print(f"Predicted hybrid per-frame totals")
    print(f"  Option B (under-the-hood, scatter unchanged):  "
          f"{predicted_B:6.2f} ms  -> {1000.0 / predicted_B:5.1f} fps  "
          f"({base['median'] / predicted_B:.2f}x vs baseline)")
    print(f"  Option A (scatter moves to Qt-native overlay): "
          f"{predicted_A:6.2f} ms  -> {1000.0 / predicted_A:5.1f} fps  "
          f"({base['median'] / predicted_A:.2f}x vs baseline)")
    print("=" * 78)
    print("Notes:")
    print("  - 'update body residual' set to 11 ms (BENCHMARKING.md probe 11).")
    print("  - Option B's mpl canvas remains full-figure-size, so Qt upload")
    print("    area is unchanged; savings = Agg imshow raster only.")
    print("  - Option A's mpl canvas is smaller (trace region only) so upload")
    print("    area shrinks proportionally. Scatter must re-render via Qt.")
    print("  - QLabel cost includes processEvents (matches probe 12's")
    print("    raster-cost methodology).")

    plt.close(ret.figure)
    plt.close(small_fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
