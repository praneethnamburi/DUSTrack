"""
Feasibility probe: how much would blit-mode save on real DUSTrack?

After the cache_quick_wins commit, real DUSTrack sits at ~93.8 ms /
frame, of which ~82 ms is canvas raster (`processEvents()`). Blit
should target that raster, but the actual savings depend on what
fraction of it is imshow content (necessarily changing) vs the
surrounding axes content (cacheable).

Approach: open the real DUSTrack UI like 09_benchmark_dustrack.py,
mark the per-frame-changing artists as `animated=True`, cache each
axis' background once, then for each iteration:

  1. Run the existing update() to set_data on dynamic artists.
  2. canvas.restore_region(bg) for each axis.
  3. ax.draw_artist(artist) for each dynamic artist.
  4. canvas.blit(ax.bbox) for each axis.
  5. canvas.flush_events() to push to the screen.

Skips `processEvents()` for this measurement (blit + flush_events is
the equivalent path).

Dynamic artists (after cache_quick_wins, the per-label trace lines
are effectively static — they only set_ydata on annotation mutation):

  _ax_image    : _im (imshow), per-annotation `labels_in_ax0` scatter
  _ax_trace_x  : _frame_marker_x, _plot_frames_of_interest_x
  _ax_trace_y  : _frame_marker_y, _plot_frames_of_interest_y

`_ax_statevar` is empty under 1.4.0-qt (the Qt overlay handles
statevariables), so no blit needed there.

The per-frame title update on _ax_image is deliberately skipped —
the title pixel region becomes stale during the run. Real production
blit-mode needs a strategy for this (separate Text artist treated as
dynamic, Qt overlay, or refresh-less-often). For *feasibility*
measurement, frozen title is acceptable.

Result (2026-05-17, dlc env, PySide6 6.4.2, interosseous_pn24-x)
-----------------------------------------------------------------
Blit-mode did **not** help: baseline 100.95 ms median -> blit
108.77 ms median (worse by ~8 ms). Internal blit breakdown:

  restore_region       0.29 ms    (cheap, as expected)
  draw_artist_light    0.49 ms    (frame_marker / FOI / scatter)
  draw_artist_imshow   13.63 ms   (Agg raster of imshow)
  canvas.blit(bbox)    83.39 ms   <-- Qt widget pixmap upload
  flush_events         0.50 ms    (cheap)
  reference: full canvas.draw()  ~80 ms

**Finding**: the per-frame raster cost on QtAgg with this figure
size is dominated by the Qt buffer upload (~83 ms), not by Agg
rasterization (~13 ms). Blit cannot beat full-draw on this stack
because both pay the same widget-upload cost.

Per-axis blits did not help either: with three axes whose bboxes
roughly tile the figure, per-axis upload area sums to ~ full-figure
area. A single full-figure blit cost the same as per-axis.

Eliminating the upload cost requires architectural change -- an
OpenGL-backed canvas (matplotlib has experimental backends; not
robust on Windows) or rendering the imshow outside matplotlib via
QGraphicsView / QOpenGLWidget / raw QLabel(QPixmap), keeping
matplotlib only for the trace axes. That is sized like the
datanavigator 2.0.0 from-scratch Qt rewrite from the spec, not
under-the-hood 1.4.0.

Kept around as load-bearing evidence -- re-run this probe if
matplotlib's Qt backend changes, if a different rendering pipeline
is being considered, or if the figure size / layout changes
materially.
"""

import argparse
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
        f"  {label:30s} "
        f"median={st['median']:7.2f}  "
        f"mean={st['mean']:7.2f}  "
        f"p95={st['p95']:7.2f}  "
        f"min={st['min']:7.2f}  "
        f"max={st['max']:7.2f}  "
        f"n={st['n']}"
    )


def collect_dynamic_artists(ret):
    """Return {Axes: [dynamic Artists]} for the live VideoPointAnnotator."""
    axes_artists = {
        ret._ax_image: [ret._im],
        ret._ax_trace_x: [ret._frame_marker_x, ret._plot_frames_of_interest_x],
        ret._ax_trace_y: [ret._frame_marker_y, ret._plot_frames_of_interest_y],
    }
    # Per-annotation scatter on the image axis.
    for ann in ret.annotations._list:
        scatter = ann.plot_handles.get("labels_in_ax0")
        if scatter is not None:
            axes_artists[ret._ax_image].append(scatter)
    return axes_artists


def enable_blit(ret):
    canvas = ret.figure.canvas
    axes_artists = collect_dynamic_artists(ret)

    # Mark dynamic artists animated so the normal draw skips them.
    for ax, artists in axes_artists.items():
        for artist in artists:
            artist.set_animated(True)

    # Force a fresh draw with the new animated flags, then cache.
    canvas.draw()
    backgrounds = {ax: canvas.copy_from_bbox(ax.bbox) for ax in axes_artists}
    return axes_artists, backgrounds


_blit_segments = {}


def blit_render(ret, axes_artists, backgrounds):
    canvas = ret.figure.canvas
    t0 = time.perf_counter()
    for ax, bg in backgrounds.items():
        canvas.restore_region(bg)
    t1 = time.perf_counter()
    # Split imshow draw_artist from the lighter dynamic artists.
    for ax, artists in axes_artists.items():
        for artist in artists:
            if ax is ret._ax_image and artist is ret._im:
                continue
            ax.draw_artist(artist)
    t2 = time.perf_counter()
    ret._ax_image.draw_artist(ret._im)
    t3 = time.perf_counter()
    # Single full-figure blit -- per-axis bboxes overlap on this layout,
    # so a single bbox covering everything uploads less total pixel area
    # to Qt than three per-axis calls.
    canvas.blit(ret.figure.bbox)
    t4 = time.perf_counter()
    canvas.flush_events()
    t5 = time.perf_counter()
    _blit_segments.setdefault("restore_region", []).append((t1 - t0) * 1000)
    _blit_segments.setdefault("draw_artist_light", []).append((t2 - t1) * 1000)
    _blit_segments.setdefault("draw_artist_imshow", []).append((t3 - t2) * 1000)
    _blit_segments.setdefault("blit", []).append((t4 - t3) * 1000)
    _blit_segments.setdefault("flush_events", []).append((t5 - t4) * 1000)


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

    print("=" * 70)
    print("Blit feasibility probe")
    print(f"  config       : {args.config}")
    print(f"  frames       : {args.n_frames} ({args.n_warmup} warmup discarded)")
    print("=" * 70)

    g = DLCProject(args.config)
    ret = g.annotate(video_index=args.video_index)
    n_total = len(ret)
    print(f"DUSTrack instance has {n_total} frames")

    for _ in range(10):
        app.processEvents()

    print("Baseline pass (no blit) ...")
    baseline_ms = []
    n_frames = min(args.n_frames, n_total)
    for i in range(n_frames):
        ret._current_idx = i % n_total
        t0 = time.perf_counter()
        ret.update()
        app.processEvents()
        t1 = time.perf_counter()
        if i >= args.n_warmup:
            baseline_ms.append((t1 - t0) * 1000)

    print("Enabling blit (marking dynamic artists animated, caching bgs) ...")
    axes_artists, backgrounds = enable_blit(ret)

    print("Blit pass ...")
    blit_ms = []
    for i in range(n_frames):
        ret._current_idx = i % n_total
        t0 = time.perf_counter()
        # Mirror VideoPointAnnotator.update() body but replace plt.draw +
        # processEvents with blit_render. Skip the per-frame title set
        # (its raster region becomes stale -- acceptable for feasibility).
        ret.update_annotation_visibility(draw=False)
        ret.statevariables.update_display(draw=False)
        ret.update_frame_marker(draw=False)
        raw = ret.data[ret._current_idx].asnumpy()
        ret._im.set_data(ret.image_process_func(raw))
        ret.update_assets()
        blit_render(ret, axes_artists, backgrounds)
        t1 = time.perf_counter()
        if i >= args.n_warmup:
            blit_ms.append((t1 - t0) * 1000)

    base = _stats(baseline_ms)
    blit = _stats(blit_ms)
    print()
    print("=" * 70)
    _print_stats("baseline (update + raster)", base)
    _print_stats("blit (update + blit_render)", blit)
    print("-" * 70)
    delta = base["median"] - blit["median"]
    speedup = base["median"] / blit["median"]
    print(f"  delta (median): {delta:+.2f} ms   ({speedup:.2f}x)")
    print(f"  baseline fps  : {1000.0 / base['median']:.1f}")
    print(f"  blit fps      : {1000.0 / blit['median']:.1f}")
    print("-" * 70)
    print("blit_render() internal breakdown (median ms):")
    seg_total = 0.0
    for name in ("restore_region", "draw_artist_light",
                 "draw_artist_imshow", "blit", "flush_events"):
        samples = _blit_segments.get(name, [])[args.n_warmup:]
        if samples:
            st = _stats(samples)
            seg_total += st["median"]
            print(f"  {name:24s} median={st['median']:6.2f}  "
                  f"mean={st['mean']:6.2f}  p95={st['p95']:6.2f}")
    print(f"  {'sum (sanity check)':24s} {seg_total:6.2f}")

    # Also measure the synchronous full draw cost, for reference.
    canvas = ret.figure.canvas
    full_draw_ms = []
    for _ in range(30):
        t0 = time.perf_counter()
        canvas.draw()
        t1 = time.perf_counter()
        full_draw_ms.append((t1 - t0) * 1000)
    full_draw_ms = full_draw_ms[5:]
    fd = _stats(full_draw_ms)
    print(f"  ---")
    print(f"  reference: full canvas.draw() median={fd['median']:.2f} ms")
    print("=" * 70)

    plt.close(ret.figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
