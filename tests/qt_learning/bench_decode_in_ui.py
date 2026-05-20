"""Probe: time per-frame decode inside the UI update() loop, post-seek-fix.

The 2026-05-20 _seek_packet fix lifted standalone sequential decode
from ~185 to ~268 fps (5.4 -> 3.7 ms/frame), but the steady-state UI
bench only moved from 40.30 -> 40.03 ms. Want to know: is decode
actually faster inside the UI, or is something compensating?

Monkey-patches dnav.VideoReader.__getitem__ to time every call.
Runs the same UI access pattern as probe 14. Reports median decode
per frame.
"""
from __future__ import annotations

import os
import statistics
import sys
import time

os.environ.setdefault("QT_API", "pyside6")

import matplotlib  # noqa: E402
matplotlib.use("QtAgg")

from qtpy.QtCore import QtMsgType, qInstallMessageHandler  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

qInstallMessageHandler(lambda *a: None)
app = QApplication.instance() or QApplication([])

# Instrument before any other dustrack/dnav code can take references.
import datanavigator.video_reader as _vr
_orig_getitem = _vr.VideoReader.__getitem__
_decode_times = []

def _timed_getitem(self, key):
    t0 = time.perf_counter()
    result = _orig_getitem(self, key)
    _decode_times.append(time.perf_counter() - t0)
    return result

_vr.VideoReader.__getitem__ = _timed_getitem

from dustrack import DLCProject, DUSTrack
from dustrack.dlcinterface import VideoFileManager

DEFAULT_CONFIG = r"M:\DLC_MODELS\general\interosseous_pn24-x-2025-10-24\config.yaml"
g = DLCProject(DEFAULT_CONFIG)
video_index = 0
if g.latest_iteration_is_trained():
    new_iteration_num = g.latest_iteration + 1
else:
    new_iteration_num = g.latest_iteration
new_annotation_suffix = f"iteration-{new_iteration_num}"
fm = VideoFileManager(g, video_index)
annotation_names = fm.get_all_annotation_layers(new_annotation_suffix)
annotation_names["buffer"] = fm.get_new_json("buffer")

ret = DUSTrack(
    g.video_list[video_index],
    annotation_names,
    height_ratios=(3, 1, 1),
    fast_render=True,
)
for ann in ret.annotations:
    if "dlc_" in ann.name:
        ann.set_plot_type("line")
ret.update()
for _ in range(10):
    app.processEvents()

n_total = len(ret)

# Warmup phase: don't reset decode_times since we want all reads visible
warmup_start = len(_decode_times)
for i in range(15):
    ret._current_idx = i
    ret.update()
    app.processEvents()
warmup_end = len(_decode_times)

# Measured phase
measure_start = warmup_end
for i in range(15, 200):
    ret._current_idx = i
    ret.update()
    app.processEvents()
measure_end = len(_decode_times)

warmup_times = _decode_times[warmup_start:warmup_end]
measure_times = _decode_times[measure_start:measure_end]

print(f"warmup decode reads:  {len(warmup_times)}, median {statistics.median(warmup_times)*1000:.3f} ms")
print(f"measure decode reads: {len(measure_times)}, median {statistics.median(measure_times)*1000:.3f} ms")
print(f"measure decode mean:  {statistics.mean(measure_times)*1000:.3f} ms")
print(f"measure decode max:   {max(measure_times)*1000:.3f} ms")
