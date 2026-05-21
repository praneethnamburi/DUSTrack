# DUSTrack -- production workflow benchmarks

Wall-clock numbers for the user-felt DUSTrack workflow on the
`interosseous_pn24-x` pia02 DLC project, tracked across releases.
Three axes:

- **Steady-state per-frame responsiveness** — once a session is
  open, how fast does `browser.update()` repaint after a navigation
  key? Probes
  [`09_benchmark_dustrack.py`](tests/qt_learning/09_benchmark_dustrack.py)
  (Tier 1 / mpl-path) and
  [`14_benchmark_fast_render.py`](tests/qt_learning/14_benchmark_fast_render.py)
  (Tier 2 / fast_render Qt-native).
- **Cold-open** — clock starts at `DLCProject(config)`, stops at
  the first painted frame of `g.annotate(video_index=…)`. Probe
  [`24_benchmark_cold_open.py`](tests/qt_learning/24_benchmark_cold_open.py).
  Dominant cost on pia02-scale sessions (120 videos, many annotation
  layers per video, network-drive video paths).
- **Post-processing throughput** — `lk_moving_average_filter`
  (the `Reduce jitter` GUI button) wall-clock + memory, tracked
  across the 2026-05-21 perf pass. Probes
  [`25_benchmark_lk_rstc.py`](tests/qt_learning/25_benchmark_lk_rstc.py)
  (micro + macro per-LK-call benches on the dnav example video) and
  [`_reduce_jitter_real_bench.py`](tests/qt_learning/_reduce_jitter_real_bench.py)
  (end-to-end on the pia02 real-shape fixture).

Companion document
[`datanavigator/BENCHMARKING.md`](../datanavigator/BENCHMARKING.md)
holds the synthetic probe-08 baseline and the dnav-side render-pipeline
lessons (blit feasibility on QtAgg, fast_render Tier 2 architecture,
pre-decode rationale) — those are about `VideoBrowser` per-frame
infrastructure that lives in dnav.

## Steady-state per-frame -- summary

Real DUSTrack UI, dlc env, PySide6 6.4.2, matplotlib 3.8.4,
`interosseous_pn24-x` video.

| Release / Branches                   | Median ms | Mean ms | p95 ms | fps (median) | Speedup |
|--------------------------------------|-----------|---------|--------|--------------|---------|
| 1.3.0 (datanavigator+dustrack)       | 141.6     | 141.7   | 144.1  | 7.1          | 1.00x   |
| 1.4.0-qt (both)                      | 128.6     | 128.8   | 130.8  | 7.8          | 1.10x   |
| 1.4.0-qt + cache_quick_wins          | 93.8      | 93.8    | 95.9   | 10.7         | **1.51x** |
| 1.5.0-fast-render Tier 2             | 36.0      | 36.0    | 37.2   | 27.8         | **3.94x** |
| 1.4.0rc1 / 1.1.0rc1 (rerun 2026-05-19, pia02 vid) | 38.1 | 38.7 | 43.2 | 26.2 | 3.71x |
| **1.4.0rc2 / 1.1.0rc2** (2026-05-19, pia02 vid)   | **46.7** | **47.1** | **50.5** | **21.4** | **3.03x** |

The bottom two rows were measured on `pia02_s001_007_LFA2.mp4` —
`g.video_list[0]` is now a pia02 video on this DLC project (which
grew to 120 videos). The rc1 rerun (38.1 ms) reproduces the published
1.5.0-fast-render Tier 2 baseline (36.0 ms) within ~5%, so the
videos are perf-equivalent and the rc1↔rc2 delta is directly
comparable.

**rc2 regression vs rc1, accepted as the cost of new features.**
The +8.6 ms / +22% jump from rc1 to rc2 is the price of the rc2
UX surface (workflow-grouped sidebar, EnhanceWidget with two sliders
+ None/Auto, ConfirmOverlay, save-on-close guard, layer-lifecycle
buttons, statevars promoted from QLabel overlay to full Qt widget
with dropdown/toggle controls). Probe 11 ran on both branches with
`fast_render=False` to isolate the dnav-side segments: cache-keyed
segments (`annotation_visibility` 0.28 → 0.33 ms, `frame_marker`
0.02 → 0.02 ms) confirm the `_TrackedFrameDict` mutation guard is
NOT a contributor. The cost lives in **`update_assets`** (1.38 → 3.01
ms; sidebar grew from a flat list to five workflow groups + the new
EnhanceWidget), **`statevars_display`** (0.20 → 0.51 ms; Qt
widget replaces QLabel overlay), and the **Qt raster drain on the
fast_render path** (the bigger sidebar + statevars widget cost extra
paint events drained in `process_events`). Tradeoff judged worth
it 2026-05-19 — rc2 ships at ~21 fps which is still well above the
interactive threshold and the UX value bought is high.

### How the cache_quick_wins layer fits in

Probe 11 (`tests/qt_learning/11_profile_dustrack_update.py`) broke
the 128.6 ms 1.4.0-qt baseline into per-segment cost and surfaced two
items inside the update() body — independent of canvas raster — that
were doing work proportional to the dataset, not the frame:

- `VideoPointAnnotator.update_frame_marker` was rebuilding
  `np.hstack([ann.to_trace(label).T for ann in ...])` and recomputing
  `nanmin` / `nanmax` per frame to set trace ylim. The output only
  changes when annotations / label / frames_of_interest change, never
  when only `_current_idx` moves. Cost: ~18.7 ms / frame on
  interosseous_pn24-x (36,715 frames × N labels).
- `VideoAnnotation.update_display_trace` was calling `to_trace(label)`
  and `set_ydata(...)` for every label per frame. Same observation:
  the trace contents are a pure function of the annotation data, not
  the frame index. Cost: ~15.8 ms / frame.

Fix: bump a `_revision` counter on every `VideoAnnotation.data`
mutation (`add` / `remove` / `add_at_frame` / `add_label` /
`sort_labels` / `sort_data` / `clip_*` / etc.) and cache both code
paths on it. Per-frame work for these two segments dropped to ~0.3 ms
combined. No API change; both methods accept the same arguments and
produce the same visual output.

Full probe-11 segment breakdown, 105 frames on the interosseous
video (medians, ms). Segments in call order inside
`VideoPointAnnotator.update()` (+ `VideoBrowser.update()` body
inlined for finer attribution); `process_events` is timed
separately outside the update body and is the actual canvas
rasterization (what blit would target).

| segment                  | 1.4.0-qt | + cache | delta   | what it does |
|--------------------------|---------:|--------:|--------:|--------------|
| annotation_visibility    |    15.84 |    0.30 |  -15.54 | scatter `set_offsets` + per-label trace `set_ydata` (cached: only on annotation revision change) |
| statevars_display        |     0.32 |    0.23 |   -0.09 | Qt overlay text push |
| frame_marker             |    18.71 |    0.02 |  -18.69 | `_frame_marker_{x,y}` `set_data` + FOI `set_data` + ylim recompute (cached: only on annotation revision change) |
| decode                   |     7.70 |    7.45 |   -0.25 | `self.data[idx].asnumpy()` — PyAV+TOC decode |
| image_process            |     0.95 |    1.00 |   +0.05 | DUSTrack `enhance_ultrasound_image` (CLAHE + gamma + brightness; see probe 10) |
| imshow_set_data          |     0.80 |    0.77 |   -0.03 | `self._im.set_data(processed)` |
| title                    |     0.17 |    0.16 |   -0.01 | `self._ax.set_title(self.titlefunc(self))` |
| update_assets            |     1.47 |    1.49 |   +0.02 | buttons / memslots / events display push |
| plt_draw                 |     0.00 |    0.00 |    0.00 | `plt.draw()` × 2 (scheduling only, no synchronous raster) |
| **subtotal (update body)** | **45.97** | **11.43** | **-34.54** | sum of the above |
| process_events (raster)  |    83.02 |   82.39 |   -0.63 | `app.processEvents()` drains Qt's paint queue (the actual rasterization) |
| **total (update + raster)** | **129.17** | **94.28** | **-34.89** | per-frame budget |

After cache_quick_wins, process raster is **87% of total** — the
remaining big-win item lives behind it. Probe 12
(blit-feasibility, [discussed in dnav's BENCHMARKING.md](../datanavigator/BENCHMARKING.md))
showed blit doesn't help on QtAgg with this figure size; the
architectural fix was the 1.5.0 fast_render Tier 2 (below).

## Steady-state per-frame -- 1.5.0 fast_render (Tier 2 Qt-native)

The architectural change probe 13 surveyed was implemented in 1.5.0
as an opt-in second tier (`VideoBrowser(..., fast_render=True)`,
threaded through `VideoPointAnnotator` and `DUSTrack`). Tier 1
(`fast_render=False`, default) is unchanged. The Tier 2 architecture
itself is documented in
[`datanavigator/BENCHMARKING.md`](../datanavigator/BENCHMARKING.md)
(VideoBrowser is dnav-side); this section captures the production
delta on the DUSTrack workflow.

### Probe 14 result

`tests/qt_learning/14_benchmark_fast_render.py` measured the same
real DUSTrack UI on the same `interosseous_pn24-x` video used by
probes 09/11/12/13:

| segment / total        | 1.4.0-qt + cache | 1.5.0 fast_render | delta |
|------------------------|-----------------:|------------------:|------:|
| update body            |            11.43 |             11.42 |  ~0    |
| process_events (raster + upload) |  82.39 |             24.42 | -57.97 |
| **total (median)**     |        **93.80** |         **35.97** | -57.83 |
| **fps (median)**       |         **10.7** |          **27.8** | +17.1  |

**Speedup: 2.6x over 1.4.0-qt + cache_quick_wins, 3.94x over
1.3.0 baseline.**

The plan's aspirational threshold was 25 ms / 40 fps (probe 13's
~17-22 ms prediction); we landed at 36 ms / 28 fps. The gap is in
process_events (24 ms vs predicted 6-12 ms). Diagnostic probes
during 14 development surfaced two contributors:

- **`constrained_layout=True` re-runs on every draw**: cost ~25 ms
  on a trace-only canvas where the layout solver runs over many
  Line2D objects. Tier 2 explicitly uses `constrained_layout=False`
  with a manual `subplots_adjust` to bypass this; without that
  change, fast_render measured 60-70 ms (1.4x speedup -- not the
  ship-worthy claim).
- **`canvas.draw()` is ~15 ms** even on a 1200x100 px trace canvas
  with mostly invisible Line2Ds. Hidden artists still pay some
  Agg-render overhead; the trace canvas keeps ~30 lines per axis
  (one per (label, annotation) pair), of which only ~3 are visible
  in any given frame. Reducing the line count would help; this is
  scoped for a future iteration.

## Cold-open -- summary

Wall clock for `DLCProject(config) -> g.annotate(video_index=…)`,
measured by [`24_benchmark_cold_open.py`](tests/qt_learning/24_benchmark_cold_open.py)
on `interosseous_pn24-x`, video 0 (`pia02_s001_007_LFA2.mp4`,
36,715 frames, 1 JSON annotation + 2 DLC traces, PyAV TOC cached).
env: `dlc3rc14`.

| Branch / fix                                              | `g.annotate()` | Cumulative speedup |
|-----------------------------------------------------------|----------------|--------------------|
| 1.2.0a1 baseline (pre-vectorise)                          | **7.95 s**     | 1.00x              |
| 1.2.0a2 + `_dlc_trace_to_annotation_dict` vectorise       | **4.62 s**     | 1.72x              |
| 1.2.0a2 + shared-VideoReader across annotation layers     | **2.96 s**     | **2.69x**          |

The 1.2.0a2 cold-open work is two independent wins folded together:

1. **`VideoAnnotation._dlc_trace_to_annotation_dict` vectorised**
   (`dustrack/pointtracking.py:1895`). One column-slice +
   `.to_numpy()` per label + a NaN-row mask + a dict comprehension,
   replacing the per-frame `.loc[frame]` loop that fired ~73 k pandas
   cross-section calls per DLC trace on a 36 715-frame video.
   Semantics unchanged: skip rows where both x and y are NaN,
   otherwise record `[x, y]` for that frame; frame keys stay as
   Python ints so downstream `frame in dict` lookups keep matching.
   - `g.annotate()`: 7.95 s → 4.62 s (−3.33 s, **1.72×**).
   - `add_annotation_layers` segment: 6.29 s → 2.82 s.
   - Isolated `_dlc_trace_to_annotation_dict` smoke (36 715 × 2
     labels, in `tests/test_dlc_trace_vectorise.py`): **1 214 ms →
     26 ms (46×)**.
   - pandas `.loc` / `xs` calls per cold-open: 146 948 → 0.

2. **Annotation layers share the browser's single open
   `VideoReader`** (`dustrack/pointtracking.py:1668`, `:264`).
   `VideoAnnotation.__init__` accepts a `video=` kwarg; when
   supplied (the cold-open path), the layer reuses the caller's
   already-open reader instead of constructing a fresh
   `utils.Video(vname)`. `_DUSTrackBase.add_annotation_layers`
   threads `video=self.data` (the browser's reader) into every
   `annotations.add(...)` call.

   Pre-fix, each layer paid 3 `av.container.core.open` calls (one
   for `PyAVReaderIndexed`'s metadata probe, one for the persistent
   `_load_fresh_file` decoder, and one for
   `VideoReader._probe_avg_fps`) plus an OpenCV `is_video` probe —
   once per layer. The pia02 video 0 cold-open had 6 annotation
   layers + the browser, so 7 readers × 3 = 21 `av.open` calls on a
   network drive, where each open on `M:` costs ~80 ms even with
   the TOC cached.

   - `av.container.core.open` calls per `g.annotate()`: **21 → 3**.
   - `g.annotate()`: 4.62 s → 2.96 s (−1.66 s, **1.56×**).
   - `add_annotation_layers` segment: 2.82 s → 1.30 s.

   Surfaces a dnav-side prerequisite (1.5.0a2): `VideoReader` now
   exposes `fname` and `name` attributes on the base class so the
   shared reader satisfies VideoAnnotation's
   `self.video.fname` / `.name` access patterns without constructing
   the `utils.Video` subclass.

### Cold-open -- raw segment breakdown (video 0, post-fix, 2026-05-20)

```
DLCProject(config)                         169.98 ms    3.4%
scan video layer counts                    182.53 ms    3.6%
  VideoBrowser.__init__                    239.26 ms    4.8%
  add_annotation_layers                   1297.93 ms   25.8%
  statevariables.show                        4.24 ms    0.1%
  add_events                                 0.63 ms    0.0%
  set_key_bindings                          13.00 ms    0.3%
  _add_enhance_widget                        0.35 ms    0.0%
  _install_close_guard                       0.02 ms    0.0%
ret.update() (final)                        37.79 ms    0.8%
_normalize_dlc_layer_display                 0.21 ms    0.0%
_restructure_annotation_order                0.01 ms    0.0%
ret.update() (final)                        20.69 ms    0.4%
g.annotate(...) TOTAL                     2955.51 ms
first paint drain (processEvents x20)      106.20 ms
```

### Cold-open -- known follow-ons (not chased)

- The remaining 3 `av.open` calls per cold-open come from the one
  surviving `VideoReader` instantiation. Collapsing to 1 would need
  either a vendored-pims edit or a TOC-sidecar extension caching
  stream-geometry + avg_fps; ~30 ms additional win on `M:`. Not
  worth the vendored-code modification on its own.
- First-time PyAV TOC build on `M:` is ~37 s on a never-opened
  video, cached on disk after. Network-drive first-touch cost, not
  a code bug; pre-building TOCs server-side or warming them on
  project open would help.

## Post-processing throughput -- summary

Wall-clock + memory for the `Reduce jitter` GUI path
(`process_with_lk` → `lk_moving_average_filter`, parallel,
`save_raw=False` GUI default), measured on the pia02 fixture at
`S:\_corpus\dustrack\pia02_s001_011_RFA2_min1_15s.mp4` (1111 frames,
706×558, 73.94 fps, 1 label, 1075 windows at the default 0.5 s =
37-frame window).

env: `dlc3rc14`.

### Reduce jitter -- real-video end-to-end

| Branch / fix                                          | Wall time | Cumulative speedup |
|-------------------------------------------------------|-----------|--------------------|
| 1.2.0a1 baseline (pre-perf-pass, commit `1fc10c6`)    | **20.28 s** | 1.00×            |
| 1.2.0a2 LK perf pass (commit `6d3bb03`, save_raw=True) | **16.01 s** | **1.27×**        |

The 1.2.0a2 post-processing perf pass is three independent wins
folded together; numbers below from the example-video bench
([`25_benchmark_lk_rstc.py`](tests/qt_learning/25_benchmark_lk_rstc.py),
720p, 16-frame window, mode-minor interleaved, demuxer state
normalised between benches to avoid one bench's exit state poisoning
the next).

| Micro-bench                                       | base      | final    | speedup    | parity     |
|---------------------------------------------------|----------:|---------:|-----------:|------------|
| `lk_rstc_full` (forward + reverse, 16 frames)     | 1343 ms   | 247 ms   | **5.44×**  | bit-exact  |
| `lk_full` (single direction, 16 frames + decode)  | 217 ms    | 213 ms   | unchanged  | bit-exact  |
| `lk2_pair` (15 LK calls, pre-decoded frames)      | 34 ms     | 33 ms    | unchanged  | bit-exact  |
| `movavg` parallel (286 windows × 16 frames)       | 6.43 s    | 5.79 s   | 1.11×      | bit-exact  |
| `movavg` sequential (same shape)                  | 21.9 s    | 21.8 s   | unchanged  | bit-exact  |

1. **`lucas_kanade_rstc` decodes each frame once across forward +
   reverse passes** (`dustrack/opticalflow.py:158`). Pre-fix the
   function called `lucas_kanade` twice; the reverse direction's
   PyAV+TOC backward seeks (~60 ms / frame surcharge on 720p — see
   [[videoreader-demuxer-state-bench-artifact]]) dominated. Now
   decodes the window once, shares the frame list via `frames[::-1]`,
   delegates to the canonical `_lk_track_frames` helper.
   - `lk_rstc_full`: **1343 ms → 247 ms (5.44×)**, bit-exact.
   - Drives the reduce_jitter wall-clock improvement above — the
     same code path runs per window inside the moving-average filter.

2. **`cv.setNumThreads(1)` scoped via try/finally around the
   parallel ThreadPoolExecutor block** (`dustrack/postprocess.py:447`).
   cv2 defaults to `cpu_count` internal threads on Windows
   (Concurrency framework). With a `cpu_count`-sized Python worker
   pool every cv call inside every worker spawns up to `cpu_count`
   more internal threads — `cpu_count²` thread-slots fighting for
   `cpu_count` cores. Sweep on the 24-core dev box, 300-frame bench,
   `save_raw=False`:

   | cv_threads | workers | median (s) |
   |-----------:|--------:|-----------:|
   | 24 (default) | 28 (default) | 6.49 |
   | 1 | 8 | 5.77 |
   | 1 | 24 | 5.73 |
   | 2 | 12 | 5.76 |
   | 8 | 4 | 6.93 |

   Final: `cv=1` + executor default. ~12% faster, parity bit-exact.
   `cv.setNumThreads` is global; restoring it in `finally` keeps cv
   defaults intact for any concurrent caller.

3. **GUI `Reduce jitter` defaults to `save_raw=False`**
   (`dustrack/dlcinterface.py:2775`). `process_with_lk` does
   `kwargs.setdefault("save_raw", False)` so the button path skips
   the per-window `.pkl` sidecar. Direct API callers (wobble +
   gaitmusic `.rawlk` consumers, see
   [[lkmovavg-pkl-consumers]]) still default to `save_raw=True`
   because they call `lk_moving_average_filter` themselves. Trade:
   peak Python memory for the accumulator drops from `W·N·L·16` to
   `N·L·20` bytes — **35 MB → 2.9 MB on the pia02 W=15, L=4,
   N=36 715 shape** (~12×). Wall time on the example bench is
   essentially unchanged (post-processing is <0.3 % of the LK call
   budget; the win is the .pkl I/O round-trip on real videos and the
   accumulator allocation, not CPU). FP summation order differs
   between modes, so `save_raw=False` output is numerically close
   (`max|delta| = 2.27e-13` on the 300-frame example bench) but not
   bit-exact vs `save_raw=True`.

Bundled bug-fix: **`postprocess.gray` switched to `COLOR_RGB2GRAY`**
(was `COLOR_BGR2GRAY` on RGB input from dnav `VideoReader`, swapping
the R/B coefficients in BT.601 luminance). Unified with
`dustrack.opticalflow._gray_rgb`. Impact on real ultrasound:
`max|delta| = 0.02 px`, `p99 = 0.005 px` (R = G = B on grayscale
source, so the conversion swap was a no-op). On a generic color
video (the dnav example) the LK gradient field genuinely differs:
`max|delta| ≈ 30 px`, `p99 ≈ 11 px` — visible in the parity diff
against the pre-fix baseline but irrelevant for clinical ultrasound.

### Post-processing -- the bandwidth ceiling (why we stopped)

After the three wins above the `movavg` parallel path plateaus at
~5.7 s on the 300-frame example bench, with **3.78× speedup of
sequential on 24 cores (16 % efficiency)**. The GIL-breaking probe
at [`_movavg_gil_probe.py`](tests/qt_learning/_movavg_gil_probe.py)
confirmed the cap is **not** the GIL:

| path | rep 0 | rep 1 | rep 2 | rep 3 | rep 4 |
|---|---:|---:|---:|---:|---:|
| ThreadPool reference | 5.53 s | — | — | — | — |
| ProcessPool + shared-memory frames (persistent) | 21.42 s (spawn) | 5.81 s | 5.81 s | 5.72 s | 5.59 s |

Process pools with separate heaps hit the same ~5.7 s plateau as
the ThreadPool after spawn amortisation. The cap is **memory
bandwidth from cv2's per-call pyramid build/teardown**: each
`calcOpticalFlowPyrLK` call internally allocates ~12 MB of pyramid
(two pyramids × ~6 MB at 720p, 3 levels, int16 gradients), discarded
at end of call. The bench fires 8008 LK calls → ~96 GB allocation
traffic → ~17 GB/s, at the DDR4 bandwidth ceiling on this box.

Two approaches tried, **both ruled out** (memory:
[[lk-perf-dead-ends]]):

- **mimalloc-redirect** via the Microsoft `minject.exe` workflow ran
  ~5–7 % **slower** than default malloc. Allocator contention isn't
  the bottleneck; swapping the allocator doesn't reduce the memory
  traffic itself.
- **Single-level Numba LK** (`@njit(nogil=True)` with pre-computed
  Scharr gradients) ran **1.77× faster** than cv2, but parity
  diverged (p99 = 9.9 px, max = 22 px). Even sub-pixel-per-frame
  motion regimes still hit ~9 px p99 — the gap is algorithmic
  (inverse-compositional vs cv's forward-additive, Scharr vs Sobel,
  no pyramid), not closable without ~500–800 more lines matching
  cv exactly which then drops the speedup to ~1.3–1.5×.

### Post-processing -- known follow-ons (not chased)

- **Pyramid reuse via the cv2 Python binding is blocked**.
  `cv.buildOpticalFlowPyramid` works, but `cv.calcOpticalFlowPyrLK`
  in opencv-python 4.11 doesn't accept the pre-built pyramid as
  `prevImg` / `nextImg` (only the single-Mat overload is exposed
  to Python; the `vector<Mat>` C++ overload demands `Ptr<UMat>`).
  Re-evaluate if a future opencv-python release surfaces it.
  Memory: [[cv2-pyrlk-pyramid-binding]].
- **GPU LK** via `cv.cuda.SparsePyrLKOpticalFlow` would bypass the
  bandwidth ceiling but needs a CUDA-built opencv wheel (cudawarped
  or source build, ~1–2 hr first-time setup). Spec'd in
  `specs/dustrack.md → Roadmap → Later`. The per-call overhead on
  sparse 2-points-per-call workloads only buys ~3× — not the 20–50×
  dense flow gets — so deferred until a batch reprocessing campaign
  or a second GPU CV path (NVDEC decode, dense flow) makes the
  install cost amortise.

## Methodology

- **Steady-state probes** (09, 14): each iteration sets
  `_current_idx`, calls `browser.update()`, calls
  `QApplication.processEvents()` to drain Qt's paint queue. Time
  the pair. `processEvents()` is essential because matplotlib's
  `canvas.draw_idle()` only *schedules* a repaint; timing
  `update()` alone misses the actual rasterization that happens on
  Qt's next idle tick. First 10-15 iterations discarded as warmup.
- **Cold-open probe** (24): wraps each `g.annotate()` segment via
  monkey-patched entry points (`_DUSTrackBase.add_annotation_layers`,
  `VideoBrowser.__init__`, `StateVariables.show`, etc.) under
  `time.perf_counter`, with a `cProfile` pass over the whole
  `annotate()` call. Reports top-N cumulative + self-time entries.
- **Post-processing probes** (25, `_movavg_gil_probe`,
  `_reduce_jitter_real_bench`): mode-minor interleaved (rep r runs
  every micro-bench in turn before rep r+1) per memo
  [[thermal-confounding-mode-major-iteration]]; demuxer state
  normalised between benches by reading frame 0 untimed so PyAV+TOC
  backward-seek artifacts don't poison the next bench's timing
  ([[videoreader-demuxer-state-bench-artifact]]). Reference output
  arrays are frozen per-step and parity-diffed across optimisation
  steps with `np.allclose(atol=1e-6, rtol=0)` (or relaxed for the
  numerically-equivalent paths e.g. `save_raw=False`).
- Known-cosmetic Qt warnings (`QFontDatabase: Cannot find font
  directory`, offscreen plugin `does not support raise()` /
  `propagateSizeHints()`) are filtered via a Qt message handler so
  benchmark output is clean.

## How to run

```powershell
# Steady-state per-frame on the DUSTrack UI:
C:\Users\praneeth\anaconda3\envs\dlc3rc14\python.exe `
    C:\dev\DUSTrack\tests\qt_learning\09_benchmark_dustrack.py

# Steady-state with fast_render Tier 2:
C:\Users\praneeth\anaconda3\envs\dlc3rc14\python.exe `
    C:\dev\DUSTrack\tests\qt_learning\14_benchmark_fast_render.py

# Cold-open on video 0 (or any --video-index):
C:\Users\praneeth\anaconda3\envs\dlc3rc14\python.exe `
    C:\dev\DUSTrack\tests\qt_learning\24_benchmark_cold_open.py `
    --video-index 0 --top 20

# Cold-open: scan layer counts only (worst-case video selection):
C:\Users\praneeth\anaconda3\envs\dlc3rc14\python.exe `
    C:\dev\DUSTrack\tests\qt_learning\24_benchmark_cold_open.py `
    --max-layers --top 20

# Post-processing throughput on the example video (LK micro+macro):
C:\Users\praneeth\anaconda3\envs\dlc3rc14\python.exe `
    C:\dev\DUSTrack\tests\qt_learning\25_benchmark_lk_rstc.py `
    --step myrun [--compare-to base] [--no-movavg-save-raw]

# Reduce jitter end-to-end on the pia02 real-shape fixture:
C:\Users\praneeth\anaconda3\envs\dlc3rc14\python.exe `
    C:\dev\DUSTrack\tests\qt_learning\_reduce_jitter_real_bench.py `
    --label myrun --save-raw both --n-reps 3

# GIL / ProcessPool feasibility probe (one-off, dead-end finding):
C:\Users\praneeth\anaconda3\envs\dlc3rc14\python.exe `
    C:\dev\DUSTrack\tests\qt_learning\_movavg_gil_probe.py
```

The steady-state probes accept `--record LABEL` to append a
timestamped block under *Raw results* below.

## Hardware / environment

- Machine: Windows 11, the development workstation
- Conda env: `dlc3rc14` (Python 3.10, DLC 3.0.0rc14, PySide6 6.4.2)
- matplotlib: 3.8.4 with QtAgg backend
- Video storage: `M:` (network drive); first-touch TOC build is
  ~37 s and not benchmarked further

## Raw results

<!-- Steady-state probes append timestamped blocks below when run with
     --record LABEL. Newest results land at the bottom. -->

### DUSTrack UI -- 1.4.0-qt -- 2026-05-17 10:51:35

- datanavigator: `1.4.0-qt@fc1b6f3-dirty` source `C:\dev\datanavigator\datanavigator\__init__.py`
- dustrack: `1.4.0-qt@9ba5e7f` source `C:\dev\DUSTrack\dustrack\__init__.py`
- backend: QtAgg, qt_api=pyside6, qt_plat=default
- N=90 (10 warmup discarded)

| min | median | mean | p95 | p99 | max | fps (median) |
|---|---|---|---|---|---|---|
| 126.82 | 128.59 | 128.80 | 130.77 | 134.75 | 134.75 | 7.8 |

### DUSTrack UI -- 1.3.0 baseline (master) -- 2026-05-17 10:52:37

- datanavigator: `master@b71761d-dirty` source `C:\dev\datanavigator\datanavigator\__init__.py`
- dustrack: `main@dc0cff3` source `C:\dev\DUSTrack\dustrack\__init__.py`
- backend: QtAgg, qt_api=pyside6, qt_plat=default
- N=90 (10 warmup discarded)

| min | median | mean | p95 | p99 | max | fps (median) |
|---|---|---|---|---|---|---|
| 139.34 | 141.28 | 141.51 | 144.03 | 147.08 | 147.08 | 7.1 |

### DUSTrack UI -- 1.3.0 baseline (master) run2 -- 2026-05-17 10:53:23

- datanavigator: `master@b71761d-dirty` source `C:\dev\datanavigator\datanavigator\__init__.py`
- dustrack: `main@dc0cff3` source `C:\dev\DUSTrack\dustrack\__init__.py`
- backend: QtAgg, qt_api=pyside6, qt_plat=default
- N=90 (10 warmup discarded)

| min | median | mean | p95 | p99 | max | fps (median) |
|---|---|---|---|---|---|---|
| 140.15 | 141.83 | 141.98 | 144.10 | 146.46 | 146.46 | 7.1 |

### DUSTrack UI -- 1.4.0-qt + cache_quick_wins -- 2026-05-17 20:10:47

- datanavigator: `1.4.0-qt@6c052ef-dirty` source `C:\dev\datanavigator\datanavigator\__init__.py`
- dustrack: `1.4.0-qt@9ba5e7f` source `C:\dev\DUSTrack\dustrack\__init__.py`
- backend: QtAgg, qt_api=pyside6, qt_plat=default
- N=85 (15 warmup discarded)

| min | median | mean | p95 | p99 | max | fps (median) |
|---|---|---|---|---|---|---|
| 91.72 | 93.77 | 93.80 | 95.85 | 99.24 | 99.24 | 10.7 |

### DUSTrack UI -- 1.5.0-fast-render Tier 2 -- 2026-05-17 22:39:30

- datanavigator: `1.5.0-fast-render@52f5da1-dirty` source `C:\dev\datanavigator\datanavigator\__init__.py`
- dustrack: `1.5.0-fast-render@4f271b4-dirty` source `C:\dev\DUSTrack\dustrack\__init__.py`
- backend: QtAgg, qt_api=pyside6, qt_plat=default
- fast_render: True
- N=185 (15 warmup discarded)

| min | median | mean | p95 | p99 | max | fps (median) |
|---|---|---|---|---|---|---|
| 66.23 | 68.32 | 68.63 | 70.88 | 74.03 | 75.15 | 14.6 |

### DUSTrack UI -- 1.5.0-fast-render Tier 2 (trace-only canvas) -- 2026-05-17 22:43:19

- datanavigator: `1.5.0-fast-render@52f5da1-dirty` source `C:\dev\datanavigator\datanavigator\__init__.py`
- dustrack: `1.5.0-fast-render@4f271b4-dirty` source `C:\dev\DUSTrack\dustrack\__init__.py`
- backend: QtAgg, qt_api=pyside6, qt_plat=default
- fast_render: True
- N=185 (15 warmup discarded)

| min | median | mean | p95 | p99 | max | fps (median) |
|---|---|---|---|---|---|---|
| 64.81 | 71.05 | 71.66 | 81.18 | 84.92 | 87.60 | 14.1 |

### DUSTrack UI -- 1.5.0-fast-render Tier 2 (final) -- 2026-05-17 22:51:32

- datanavigator: `1.5.0-fast-render@52f5da1-dirty` source `C:\dev\datanavigator\datanavigator\__init__.py`
- dustrack: `1.5.0-fast-render@4f271b4-dirty` source `C:\dev\DUSTrack\dustrack\__init__.py`
- backend: QtAgg, qt_api=pyside6, qt_plat=default
- fast_render: True
- N=185 (15 warmup discarded)

| min | median | mean | p95 | p99 | max | fps (median) |
|---|---|---|---|---|---|---|
| 34.43 | 35.97 | 36.03 | 37.21 | 37.97 | 39.82 | 27.8 |

### DUSTrack UI -- 1.4.0rc1 / 1.1.0rc1 rerun -- 2026-05-19

- datanavigator: `HEAD@1e8fa51` (tag `v1.4.0rc1`) source `C:\dev\datanavigator\datanavigator\__init__.py`
- dustrack: `HEAD@29775c0` (tag `v1.1.0rc1`) source `C:\dev\DUSTrack\dustrack\__init__.py`
- backend: QtAgg, qt_api=pyside6, qt_plat=default
- fast_render: True (default)
- video: `pia02_s001_007_LFA2.mp4` (g.video_list[0], 36,715 frames)
- N=185 (15 warmup discarded)
- purpose: rerun of the rc1 baseline on the same env / video / harness
  as the rc2 measurement below, so the rc1↔rc2 delta is apples-to-apples.

| min | median | mean | p95 | p99 | max | fps (median) |
|---|---|---|---|---|---|---|
| 35.85 | 38.13 | 38.68 | 43.15 | 48.12 | 49.16 | 26.2 |

### DUSTrack UI -- 1.4.0rc2 / 1.1.0rc2 -- 2026-05-19

- datanavigator: `1.4.0rc2@c5bcbd0` source `C:\dev\datanavigator\datanavigator\__init__.py`
- dustrack: `main@e8e9653-dirty` (a tiny dlcinterface.py edit committed
  shortly after as `a03877c`; not material to the per-frame budget)
  source `C:\dev\DUSTrack\dustrack\__init__.py`
- backend: QtAgg, qt_api=pyside6, qt_plat=default
- fast_render: True (default)
- video: `pia02_s001_007_LFA2.mp4` (g.video_list[0], 36,715 frames)
- N=185 (15 warmup discarded)
- delta vs rc1 rerun above: **+8.58 ms median (+22.5%), -4.8 fps median (-18.3%)**.
  Root-causing (see summary table commentary above + probe 11
  segment diff below) attributes the regression to the rc2 UX
  additions (workflow-grouped sidebar, statevars Qt widget,
  EnhanceWidget, layer-lifecycle buttons) and **not** to the
  `_TrackedFrameDict` mutation guard. Accepted 2026-05-19 as the
  cost of the new UX surface; gates the 1.4.0 / 1.1.0 final cut
  on no *further* regression beyond this point.

| min | median | mean | p95 | p99 | max | fps (median) |
|---|---|---|---|---|---|---|
| 44.69 | 46.71 | 47.09 | 50.50 | 53.53 | 61.70 | 21.4 |

### Probe 11 segment diff -- rc1 vs rc2 -- 2026-05-19

Ran `tests/qt_learning/11_profile_dustrack_update.py`'s logic with
`fast_render=False` (the canonical probe was written before
fast_render became default-on and crashes on `_im = None` when it's
on; the dnav-side segments measured are identical with or without
fast_render — fast_render only swaps the image pane). Same env,
same video, 185 measured frames each.

| Segment                  | rc1 ms | rc2 ms | Δ ms   | Notes |
|--------------------------|-------:|-------:|-------:|-------|
| annotation_visibility    |  0.28  |  0.33  |  +0.05 | flat — mutation guard NOT a contributor |
| statevars_display        |  0.20  |  0.51  |  +0.31 | new Qt statevars widget replaces QLabel overlay |
| frame_marker             |  0.02  |  0.02  |   0.00 | flat — cache invariant holds |
| decode                   |  7.69  |  7.74  |  +0.05 | flat — PyAV decode unchanged |
| image_process            |  0.99  |  0.00  |  -0.99 | rc2 default opens raw (clahe=1.0/gamma=1.0); bypass branch fires |
| imshow_set_data          |  0.77  |  0.90  |  +0.13 | small |
| title                    |  0.13  |  0.17  |  +0.04 | small |
| update_assets            |  1.38  |  3.01  |  +1.63 | sidebar grew: workflow groups + EnhanceWidget + new buttons |
| plt_draw                 |  0.00  |  0.00  |   0.00 | flat |
| process_events (raster)  | 81.69  | 83.07  |  +1.38 | small on mpl path; the bulk of fast_render-path regression lives here on the production path |
| **TOTAL (mpl path)**     | **93.40** | **95.90** | **+2.50** | mpl path: ~2.5 ms regression |
| (TOTAL fast_render path) | (38.13) | (46.71) | (+8.58) | for reference — production hot path |

The ~6 ms gap between the mpl-path regression (+2.5 ms) and the
fast_render-path regression (+8.6 ms) is most likely extra Qt paint
events drained in `process_events` on the fast_render path, caused
by the same expanded sidebar + statevars widget that show up in
`update_assets` and `statevars_display` on the mpl path. Could be
recovered by caching the per-frame asset push or auditing
paint-event frequency for the new widgets — deferred to a future
patch if the workflow ever runs into the headroom.

### Cold-open -- video 0 -- 2026-05-20

- datanavigator: `1.5.0@201f9d0` (VideoReader fname/name lift)
- dustrack: `1.2.0a2-cold-open@dd2b4aa` (shared-VideoReader fix)
- env: `dlc3rc14`
- video: `pia02_s001_007_LFA2.mp4` (g.video_list[0], 36 715 frames,
  PyAV TOC cached)
- 3 annotation layers (1 JSON + 2 DLC traces)

| metric | pre-vectorise | post-vectorise | post-shared-VR |
|---|---|---|---|
| `g.annotate()` total | 7950 ms | 4620 ms | **2956 ms** |
| `add_annotation_layers` | 6290 ms | 2820 ms | **1298 ms** |
| `av.container.core.open` calls | 21 | 21 | **3** |
| cumulative speedup | 1.00x | 1.72x | **2.69x** |

### DUSTrack UI -- post-seek-fix-2026-05-20 -- 2026-05-20 15:44:02

- datanavigator: `1.2.0a2-cold-open@cfbc5a7` source `C:\dev\datanavigator\datanavigator\__init__.py`
- dustrack: `1.2.0a2-cold-open@cfbc5a7` source `C:\dev\DUSTrack\dustrack\__init__.py`
- backend: QtAgg, qt_api=pyside6, qt_plat=default
- fast_render: True
- N=185 (15 warmup discarded)

| min | median | mean | p95 | p99 | max | fps (median) |
|---|---|---|---|---|---|---|
| 38.64 | 40.03 | 40.17 | 41.25 | 43.61 | 46.12 | 25.0 |

### DUSTrack UI -- pre-seek-fix-isolated-2026-05-20 -- 2026-05-20 15:45:12

- datanavigator: `1.2.0a2-cold-open@cfbc5a7-dirty` source `C:\dev\datanavigator\datanavigator\__init__.py`
- dustrack: `1.2.0a2-cold-open@cfbc5a7-dirty` source `C:\dev\DUSTrack\dustrack\__init__.py`
- backend: QtAgg, qt_api=pyside6, qt_plat=default
- fast_render: True
- N=185 (15 warmup discarded)

| min | median | mean | p95 | p99 | max | fps (median) |
|---|---|---|---|---|---|---|
| 38.68 | 40.30 | 40.45 | 41.82 | 45.77 | 46.47 | 24.8 |
