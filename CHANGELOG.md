# Change Log
All notable changes to this project will be documented in this file.

## [1.2.0a2] - unreleased

Cold-open optimisation: two independent wins folded together — the
vectorised DLC-trace conversion (drops the per-frame `.loc[frame]`
pandas cross-section), and a single shared `VideoReader` across all
annotation layers (drops the per-layer `utils.Video(vname)` open).
Together they take `g.annotate()` on the pia02 `interosseous_pn24-x`
production benchmark from **7.95 s → 2.96 s** (−5.0 s, **2.69× faster**)
on video 0.

Earlier 1.2.0 scope items (dnav 1.5.0 adoption + DLC `.h5` reclaim)
came along with the 1.2.0a1 relocation. The originally-scheduled
`fast_traces=True` Qt-tier was explored on a throwaway branch
2026-05-20 and reverted after the benchmark showed an 8.81×
per-frame regression on the production video; the matplotlib trace
pane stays. See portfolio memo `feedback_qt_traces_benchmark_2026_05_20`.

### Changed
- **`postprocess.gray` switched to `COLOR_RGB2GRAY`** (was
  `COLOR_BGR2GRAY` on the same RGB input from dnav `VideoReader`).
  Bug-fix that unifies the grayscale convention with
  `dustrack.opticalflow._gray_rgb`. Both sides now compute
  BT.601 luminance ``Y = 0.299 R + 0.587 G + 0.114 B``; the pre-fix
  BGR2GRAY-on-RGB path swapped the R/B coefficients. Impact:
  - Real ultrasound: ``max|delta| = 0.02 px``,
    ``p99 = 0.005 px`` (RGB channels are equal on grayscale source,
    so the conversion swap is a no-op). Measured on the pia02
    1111-frame fixture.
  - Color video (the dnav example): ``max|delta| ≈ 30 px``,
    ``p99 ≈ 11 px``. Tracking diverges visibly because the LK
    gradient field is genuinely different.

  Existing ``_lkmovavg_*.json`` / ``_lkmovavg_*.pkl`` on disk reflect
  the old (incorrect) computation; subsequent runs will regenerate
  them with the correct luminance. For clinical ultrasound the
  difference is below visual perception.

- **GUI ``Reduce jitter`` defaults to ``save_raw=False``**
  (`dustrack/dlcinterface.py:2775`). `process_with_lk` now
  ``kwargs.setdefault("save_raw", False)`` so the button path
  skips the per-window ``.pkl`` sidecar. Callers who want the
  ``.pkl`` (wobble + gaitmusic ``.rawlk``) still get it by default
  because they go through the direct
  ``dustrack.postprocess.lk_moving_average_filter`` API (default
  there is still ``True``), or by passing ``save_raw=True``
  explicitly. Saves ~35 MB of allocation + ``dill.dump`` /
  ``dill.load`` round-trip per GUI ``Reduce jitter`` click on a
  real-shape video.

- **`lucas_kanade_rstc` decodes each frame once across forward + reverse
  passes** (`dustrack/opticalflow.py:158`). Pre-fix the function called
  `lucas_kanade` twice — one forward (`start → end`) and one reverse
  (`end → start`); each pass independently decoded + grayscaled every
  frame in the window. The reverse direction is much more expensive
  than the forward one on PyAV+TOC because PyAV has to seek back to
  the nearest keyframe and re-decode forward to each requested frame,
  so the reverse pass dominated wall time. Now we decode the window
  once, share the grayscale frame list between forward and reverse
  via a `frames[::-1]` view, and delegate to a single canonical LK
  loop (`_lk_track_frames`).

- **`_rewire_to_in_project_paths` migrates layers correctly when
  `DLCProject.path` is the working directory** (`dustrack/dlcinterface.py`).
  Pre-fix the helper read ``self._dlcproject.path`` as
  ``project_root``, but ``DLCProject.path`` stores whatever the
  caller passed in -- for a brand-new project that's the WORKING
  DIRECTORY (parent of the actual project dir), because
  ``deeplabcut.create_new_project`` creates the project at
  ``<working_dir>/<name>-<experimenter>-<date>/``. The migration
  check ``ann_path.relative_to(project_root)`` then returned a
  false-positive "yes, already inside" for any annotation file sitting
  next to the original video (the working dir IS a prefix of that
  path), which silently skipped the migration and stranded those
  layers at their original outside-project locations.

  Production impact: train pre-flight saved the cleaned annotations to
  the wrong path. The project's copy of the JSON stayed stale, and
  ``extract_frames`` -> ``labeled_data`` propagated the dropped
  incomplete frame straight into ``CollectedData_*.h5`` -- DLC
  training ingested the frame the user had asked to drop. User-reported
  bug today.

  Fix: derive ``project_root`` from the in-project video path
  (``videos_dir.parent``) so it always points at the actual project
  directory regardless of what was passed to ``DLCProject(path=...)``.
  One-line change in ``_rewire_to_in_project_paths`` plus a regression
  test (``test_outside_project_migrates_even_when_dlcproject_path_is_working_dir``)
  that pins the bug-trigger: ``DLCProject(path=<working_dir>)`` with
  an annotation file at top-level must still migrate.

- **Train pre-flight now flags incomplete frames on first-time training**
  (`dustrack/dlcinterface.py:_scan_unsaved_and_incomplete`). Pre-fix
  `_is_manual_annotation_layer` returned False whenever `ann.fname is
  None`, which is the natural state of a freshly-annotated layer that
  has never been saved (the first-time-training case). The scan
  silently skipped that layer, so a click on "Train DLC model" with
  an incomplete frame in the in-memory `iteration-0` layer went
  straight to training with no modal. Now inclusion is name-only via
  a new `_is_manual_layer_name` static predicate (excludes
  `dlccorr` / `buffer` / `dlc*` -- same exclusion semantics as
  before); the disk-diff portion is still guarded on `ann.fname`
  being set AND matching the `<video_stem>_annotations*.json`
  pattern, so layers without a disk file are scanned for
  incompleteness but not for stale-on-disk. `_apply_pre_flight_remediations`
  derives a canonical `ann.fname = <video_stem>_annotations_<layer_name>.json`
  before save when the layer was previously unsaved, so the Save &
  clean path works end-to-end. 10 new tests: 6 for `_is_manual_layer_name`,
  5 for `_scan_unsaved_and_incomplete` covering unsaved-incomplete,
  unsaved-clean, dlccorr-excluded, saved-incomplete (regression),
  and saved-with-disk-diff-only.

- **`postprocess.gray` / `opticalflow._gray_rgb` / `enhance_ultrasound_image`
  short-circuit on 2D input** to ride dnav 1.5.0a1's new
  `pix_fmt='gray'` auto-detect (`postprocess.py:gray`,
  `opticalflow.py:_gray_rgb`, `dlcinterface.py:enhance_ultrasound_image`).
  When dnav decodes a monochrome-encoded source directly as `(H, W)`
  gray, these helpers no-op the `cvtColor` step. `image_processor`
  inside `DUSTrack.__init__` coerces the per-frame display path back
  to 3-channel RGB so the image pane (matplotlib + Qt-native fast
  render) sees the same shape regardless of source. `export_video`
  (`pointtracking.py:_read_frame_rgb`) does the same channel-replicate
  for the encoded output. Combined with the dnav gray path, sequential
  `decode + gray()` on the S-corpus
  `pia02_s001_011_RFA2_min1_15s_mono.mp4` clip went **164 → 1886 fps**
  (11.5x). `_extract_frames_decord` (DLC labeled-data extraction)
  explicitly passes `pix_fmt='rgb24'` so DLC's ResNet-50 backbone
  still ingests 3-channel PNGs; the gray path benefit accrues to the
  LK / postprocess / annotation paths.

  Bench (`tests/qt_learning/25_benchmark_lk_rstc.py`, 720p, 16-frame
  window, demuxer state normalised to frame 0 between benches):
  - `lucas_kanade_rstc` (mode='full'): **1 343 ms → 247 ms (5.44×)**.
  - `lucas_kanade` (mode='full'): unchanged (single-direction decode,
    nothing to share).
  - `lucas_kanade_2` micro: unchanged (the helper is now its body).

  All four bench reference outputs (`lk2_pair`, `lk_full`,
  `lk_rstc_full`, `movavg`) are bit-exact against the pre-refactor
  arrays under `atol=1e-6, rtol=0`.

- **Per-pair LK loop centralised in `_lk_track_frames`**
  (`dustrack/opticalflow.py:70`). `lucas_kanade` (per-video) and
  `lucas_kanade_2` (frame-list, in `dustrack/postprocess.py`)
  delegated their per-pair loops here so the call lives in one place.
  Carries an `np.empty` allocation tweak (every slot is overwritten
  immediately) and the documented limitation that
  `cv.calcOpticalFlowPyrLK` in opencv-python 4.11 does not accept a
  pre-built pyramid list as `prevImg` / `nextImg` (the `vector<Mat>`
  C++ overload is not exposed to Python — it rejects both tuple and
  list of ndarrays, demanding `Ptr<UMat>`). Pyramid reuse across pairs
  is therefore not viable from Python; the docstring records the
  finding so future readers don't retry it.

- **`lk_moving_average_filter` parallel path pins
  ``cv.setNumThreads(1)`` via try/finally**
  (`dustrack/postprocess.py:447`). Before the tweak, every Python
  worker's ``cv.calcOpticalFlowPyrLK`` call spawned up to
  ``cpu_count`` OpenCV-internal threads on top of the
  ``cpu_count``-sized worker pool -- ``cpu_count**2`` thread-slots
  fighting for ``cpu_count`` cores. Sweep on a 24-core machine
  (300-frame bench, ``save_raw=False``):

  | cv_threads | workers | median (s) |
  |---|---|---|
  | 24 (default) | 28 (default) | 6.49 |
  | 1 | 8 | 5.77 |
  | 1 | 24 | 5.73 |
  | 2 | 12 | 5.76 |
  | 8 | 4 | 6.93 |

  Final landing: ``cv=1`` + executor default. ~12% faster, parity
  bit-exact. Scope: ``cv.setNumThreads(1)`` is global; restoring it
  in ``finally`` keeps cv defaults intact for any concurrent caller.

  GIL-breaking investigation (deferred): a sibling probe at
  ``tests/qt_learning/_movavg_gil_probe.py`` ran ``ProcessPoolExecutor``
  with frames in ``multiprocessing.shared_memory``. After amortising
  Windows worker-spawn cost across reps, the persistent process pool
  plateaus at **~5.7 s** -- identical to ``ThreadPool``. The GIL is
  **not** the bottleneck; the cap is likely OpenCV's heap allocator
  contending on the ~7 MB pyramid that
  ``cv.calcOpticalFlowPyrLK`` builds for each frame in each call
  (~117 GB of allocation traffic across the bench's 8008 LK calls).
  Going below 5.7 s would need either a custom allocator
  (mimalloc/jemalloc as the host's default malloc) or a custom LK
  that pools pyramid memory across calls -- both out of scope.

- **`lk_moving_average_filter` accepts `save_raw=False` to skip the
  per-window `.pkl` sidecar** (`dustrack/postprocess.py:248`). Default
  is `True` — preserves the existing `(W, N, L, 2)` `.pkl` that
  `pn-projects/wobble` and `gaitmusic` consume via their `.rawlk`
  property + `lk_gradients` velocity estimation, so direct API
  callers see no behaviour change. `save_raw=False` switches to a
  streaming `(N, L, 2)` sum and `(N, L)` count, divides at the end,
  and skips the `.pkl` write entirely. Trade-off: peak Python memory
  for the accumulator drops from `W*N*L*16` bytes to roughly
  `N*L*20` bytes — a ~10-12× reduction on typical configs (W=15,
  L=4, N=36 715: **35 MB → 2.9 MB**). Wall time is essentially
  unchanged (post-processing is <0.3% of the LK call budget on this
  bench video; the .pkl round-trip is a sub-second I/O on real
  videos). FP summation order between the two modes differs, so
  `save_raw=False` output is numerically close but not bit-exact
  vs `save_raw=True`: measured `max|delta| = 2.27e-13` on the
  300-frame example video, well below the float64 noise floor
  (image-coordinate scale ~O(100), so relative error ~1e-15).

  Cache short-circuits restructured so an existing `.pkl` is always
  honoured (it's cheaper to average a cached `.pkl` than to recompute
  even when `save_raw=False`), and the assert in
  `tests/qt_learning/25_benchmark_lk_rstc.py` pins the contract:
  `.pkl` present iff `save_raw=True`.

- **`lk_moving_average_filter` parallel path uses bounded-inflight
  submission with a single fused progress bar**
  (`dustrack/postprocess.py:376`). Pre-fix the executor pre-submitted
  all N futures up front then collected them in submission order
  under two separate tqdm bars ("Submitting jobs" → "Processing
  results"); on long videos the Submitting bar filled almost instantly
  and Processing sat at 0 for ages, making the Qt overlay phase label
  uninformative. Now the loop interleaves decode + submit + collection
  via `concurrent.futures.wait(FIRST_COMPLETED)` with at most
  `2 × pool_size` futures pending, and a single tqdm bar
  ("Processing tracking jobs") drives the overlay throughout.
  `_JITTER_PHASES` in `dustrack/dlcinterface.py:233` recognises the
  new label and keeps the old ones as fallbacks. Output bit-exact
  vs. the sequential path; parallel-vs-sequential timing on the
  300-frame example video: parallel **6.43 s** / sequential
  **21.62 s** (3.36× via threading, unchanged from pre-refactor).

- **RSTC sigmoid blend uses two ufunc-with-out calls instead of three
  implicit allocations** (`dustrack/opticalflow.py:226`,
  `dustrack/postprocess.py:209`). `np.flip(reverse_path, 0)` collapsed
  to a `[::-1]` view (no copy). Below the noise floor on small
  arrays; included for hygiene.

- **`VideoAnnotation._dlc_trace_to_annotation_dict` vectorised**
  (`dustrack/pointtracking.py:1895`). One column-slice +
  `.to_numpy()` per label + a NaN-row mask + a dict comprehension,
  replacing the per-frame `.loc[frame]` loop that fired ~73 k pandas
  cross-section calls per DLC trace on a 36 715-frame video.
  Semantics unchanged: skip rows where both x and y are NaN, otherwise
  record `[x, y]` for that frame; frame keys stay as Python ints so
  downstream `frame in dict` lookups keep matching.

  Real cold-open benchmark (`tests/qt_learning/24_benchmark_cold_open.py`
  on `interosseous_pn24-x`, video 0, PyAV TOC cached):
  - `g.annotate()` total: **7.95 s → 4.62 s** (−3.33 s, **1.72× faster**).
  - `add_annotation_layers` segment: 6.29 s → 2.82 s.
  - Isolated `_dlc_trace_to_annotation_dict` smoke (36 715 × 2 labels,
    in `tests/test_dlc_trace_vectorise.py`): **1 214 ms → 26 ms (46×)**.
  - pandas `.loc` / `xs` calls per cold-open: 146 948 → 0.

- **Annotation layers share the browser's single open `VideoReader`**
  (`dustrack/pointtracking.py:1668`, `:264`). `VideoAnnotation.__init__`
  now accepts a `video=` kwarg; when supplied (the cold-open path),
  it reuses the caller's already-open reader instead of constructing
  a fresh `utils.Video(vname)`. `_DUSTrackBase.add_annotation_layers`
  threads `video=self.data` (the browser's reader) into every
  `annotations.add(...)` call.

  Pre-fix, each layer paid 3 `av.container.core.open` calls (one for
  `PyAVReaderIndexed`'s metadata probe, one for the persistent
  `_load_fresh_file` decoder, and one for `VideoReader._probe_avg_fps`)
  plus an OpenCV `is_video` probe — once per layer. The pia02 video 0
  cold-open had 6 annotation layers + the browser, so 7 readers ×
  3 = 21 `av.open` calls on a network drive, where each open on `M:`
  costs ~80 ms even with the TOC cached.

  Real cold-open benchmark (same harness, video 0, PyAV TOC cached,
  starting from the post-vectorise baseline):
  - `g.annotate()` total: **4.62 s → 2.96 s** (−1.66 s, **1.56× faster**).
  - `add_annotation_layers` segment: 2.82 s → 1.30 s.
  - `av.container.core.open` calls per `g.annotate()`: **21 → 3**.
  - Combined with the vectorise: **7.95 s → 2.96 s, 2.69× cumulative**.

  Surfaces a small dnav-side prerequisite (1.5.0a2): `VideoReader` now
  exposes `fname` and `name` attributes on the base class so the
  shared reader satisfies VideoAnnotation's
  `self.video.fname` / `.name` access patterns without constructing
  the `utils.Video` subclass.

### Added
- **`dustrack.convert_to_mono(sources, ...)`** (`dustrack/convert.py`):
  batch re-encode helper that walks a file / list / directory and
  writes `<stem>_mono.mp4` next to each source via
  `ffmpeg -c:v libx265 -pix_fmt gray -crf 22 -preset slow`. Originals
  are never touched. Defaults match the immersionlab telemed
  convention adjusted for h265's better compression: CRF 22 lands
  within ~6% of typical ultrasound capture file size while inference
  parity stays sub-pixel (median 0.19 px, p99 0.72 px on the pia02
  36715-frame production clip). Skips sources that already have a
  monochrome `pix_fmt` and outputs that already exist. After
  conversion, the new files trigger dnav 1.5.0a1's `pix_fmt='gray'`
  auto-detect path automatically, so no caller-side change is needed
  to claim the decode-side speedup. Spike artefacts:
  `S:/_corpus/dustrack/mono_encode_bench_2026-05-21/`. Public via
  `dustrack.convert_to_mono(...)`.
- **`tests/test_gray_pix_fmt.py`** (7 tests) — pin the 2D-input
  short-circuits in `postprocess.gray`, `opticalflow._gray_rgb`, and
  `enhance_ultrasound_image` plus an end-to-end smoke that opens an
  h265 monochrome fixture and exercises the gray helpers.
- **`tests/test_convert_to_mono.py`** (7 tests) — single-file +
  directory walk + iterable input, skip-existing, skip-already-mono,
  empty-suffix refusal, missing-source tolerance.
- **`tests/qt_learning/25_benchmark_lk_rstc.py`** — LK / LK-RSTC /
  `lk_moving_average_filter` benchmark + parity harness. Mode-minor
  interleaving, demuxer state normalised between benches to avoid
  the reverse-seek artifact that lets one bench's exit state
  contaminate the next bench's timing. Writes per-rep timings,
  reference arrays, and a summary JSON under
  `%TEMP%/dustrack_lk_bench/<step>/` so a follow-up step can pass
  `--compare-to <step>` to diff parity against frozen reference
  arrays (default `atol=1e-6, rtol=0`).
- **`tests/test_dlc_trace_vectorise.py`** — 10 parity tests pinning the
  legacy implementation against the vectorised one across realistic
  inputs (dense / partial-NaN / all-NaN / single-frame / 36 715-frame
  pia02-shape, plus `pointN`-vs-named-label naming and value-type +
  frame-key contracts). The legacy implementation is preserved in
  the test file for the parity assertions, so the contract has both
  versions available in one place.
- **`tests/qt_learning/24_benchmark_cold_open.py`** — cold-open
  profiler. Times each `g.annotate()` segment under perf_counter +
  cProfile, scans the whole DLC project tree in one `os.walk` pass
  to find the worst-case multi-layer video, and reports the top-N
  cumulative + self-time entries.
- **`tests/test_pointtracking.py::test_video_annotation_accepts_passed_video`**
  + **`...::test_video_annotation_shared_video_skips_extra_av_opens`**
  — pin the shared-video contract: `video=` is identity-held, and
  six layers constructed against a shared reader add zero
  `av.container.core.open` calls.

### Notes — out of scope, kept for next session
- First-time PyAV TOC build on `M:` is ~37 s on a never-opened video,
  cached on disk after. Network-drive first-touch cost, not a code
  bug; pre-building TOCs server-side or warming them on project open
  would help.
- The remaining 3 `av.open` calls per cold-open come from one
  `VideoReader` instantiation (`PyAVReaderIndexed`'s metadata probe +
  `_load_fresh_file` + `VideoReader._probe_avg_fps`). Collapsing
  those to 1 would need either a vendored-pims edit or extending the
  TOC sidecar to cache stream-geometry + avg_fps; ~30 ms additional
  win on a network drive. Not chased here.

## [1.2.0a1] - unreleased

Structural refactor: DUSTrack absorbs the VideoPointAnnotator UI,
VideoAnnotation / VideoAnnotations data containers, and Lucas-Kanade
optical-flow helpers from datanavigator. After this release dustrack
owns its DLC story end-to-end -- VideoAnnotation's DeepLabCut HDF5
interop, the LK-RSTC `postprocess` hook, and the VPA labeling UI all
live under one roof. datanavigator narrows to data-navigation
primitives (browsers, asset managers, events).

The files moved with **full git history preserved** via `git
filter-repo` + `git merge --allow-unrelated-histories`. `git log
--follow dustrack/pointtracking.py` traces every dnav-era commit
(rc1-rc2 perf work, label-aware y-refit, `_TrackedFrameDict` mutation
guard, etc.).

### Changed
- **Absorbed `datanavigator.pointtracking`** -- `VideoPointAnnotator`,
  `VideoAnnotation`, `VideoAnnotations`, `_TrackedFrameDict` relocated
  to `dustrack/pointtracking.py` (2813 LOC lifted via filter-repo).
  VPA remains a subclass of `datanavigator.VideoBrowser` (cross-package
  import); asset containers, events, utils, `_qt` scaffolding stay in
  datanavigator. Direct construction is the building-block entry point
  (`dustrack.VideoPointAnnotator(video, ["pn", "buffer"])`,
  `dustrack.VideoAnnotation(json, video).to_signals()`);
  `dustrack.open()` remains the workflow entry point.
- **Absorbed `datanavigator.opticalflow`** -- `lucas_kanade`,
  `lucas_kanade_rstc` relocated to `dustrack/opticalflow.py` (the
  per-video shapes used by
  `VideoPointAnnotator.predict_labels_with_lucas_kanade`). The
  frame-list shapes `lucas_kanade_2` / `lucas_kanade_rstc_2` in
  `dustrack/postprocess.py` are siblings (different signature,
  shared algorithm); no rename.
- **Dropped the `class VideoAnnotation(dnav.VideoAnnotation)` subclass
  in `dlcinterface.py`** -- the `postprocess = lk_moving_average_filter`
  hook now attaches directly to the relocated
  `dustrack.pointtracking.VideoAnnotation` at import time inside
  `dustrack/__init__.py`. Removes a long-running
  `isinstance(obj, Subclass)`-narrowing trap (see
  feedback_isinstance_subclass_narrowing) -- with only one
  VideoAnnotation class, that whole class of bug is unrepresentable.
- **Floor on `datanavigator>=1.5.0a1`** in pyproject.toml -- the dnav
  release where pointtracking + opticalflow are dropped from the dnav
  surface.

### Added
- Test relocations from datanavigator: `tests/test_pointtracking.py`,
  `tests/test_opticalflow.py`, `tests/test_fast_render_parity.py`
  (narrowed to Tier-1 only -- Tier-2 image-pane parity stays in dnav),
  and `tests/qt_learning/{11,12,13,16,17,21}_*.py` perf/binding probes
  (also via filter-repo, so `git log --follow` works on each).
  New `tests/conftest.py` carries the shared helpers
  (`simulate_key_press`, `simulate_key_press_at_xy`,
  `simulate_mouse_click`, `setup_folders`, `close_figures`).

### Removed
- **`dustrack.VideoPointAnnotator`** removed from the public surface.
  Direct VPA instantiation was confirmed buggy in practice (the
  mpl-fallback path or some half-initialized state expectation that
  DUSTrack's `__init__` resolves on the subclass side); the class is
  renamed to `_DUSTrackBase` in `dustrack/pointtracking.py` to signal
  "internal -- the base that DUSTrack subclasses, not a usable
  building block." Drop-in replacement at every callsite is
  `dustrack.DUSTrack(...)`, whose constructor has the same
  `(vid_name, annotation_names, ...)` signature. `dustrack.VideoAnnotation`
  + `dustrack.VideoAnnotations` (the data containers) stay public --
  they're the programmatic surface used by ~14 portfolio files. The
  internal class is still reachable as
  `dustrack.pointtracking._DUSTrackBase` for tests + debugging.

## [1.1.0] - 2026-05-19

Minor release on top of 1.0.0. Two arcs from rc1 → rc2 fold into one
1.1.0 cut:

- **rc1 — backend perf adoption.** Adopt datanavigator 1.4.0's
  Qt-native rendering (`fast_render=True` by default; image pane on
  `QGraphicsView` + `QPixmapItem`) and `_revision`-counter cache, for
  ~4× frame-update speedup on real ultrasound sessions.
- **rc2 — DLC pipeline UX + robustness.** All three DLC-pipeline
  buttons (Train DLC model, Reduce jitter, Create DLC project) now
  run on a background thread under a shared modal `ProgressOverlay`
  instead of freezing the GUI; a unified **Done** button on the
  overlay lets the user review the final stdout (or read the error)
  before the underlying UI becomes interactive again. The training
  overlay no longer auto-dismisses, and the pre-rc2 `QMessageBox`
  failure dialogs are folded into the overlay itself. A
  save-on-close guard intercepts every way the window can close (X
  button, alt-F4, `plt.close()`) and offers *Save all / Discard /
  Cancel* on any unsaved diff. New `dustrack.open(path, layer_name=,
  ...)` workflow entry point auto-detects Phase 1 (bare video) and
  Phase 2 (DLC project) paths.

### Added
- **Save-on-close guard** -- new `DUSTrack._install_close_guard()`,
  called at end of `__init__`, patches the QMainWindow's
  `closeEvent` so it first runs `_scan_unsaved_layers()` (sibling
  of the Train pre-flight's `_scan_unsaved_and_incomplete`, scoped
  to in-memory-vs-disk diff only -- incomplete-frame quality is a
  training concern, not a close-time concern). When any layer has
  unsaved changes, a modal lists per-layer
  `+added / -removed / ~modified` counts and offers **Save all**,
  **Discard**, or **Cancel**, with *Cancel* as the default so
  accidental Enter/Esc does not lose data. Idempotent install via
  `_dustrack_close_guard_installed` sentinel attribute (a second
  install pass, e.g. from a subclass re-entry, is a no-op).
  Defensive: scan failures (e.g. annotations list torn down
  mid-shutdown) don't strand the user with an un-closeable window
  -- the guard treats its own errors as "no issues found" and
  chains through to the original `closeEvent`. mpl-fallback path
  (no Qt window) is a no-op. Sibling helpers
  `_format_unsaved_summary`, `_prompt_save_on_close`,
  `_save_unsaved_layers`. 18 new tests in
  `tests/test_save_on_close.py`.
- **`ConfirmOverlay`** -- new module-level `_make_confirm_overlay_class`
  factory in `dustrack/dlcinterface.py`, sibling to
  `_make_progress_overlay_class`. Modal confirm overlay parented to
  the DUSTrack QMainWindow that shares the dark-translucent backdrop +
  reposition + event-filter scaffolding with `ProgressOverlay`;
  synchronous (`exec_()` runs a local `QEventLoop` and returns the
  clicked button's label string). Severity-aware title color
  (`info` / `warning` / `destructive`, reusing ProgressOverlay's
  green `#7cdb7c` / red `#ff7c7c` palette so the two overlays share
  visual vocab); per-button QSS styled by role
  (`primary` / `destructive` / `neutral`); `default=` names the
  button that receives focus. Replaces the two pre-rc2 `QMessageBox`
  sites (`_prompt_unified_pre_flight`, `_prompt_save_on_close`) and
  hosts the two new rc2 confirms below. External contracts
  (`bool` for pre-flight, `"save"`/`"discard"`/`"cancel"` for
  save-on-close) preserved.
- **`Discard unsaved annotations` button** (display group) +
  `DUSTrack.discard_unsaved_annotations()` action. Drops the active
  layer's in-memory edits and re-syncs from disk via
  dnav 1.4.0rc2's new `VideoAnnotation.reload()`; confirm body
  branches on whether the layer's backing file exists (Reload from
  disk vs Reset to empty). Refuses on `dlccorr` and any layer
  matching `_is_dense_layer_name` with an info notice pointing the
  user at `Remove layer` -- "discard" has no meaningful semantic
  for layers regenerated from other layers. mpl-fallback path
  (no Qt window): falls through to `ann.reload()` + `update()`
  silently, same convention as the rest of DUSTrack's Qt-specific
  UI.
- **`Remove layer` button** (niche group) +
  `DUSTrack.remove_current_layer()` action. Drops the active
  annotation layer from the session via dnav 1.4.0rc2's new
  `VideoPointAnnotator.remove_annotation_layer()`. **Session-only:**
  the backing file on disk is *not* deleted, so the layer reappears
  on next launch unless the file is removed manually -- pair with a
  filesystem delete (or `Save annotation as...` to a different name
  + delete the original) when the intent is "undo manual
  corrections". Severity-aware confirm body via
  `_is_dense_layer_name`: dense/derived layers (regenerable) default
  to `Remove layer`; sparse/authored layers (irreversible) default
  to `Cancel` and include the to-be-dropped frame count. Refuses
  with an info notice if only one removable layer remains
  (excluding the implicit `"buffer"`), pointing the user at
  `Discard unsaved annotations` for the "reset contents" semantic.
  Requires dnav 1.4.0rc2 for the underlying
  `remove_annotation_layer` surface.
- **Two-slider `EnhanceWidget` + `[None | Auto]` row** -- new
  `_make_enhance_widget_class` factory + new
  `DUSTrack._add_enhance_widget()` that mounts the widget below the
  statevars widget in the rc2 left-column dock. Two `QSlider`
  controls with live numeric labels: **CLAHE clip** maps to
  `[1.0, 4.0]` and **Gamma** maps to `[1.0, 2.0]` (extended from
  the originally-proposed `[1.0, 1.5]` to give headroom for darker
  ultrasound footage where the 1.5 ceiling was being hit by Auto).
  CLAHE grid (`8`) stays at the `__init__` default. Slider values
  update `self._clahe_clip` / `self._gamma` and call
  `self.update()` on every value change so the image redraws live.
  Two trigger buttons below the sliders:
  - **None**: snap both sliders to leftmost (the
    `_enhance_is_passthrough` bypass position). Convenience undo
    for Auto / manual enhancement.
  - **Auto**: one-shot inference of slider values from the
    current raw frame via the new module-level
    `_auto_enhance_params(image)` helper. Heuristic on grayscale
    histogram percentiles -- low p95-p5 dynamic range pushes
    CLAHE clip up; low p50 (dark midtones) pushes gamma up; both
    clamped to slider extents. Slider values stay where Auto
    put them across frame navigation -- the heuristic only runs
    on click, not per-frame. Anchors tuned in four passes against
    real ultrasound footage; the pass-4 anchors
    (`LOW=0`/`HIGH=75`/`DARK=0`/`MID=25`) are calibrated against
    the S-corpus DUSTrack clip (inferred stats dyn~61, p50~20) so
    the user-target clip~1.6 / gamma~1.3 falls in range. Anchors
    deliberately make "typical brightish" ultrasound
    (p50~60, dyn~80) a near-bypass; Auto only kicks in noticeably
    for dark + low-contrast frames.

  **Per-slider bypass for smooth left-end transitions.** Image
  processor has three paths now: both at min -> raw image; clip at
  min + gamma off -> `_apply_gamma_only(im, gamma)` (per-channel
  `cv.LUT`, no CLAHE, no `cvtColor` roundtrip) so the gamma slider
  transitions visually continuously from raw; clip off min -> full
  `enhance_ultrasound_image`. The clip slider still has a step at
  left-end zero because CLAHE startup at clip=1.0 isn't a true
  identity, but moving the gamma slider alone no longer drags the
  CLAHE pipeline along for the ride. Pre-fix, moving either
  slider one tick off zero triggered CLAHE@1.0 + `cvtColor`
  roundtrip -- visibly noticeable jump.

  Shared helper `EnhanceWidget._apply_param_pair(clip, gamma)`
  factors the signal-blocking + integer-quantization + label sync
  + single-redraw tail used by both `None` and `Auto`. Sliders are
  integer `0..100`; pure-function helpers
  (`_slider_to_clahe_clip` / `_slider_to_gamma` / inverses,
  `_enhance_is_passthrough`, `_apply_gamma_only`,
  `_auto_enhance_params`) live at module scope and are
  unit-tested in `tests/test_enhance_widget_mapping.py`
  (42 cases).

  **Sliders-at-minimum is the true bypass**:
  `_enhance_is_passthrough(clip, gamma)` returns `True` when both
  sliders sit at their leftmost position (clip=1.0 AND gamma=1.0);
  the image processor short-circuits and returns the raw frame
  untouched (skipping the CLAHE pass and the RGB->gray->RGB
  roundtrip). Replaces the originally-rc2 `Toggle enhance`
  button -- with slider-driven bypass at min, a separate toggle is
  redundant. Constructor defaults shifted to
  `clahe_clip=1.0` (was 2.0), `gamma=1.0` (was 1.2),
  `brightness=0` (was 10) so DUSTrack opens with the raw frame and
  the user dials enhancement in via the sliders. Dropping the
  `brightness=+10` baseline means nudging either slider one tick
  off zero doesn't visually jump a +10 brightness offset in
  alongside CLAHE/gamma -- the transition off the bypass is purely
  the user-driven slider values. `clahe_clip` / `clahe_grid` /
  `gamma` / `brightness` kwargs on `DUSTrack.__init__` stay
  available for callers who want to seed the sliders to non-default
  positions. mpl-fallback path: `_add_enhance_widget` is a no-op
  and the constructor defaults stand.
- **Pinned Qt palette** -- new module-level helper
  `_pin_qt_palette(dark: bool)` called from `DUSTrack.__init__` so
  the GUI looks the same regardless of Qt binding + Windows system
  theme. Sets `QApplication.setStyle("Fusion")` and writes an
  explicit `QPalette` (light or dark variant keyed off the
  `dark_mode` kwarg) so PySide6 6.5+ on Windows -- which honors
  the OS color scheme by default while PyQt6 does not -- can't
  silently flip DUSTrack to dark mode just because the host
  machine is in dark mode. Both bindings are in play across
  portfolio envs (DLC mandates PySide6 via
  `deeplabcut/gui/__init__.py:14` setting `QT_API=pyside6`, while
  matplotlib/older envs prefer PyQt6), so without the pin the
  same DUSTrack code paints differently on different machines --
  including dnav's built-in stylers, which sample the live
  palette via `datanavigator.styles._is_dark_mode`. The explicit
  light palette also avoids `app.style().standardPalette()`,
  which under Qt 6.5+ Fusion follows the OS color scheme and
  defeats the pin. No-op on the mpl-only path (qtpy import
  fails).
- **Canonical layer regrouping** -- new
  `DUSTrack._restructure_annotation_order()` partitions
  `self.annotations` into six groups and regroups via dnav 1.4.0rc2's
  new `VideoAnnotations.reorder(names)`, preserving intra-group
  order: `manuals -> manual_corrections -> labeled_data -> dlc_* ->
  dlccorr* -> buffer`. Wired into `DLCProject.annotate` (fresh-open),
  `_refresh_dlc_layers` (post-train), and `_adopt_layer`
  (Reduce-jitter / `apply_manual_corrections`) so all three entry
  points end at the same canonical order. Pre-rc2 those paths
  appended new layers to the tail of `self.annotations`, which
  interleaved manuals with prior DLC traces and pushed the next-
  iteration manual layer behind the previous iteration's dlc_*
  outputs in the layer dropdown. Active layer + overlay are
  preserved by name across the reorder. `dlccorr` (and its derived
  `dlccorr_lkmovavg_*` LK outputs) gets its own group at the tail of
  the DLC chain rather than folded into manuals -- the manual
  entries it incorporates live in a separate active layer; `dlccorr`
  itself is the spliced output, not a hand-edited layer. The new
  `manual_corrections` group sits right after manuals so the
  source-of-corrections layer (see `apply_manual_corrections`
  rename entry below) stays adjacent to its `iteration-N` peer.
- **Apply manual corrections: preflight save + source-layer rename.**
  `apply_manual_corrections` now (a) auto-saves the active patch
  layer before splicing if it has any unsaved diff (so the on-disk
  state stays coherent with the `dlccorr` it's about to write),
  and (b) renames the patch layer to `<old_name>_manual_corrections`
  on success -- in-memory `.name` + `.fname` plus an on-disk file
  move (write-new-before-unlink-old so an interrupted rename leaves
  data recoverable). For the canonical `iteration-N` patch the new
  name is `iteration-N_manual_corrections`, picked up by the new
  `manual_corrections` group in `_restructure_annotation_order`'s
  classifier. Idempotent: re-applying when the patch is already
  suffixed skips the rename. New helper
  `DUSTrack._rename_annotation_layer(old, new)` factors the
  file-move + statevar resync + selection-preservation mechanics so
  the rename machinery is independent of the corrections workflow.
  New class constant `DUSTrack.MANUAL_CORRECTIONS_SUFFIX`
  (`"_manual_corrections"`) is the single source of truth for the
  suffix string.

### Fixed
- Latent shadowing bug in `_load_layer_disk_data`: was calling the
  module-shadowed `dustrack.open` workflow entry point instead of
  `builtins.open`, which would dispatch JSON paths through
  `dustrack.open`'s Phase-1 branch and raise `ValueError("layer_name
  is required")`. Existing pre-flight tests stubbed the method at a
  higher layer so the bug was latent in rc2. Switched to
  `Path.read_text(encoding="utf-8")` + `json.loads`, the same
  convention `DLCProject._read_trackermap` already uses for the
  same reason.
- Sibling shadowing bug in `_save_dropped_incomplete_sidecar`: was
  calling bare `open(sidecar, "w")` which resolved to the
  module-level `dustrack.open()` workflow entry point and raised
  `FileNotFoundError` on the not-yet-existing sidecar path. Switched
  to `Path.write_text`, matching the convention now documented in
  `_load_layer_disk_data`.
- **`_dlcproject` wiring on re-entered sessions.** `DLCProject.annotate`
  constructs a fresh `DUSTrack` but did not set
  `ret._dlcproject = self` on the returned instance. Result: the
  Train DLC model / Reduce jitter buttons (and `_refresh_dlc_layers`)
  raised `"DLCProject not created."` on a re-entered project
  session, since `__init__`'s default left `_dlcproject = None`. Now
  wired explicitly at the tail of `DLCProject.annotate`.
- **`copy_existing_annotations_from_overlay` ("Replace existing from
  overlay" button) now routes through `ann.add()`** instead of
  writing directly into `self.ann.data[L][F]`. On dnav 1.4.0+ the
  direct write skipped the `_revision` bump and the trace-display
  cache kept serving pre-mutation ydata, so the button looked like a
  no-op on the trace plot. Same bug class as dnav's
  `check_labels_with_lk` (fixed upstream in datanavigator 1.4.0).
  Bug latent since 1.1.0rc1 picked up dnav 1.4.0's cache; 1.0.0 +
  dnav 1.3.x stack was unaffected.
- **`apply_manual_corrections` no longer crashes when the user
  has manually corrected only one of two (or more) labels.**
  Pre-fix path: `VideoAnnotation.save()` pruned empty labels at
  serialization time, so the patch overlay's untouched label
  silently disappeared from disk and from `self.labels`; the
  subsequent `self.update()` reached
  `VideoPointAnnotator.update_frame_marker`, which iterates every
  layer with the active label and calls `to_trace(label)` on each
  — for the pruned-label layer that asserted
  `label in self.labels` and blew up. Fix relies on dnav 1.4.0rc2's
  three-part label-schema rework (save preserves empty labels +
  schema-tolerant `to_trace` + n_labels default drops to 1). On the
  DUSTrack side this required no new code; `_normalize_layer_data`'s
  empty-label filter still applies symmetrically to both sides of
  the diff and the comment was updated to reflect the new dnav
  contract.

### Changed
- **`DUSTrack(...)` annotation layer name default** is now
  `"iteration-0"` (was: required positional or empty string from
  dnav). `dustrack.open(path)` no longer raises when called without
  `layer_name` on a bare video (Phase 1); the constructor default
  takes over. `iteration-0` seeds the canonical DLC iteration-N
  naming so the next DLC training round lands as `iteration-1`
  rather than colliding with whatever ad-hoc name the user picked.
  Constructor signature shifts from `__init__(self, *args, ...)` to
  `__init__(self, vid_name, annotation_names="iteration-0", *args,
  ...)`; existing positional callers (e.g.
  `DLCProject.annotate`'s `DUSTrack(video, annotation_names,
  height_ratios=...)`) are unaffected.
- `dustrack/dlcinterface.py`: button-column separators promoted from
  single to **double** (dnav 1.4.0rc2's new
  `Buttons.add_separator(style="double")`) to mark the major
  functional groups in the rc2 sidebar. Initial rc2 grouping was
  `shortcuts | DLC pipeline | trace + display controls`; a later
  rc2 polish pass (same release window) re-ordered the buttons to
  the final task-flow layout below and folded "Save annotation
  as..." into the workflow group so it stays adjacent to the
  pipeline actions that produce annotations. Final rc2 column order
  (top-to-bottom):
    1. **Workflow** -- Create DLC Project → Train DLC model → Apply
       manual corrections → Reduce jitter → Save annotation as...
    2. **Display / trace** -- Discard unsaved annotations →
       [Trace: line | Trace: dot] →
       [Freeze plot axes | Unfreeze plot axes]
       (the bracketed pairs render as one row each via
       `Buttons.add_multi`; see the next bullet)
    3. **Niche op** -- Replace existing from overlay → Remove layer
       *(the original niche-op-keyboard-shortcut question is parked
       — adding Remove layer here makes the group a coherent
       "layer-mutating affordances" cluster.)*
    4. **Utilities + Swap** -- Refresh UI → Keyboard shortcuts →
       Swap layers
  Groups 1-3 are separated by single double-separators; group 3 is
  separated from group 4 by **two** double-separators so Swap
  layers reads as a deliberately set-apart trailing action. The
  state-variables section below appends its own trailing double
  separator for free via dnav's `_QtStatevarsWidget`. Users no
  longer have to scan for group boundaries in a long flat list of
  buttons.
- `_prompt_unified_pre_flight` and `_prompt_save_on_close` now route
  through the new `ConfirmOverlay` instead of `QMessageBox`. External
  return contracts (`bool` for pre-flight, `"save"`/`"discard"`/
  `"cancel"` for save-on-close) are unchanged; the change is purely
  visual + interactive (shared backdrop + per-role button styling
  with the other rc2 modals). The native `QFileDialog.getSaveFileName`
  in `save_annotation_as` stays native -- file pickers carry too much
  UX equity (recents, drive nav, paste-path) to replace.
- `Toggle enhance` button and the `_toggle_enhancement` /
  `_enhance_enabled` plumbing are gone -- the EnhanceWidget
  sliders own the on/off semantics now (both at min = bypass).
  See the EnhanceWidget bullet above.
- `dustrack/dlcinterface.py`: the **Trace: line / Trace: dot** and
  **Freeze plot axes / Unfreeze plot axes** pairs in the Display /
  trace group now render as side-by-side two-button rows via dnav
  1.4.0rc2's new `Buttons.add_multi(*specs)`. Each pair consumed two
  vertical sidebar slots pre-rc2-polish; now each consumes one, so
  the Display group occupies 3 column slots instead of 5. Labels
  kept verbatim (half-column width is comfortable and the existing
  keybind muscle memory wins). Each spec carries `style_tag="display"`
  so the per-button color styling is applied uniformly across the
  row (see the styling bullet below).
- `dustrack/dlcinterface.py`: `DUSTrack._add_default_buttons()`
  overrides the dnav 1.4.0rc2 hook to a no-op. Pre-fix, dnav's
  `VideoPointAnnotator.__init__` installed "Refresh UI" at slot 0
  of the buttons column; DUSTrack now places "Refresh UI" itself
  next to "Keyboard shortcuts" as a utility pair just above
  "Swap layers" (see the column order above).
- `dustrack/dlcinterface.py`: per-group color styling now rides
  dnav 1.4.0rc2's new `Buttons.register_style` / `style_tag=` API.
  A per-group styler closure is built from `_SIDEBAR_PALETTE` via
  the new module-level `_make_group_styler(spec)` helper and
  registered on `self.buttons` once at the top of the rc2 sidebar
  block; each `add` / `add_multi` call then declares its
  `style_tag="workflow"` (etc.) inline, and the styler runs
  per-button at add-time inside dnav's `_finalize_button`. The
  pre-rc2 intermediate machinery (`_btns_workflow / _btns_display /
  _btns_niche / _btns_utilities / _btns_swap` collection lists
  walked by an end-of-setup `_style_sidebar_buttons` batch pass) is
  gone; each sidebar button now lives in one place (its `add`
  call) with its styling tag right there. Qt path only -- the
  styler closure no-ops when `_qt_btn` is absent, matching the
  pre-refactor mpl-fallback behavior. Applied unconditionally so
  users on the default `dark_mode=False` still see the coordinated
  rc2 sidebar. Pastel analogous band (cool -> warm -> neutral)
  with a single dark-slate text color across all groups so the eye
  doesn't retune contrast row-to-row:
    - **Workflow** (5 btns) -- powder blue `#cfdef3`; primary
      pipeline, coolest end of the band.
    - **Display** (5 btns) -- pale mint `#d4ebd4`; cool green,
      analogous step from blue.
    - **Niche** (Replace from overlay) -- pale apricot `#f5d9c0`;
      warm shift signals "use sparingly".
    - **Utilities** (Refresh UI, Keyboard shortcuts) -- pale sand
      `#ece6d5`; neutral warm.
    - **Swap layers** -- pale silver `#e0e4e8`; matches the
      statevars widget bg.
  Each group's stylesheet sets bg / fg / border + hover / pressed
  states only (no `border-radius` or padding overrides) so the
  QSS rendering stays close to a flat-colored variant of the
  native button. The statevars widget bg is set via **palette
  manipulation** (not QSS) so child QComboBoxes inside it keep
  their native Windows-style dropdown rendering -- a `QWidget`
  QSS selector cascades into children and re-skins the combos
  flat, which an earlier rc2 attempt did and got noticed; palette
  propagation respects native widget styling. The statevars
  palette logic lives in a small `_paint_statevars_widget` helper
  called once after all `add` calls -- it can't ride the
  `Buttons` styling registry because the registry is per-button
  (QSS-only), and the statevars widget needs a sibling-of-buttons
  palette pass. Earlier rc2 iterations went through a
  high-contrast white-on-black workflow accent, a saturated
  cool-tone dark palette, and a teal/rose pastel before settling
  on this mint/apricot variant; the dark versions read as gaudy
  next to a dark figure canvas and the teal/rose felt off-key.
- `datanavigator/_qt.py` (via dnav 1.4.0rc2): `_QtStatevarsWidget`
  no longer renders the bold "State variables:" title row -- the
  trailing double separator and per-row dividers already delimit
  the section. The widget paints itself with a slightly darker
  background than the parent dock (palette `base.darker(120)`,
  theme-adaptive) so the statevars area is visually distinct from
  the buttons column above it. DUSTrack inherits both changes
  without a code change of its own.
- `dustrack/dlcinterface.py`: `DUSTrack.process_dlc_project`
  (**Train DLC model**) no longer closes the figure and re-opens it
  after training. On a Qt backend (the default for
  `DUSTrack(..., fast_render=True)`, which is the default), training
  now runs on a background thread under a modal "Training in
  progress" overlay parented to the QMainWindow. The overlay shows
  the current pipeline phase (extract / train / evaluate / analyze
  / labeled-video), a progress bar driven by parsed `Epoch X/Y` and
  iteration markers in DLC's stdout, and a scrolling tail of the
  last few hundred log lines. The newly-produced DLC trace layers
  are added to the live DUSTrack via `add_annotation_layers` -- no
  relaunch -- with the freshest `dlc_*` layer set as the annotation
  overlay and drawn as a line plot. The pre-rc2 close-and-reopen
  path is retained as the fallback when no QMainWindow can be
  located (non-Qt backend, headless run, etc.).
- `dustrack/dlcinterface.py`: `DUSTrack.process_dlc_project` gains
  a **unified pre-flight check** before the overlay starts that
  scans *every* manual annotation layer in the session, agnostic
  to which layer is the active one / overlay / placeholder.
  Manual layers are identified by file pattern (`.json` alongside
  the video matching `<video_stem>_annotations*.json`, minus
  `dlccorr` / `buffer` / `dlc*`) so a user who renamed
  `iteration-1` to `iter1` on disk or seeded the initial layer
  with experimenter initials is still picked up. For each manual
  layer the scan computes (a) in-memory diffs vs the on-disk
  JSON (added / removed / modified frames) and (b) frames missing
  one or more bodyparts. If any layer has either kind of issue,
  a single Qt modal lists the per-layer breakdown and offers
  *Save and clean* (write per-layer recovery sidecars for the
  dropped frames, drop the incomplete frames via dnav 1.4.0rc2's
  `keep_overlapping_frames`, save every affected layer, then
  train) or *Cancel* (return to the UI without changes). Layers
  without issues are not touched; DLC traces / `dlccorr` /
  `buffer` are never in scope. Recovery sidecars use the composite
  extension `<fstem>.dustrack-dropped-incomplete-<YYYYMMDDTHHMMSS>`
  so the annotation-discovery glob does not re-ingest them on
  subsequent training runs. This replaces the rc2 active-layer-only
  pre-flight (and its "empty layer skip" / "would crash on h5
  active layer" sharp edges): a user who switched the active
  layer to a DLC trace or a previous iteration before clicking
  Train no longer crashes the save call or silently overwrites
  an unrelated layer -- those layers fall out of scope by file
  pattern, while any genuinely unsaved manual edits surface in
  the modal regardless of which layer is active. Non-Qt fallback
  path (no QMainWindow) skips the modal -- no GUI to host it.
- `dustrack/dlcinterface.py`: `DUSTrack.process_dlc_project` now
  defaults `create_video=False` (via `kwargs.setdefault`) on both
  the Qt and the non-Qt fallback paths -- the **Train DLC model**
  button no longer calls `deeplabcut.create_labeled_video` after
  inference. The annotate -> train -> review -> annotate loop reads
  the new predictions through the refreshed DLC trace layers, so
  the labeled mp4 is wasted work on every UI-driven training pass.
  Direct callers of `DLCProject.process(...)` (CLI / notebook) keep
  the pre-existing `create_video=True` default; UI callers can
  still pass `create_video=True` explicitly to override. A
  user-facing toggle is parked for the planned 1.x training-control
  pass (dnav-versioned 1.6.0).
- `dustrack/dlcinterface.py`: `DUSTrack.process_with_lk`
  (**Reduce jitter**) joins the overlay path on a Qt backend (no
  more UI freeze during long LK-RSTC passes). The overlay log
  shows the tqdm output and the progress bar is driven by parsing
  tqdm's `N/M` markers; phase label flips between "Submitting
  tracking jobs" and "Processing tracking results". On non-Qt
  backends the pre-existing synchronous behavior is retained,
  including the `VideoAnnotation` return value. On the Qt path the
  smoothed layer is added (and selected) when the user clicks Done.
- `dustrack/dlcinterface.py`: `DUSTrack.process_with_lk` now
  `save()`-es the source layer to disk right before LK kicks off,
  mirroring the pre-train save in
  `DUSTrack.process_dlc_project`. In the typical workflow the
  source is the `dlccorr` layer (active after
  `apply_manual_corrections`) and the save persists any in-memory
  manual edits before smoothing. Sources without a `.json`
  filename (raw DLC traces loaded from `.h5`) are read-only
  inputs; the save is skipped with a one-line note since
  `VideoAnnotation.save()` only writes JSON.
- `dustrack/dlcinterface.py`: `DUSTrack.create_dlc_project`
  (**Create DLC Project**) joins the overlay path on a Qt backend
  with `show_progress_bar=False` (it's a quick op with no
  determinate progress). DLC's create-project chatter is teed to
  the overlay log + the launching terminal; the Done button
  confirms the project location before the user moves on. Sync
  return on non-Qt backends is unchanged.
- `dustrack/dlcinterface.py`: training-overlay failure path no
  longer pops a separate `QMessageBox.critical` /
  `QMessageBox.warning` dialog. The error is shown in the overlay
  itself (title flips to "Failed", exception in the phase label,
  traceback in the log); the same Done button dismisses cleanly.
  Same folding applies to the rare "training succeeded but layer
  refresh failed" path -- and that follow-up branch now also streams
  its traceback into the overlay log (previously only `sys.__stderr__`,
  invisible when DUSTrack is launched from an IDE / launcher that
  doesn't show the original stderr) and falls back to `repr(exc)`
  when `str(exc)` is empty, so a bare `assert X` no longer surfaces
  as a blank "AssertionError: " summary.
- `dustrack/dlcinterface.py`: DLC's stdout/stderr during overlay
  work are teed to `sys.__stdout__` (the original terminal file
  descriptor) rather than the possibly-wrapped `sys.stdout`, so
  launching from a shell reliably shows progress in the terminal
  even when the call is initiated from a GUI button handler.
  Output also feeds the in-app overlay log in real time.
- `dustrack/dlcinterface.py`: layer-naming harmonised across the
  cold-open / post-train-refresh / in-session-adopt paths. Pre-fix,
  `DLCProject.annotate` and `DUSTrack._refresh_dlc_layers` ran every
  filepath through `VideoFileManager._get_annotation_name` /
  `_get_dlc_trace_name`, while `DUSTrack.process_with_lk` (Reduce
  jitter) constructed a `VideoAnnotation` directly and inherited
  `VideoAnnotation.__init__`'s `"noname"` fallback because the
  LK-RSTC output `<stem>_lkmovavg_<window>.json` lacks
  `_annotations_`. Symptom: jitter-reduced layer briefly showed
  `"noname"` in-session but the canonical `dlc_iteration-N_<window>`
  name on close + reopen. New `VideoFileManager.canonical_layer_name(fname)`
  is the single source of truth (stem-pattern dispatch: `_annotations`
  suffix branch / `DLC` trace branch incl. LK outputs / file-stem
  fallback); the prior `_get_annotation_name` and
  `_get_dlc_trace_name` static methods are deleted; the
  `annotations` and `dlc_traces` properties now call
  `canonical_layer_name`.
- `dustrack/dlcinterface.py`: `DUSTrack.process_with_lk`
  (**Reduce jitter**) routes the new smoothed layer through the new
  `DUSTrack._adopt_layer` helper on both sync and Qt-async paths.
  Layer name is re-derived from the filepath via
  `canonical_layer_name`, so the noname → reload mismatch is gone;
  for DLC-trace sources the new layer also gets `set_plot_type("line")`
  + the dlc-overlay re-pointing convention via
  `_normalize_dlc_layer_display`, which previously only ran on the
  cold-open / post-train paths.
- `dustrack/dlcinterface.py`: DLC-pipeline button row reordered so
  **Apply manual corrections** sits above **Reduce jitter** -- the
  workflow runs corrections, *then* smooths the corrected layer, so
  the buttons now read top-to-bottom in the order they're clicked.
- `dustrack/dlcinterface.py`: `create_dlc_project` (sync + Qt-async
  paths) now calls a new `_rewire_to_in_project_paths()` helper on
  success. `self.fname` is repointed to the in-project video copy,
  and every annotation layer whose `.fname` lives outside the
  project tree is migrated to the project's `videos/` folder
  (path-only for empty layers; layers with in-memory data also
  `save()` at the new path so subsequent reloads find the file).
  Subsequent writes (`apply_manual_corrections`, `process_with_lk`,
  `save_annotation_as`) now land inside the project rather than
  next to the original video -- which was the root cause of
  Phase-1 → Phase-2 relaunches not seeing previously-produced
  outputs. `self.data` (the video reader) is intentionally left
  pointing at the original file: DLC's `copy_videos` guarantees
  byte-identical content, and rebuilding the reader mid-session
  would invalidate the Qt image pane handle.
- `dustrack/dlcinterface.py`: `_adopt_layer` no longer short-circuits
  on `set_active` / `set_overlay` when the requested layer is already
  loaded. Reduce jitter on a layer whose cached LK output is already
  in the session now still swaps the UI to the smoothed layer with
  the source pinned as overlay (previously the early `return None`
  left the UI on the source layer).
- `dustrack/dlcinterface.py`: the "render as line plot, not dots"
  default for DLC-pipeline layers now covers `dlccorr` (the
  manual-corrections splice) and every LK-RSTC jitter-reduced
  output, not just DLC inference and LK outputs derived from a
  DLC trace. The pre-fix predicate in `_normalize_dlc_layer_display`
  and `_adopt_layer` was `name.startswith("dlc_")`, which matched
  DLC inference and the LK output of DLC traces (named
  `dlc_iteration-N_<window>` via the `DLC` branch of
  `canonical_layer_name`) but missed `dlccorr` itself and the LK
  output of the `dlccorr` layer (named `dlccorr_lkmovavg_<window>`
  via the `_annotations` branch). `dlccorr` is dense because it's
  the overlay's per-frame DLC trace with the active manual layer's
  sparse edits spliced in -- per-frame coverage is inherited from
  the overlay. Symptom: clicking *Apply manual corrections* or
  *Reduce jitter* on the manual-corrections layer produced a dense
  trajectory that rendered as disconnected dots, requiring a
  manual *Trace: line* click. Fixed by widening the plot-type
  predicate to a new module-level helper `_is_dense_layer_name`
  (matches `dlc_` or `dlccorr` prefix OR `lkmovavg` substring);
  the overlay-pin predicate stays at `dlc_*` so neither
  *Apply manual corrections* nor a Reduce-jitter click on a
  manual layer silently retargets the overlay off the latest DLC
  inference. Applies to all three layer-add paths -- cold open
  (`DLCProject.annotate`), post-train refresh
  (`_refresh_dlc_layers`), and in-session adopt (`_adopt_layer`).
  Pattern data lives in the `_DENSE_LAYER_PREFIXES` /
  `_DENSE_LAYER_SUBSTRINGS` tuples so adding a future smoothing
  recipe is a one-line edit.

### Added
- `dustrack/dlcinterface.py`: pre-flight helpers on `DUSTrack` for
  the Train DLC model unified pre-flight check. Pure staticmethods:
  `_scan_incomplete_frames` (per-frame missing-bodyparts detector),
  `_is_manual_annotation_layer` (file-pattern based, not
  name-based, so renamed `iteration-N` files still resolve),
  `_normalize_layer_data` (canonical int-keyed float-valued form
  for diffing), `_diff_ann_vs_disk` (in-memory vs disk
  added/removed/modified detector), `_load_layer_disk_data`
  (read + normalize a layer JSON; empty dict for missing files),
  `_build_dropped_incomplete_payload`,
  `_build_dropped_incomplete_sidecar_name`,
  `_format_incomplete_breakdown`, and `_format_pre_flight_summary`
  (multi-layer report). Instance methods:
  `_scan_unsaved_and_incomplete` (the orchestrator),
  `_save_dropped_incomplete_sidecar` (per-layer sidecar writer),
  `_prompt_unified_pre_flight` (`QMessageBox` modal with *Save
  and clean* + *Cancel*), and `_apply_pre_flight_remediations`
  (per-layer drop + save). The pure helpers ignore empty
  placeholder labels (a label with zero annotations is not
  treated as "every frame missing this label"), matching what
  `VideoAnnotation.save()` + dnav's `keep_overlapping_frames`
  act on.
- `tests/test_train_dlc_preflight.py`: 40 synthetic-data tests
  covering the pure pre-flight staticmethods --
  complete-frames detection across empty-data / all-placeholder-
  labels / single-missing-label / multi-missing-labels / sort-
  order cases; sidecar payload includes only present labels (not
  missing) and casts to floats; sidecar filename uses the
  composite suffix (no `.json` extension) and lives next to the
  annotation; breakdown formatter truncates at `max_rows` with a
  tail line; `_is_manual_annotation_layer` covers the
  iteration-N / unsuffixed / renamed-to-iter1 /
  experimenter-initials / dlccorr / buffer / dlc_* / h5 /
  different-dir / wrong-stem-pattern / None matrix;
  `_normalize_layer_data` casts int / float as expected and
  drops empty labels; `_diff_ann_vs_disk` returns the right
  add/remove/modify partition with label-then-frame sort order;
  `_format_pre_flight_summary` renders diffs-only / incomplete-
  only / both / multi-layer / truncated cases. The instance
  methods (`_save_*` writes a file, `_prompt_*` shows a modal,
  `_apply_*` mutates layer state) and the `process_dlc_project`
  wiring are not unit-tested -- they need a live GUI session.
- `dustrack/dlcinterface.py`: `DUSTrack._refresh_dlc_layers(video_index=0)`
  helper -- factors the "load trained outputs from disk into the live
  session" step out of `DLCProject.annotate`. Now mirrors `annotate()`
  more completely: also creates an empty
  `iteration-{latest+1}` layer to capture next-round manual
  refinements, and activates it as the current annotation layer so the
  user can immediately start labeling. The freshest `dlc_*` layer is
  still set as the annotation overlay. Falls back gracefully (no
  raise) if the new-iteration JSON already exists on disk from a
  prior refresh. Idempotent; safe to call more than once.
- `dustrack/dlcinterface.py`: module-level `_Tee`, `_QueueWriter`,
  and lazily-built `_make_progress_overlay_class()` -- the plumbing
  for off-thread DLC-pipeline work + modal overlay. The Qt class
  builder mirrors datanavigator's `_make_qt_text_overlay_class`
  pattern so importing dustrack on a no-Qt-binding machine doesn't
  touch qtpy. The overlay is parameterized by title / initial phase
  / hint / progress-bar visibility, and exposes a
  `mark_done(success, summary)` method that swaps in a styled
  **Done** `QPushButton` wired to dismiss the overlay and fire a
  post-completion callback; the title flips to "Complete" (green)
  or "Failed" (red).
- `dustrack/dlcinterface.py`: `DUSTrack._run_with_overlay(qt_window,
  *, work_fn, on_success, title, hint, ...)` -- the generic
  worker-thread + QTimer + queue plumbing shared by all three
  DLC-pipeline buttons. Future buttons (e.g. re-train from a
  snapshot, post-hoc evaluation) plug in by passing a `work_fn` +
  `on_success` pair. `DUSTrack._find_qt_window` factored out as a
  small helper for the Qt-vs-headless dispatch.
- `dustrack/dlcinterface.py`: overlay log drainer now feeds
  `\r`-separated tqdm redraws through the phase + progress
  matchers (so the progress bar updates live during tqdm bars) but
  only appends `\n`-terminated lines to the visible log, so the
  log doesn't flood with partial redraws.
- `dustrack/dlcinterface.py`: overlay worker captures and surfaces
  the full traceback on failure. The traceback is pushed through
  the teed sink so it lands in both the overlay log + the
  launching terminal; the overlay summary line uses
  `"{ExcType}: {str(exc)}"` so single-character `KeyError` / `IndexError`
  arguments (e.g. `KeyError(0)`) read as `"KeyError: 0"` instead of
  just `"0"`.
- `dustrack/postprocess.py`: `lk_moving_average_filter` now filters
  the input layer's labels to those with complete frame coverage
  before entering the LK-RSTC loop. Sparse labels (e.g. a
  freshly-clicked manual refinement point that exists at only one
  frame) are skipped with a one-line `[reduce_jitter] layer={name}:
  skipping N/M label(s)` warning that names the source layer and
  the skipped labels with their per-label coverage. If *no* labels
  have complete coverage the function raises a clear `ValueError`
  naming the layer and listing per-label coverage instead of
  crashing later with a useless `KeyError(0)` from the inner loop.
  Pre-existing bug: the inner loop did
  `[ann.data[label][start_frame] for label in label_list]`
  unconditionally, so any sparse label made Reduce jitter fail
  silently (when called from a button handler) or noisily
  (when called from the new rc2 overlay).
- `dustrack/dlcinterface.py`: **Reduce jitter** overlay title /
  initial phase / success summary now name the source layer
  (`"Reducing jitter (dlc_iteration-2_150)"`, etc.) so the user can
  see at a glance which layer is being smoothed.
- `dustrack/dlcinterface.py`: `_TRAINING_PHASES` joined by
  `_JITTER_PHASES` (matches tqdm `desc=` prefixes for the LK loop)
  and `_CREATE_PROJECT_PHASES` (matches DLC's create-project
  stdout chatter).
- `dustrack/dlcinterface.py` + `dustrack/__init__.py`: new
  `dustrack.open(path, layer_name=None, **dustrack_kwargs)` unified
  entry point. Pass a video to start fresh (layer_name required), a
  DLC project root (or its `config.yaml`, or a video inside one) to
  resume in place. Dispatch helpers: `_is_dlc_project_root`,
  `_find_dlc_config`, `_find_video_index`. Tests at
  `tests/test_open.py` cover the dispatch error branches + the
  path-classification helpers with synthetic filesystem inputs;
  the Phase 1 / Phase 2 happy paths require the GUI / a real DLC
  project and stay on manual / integration testing.
- `dustrack/dlcinterface.py`: `DUSTrack._adopt_layer(ann_or_fname,
  *, set_active=False, set_overlay=None)` -- in-session layer-add
  helper that's the single entry point for layer additions which
  bypass `VideoFileManager`. Resolves the layer name via
  `VideoFileManager.canonical_layer_name` (ignoring any caller-set
  `.name`), adds via `add_annotation_layers({name: fname})`, runs
  `_normalize_dlc_layer_display(scope=[name])` for `dlc_*` layers,
  and sets statevars per kwargs. Idempotent: re-adopting a layer
  already in `self.annotations.names` returns `None` and is a no-op.
- `tests/test_canonical_layer_name.py` -- pin the
  `canonical_layer_name` dispatch matrix: manual annotations
  (incl. multi-token suffix, iteration suffix, buffer suffix,
  LK-on-manual), DLC traces (`.h5`, `.json`, LK-on-dlc), bare-file
  fallback, `str` / `Path` argument acceptance. 12 tests.
- `dustrack/dlcinterface.py`: **Save annotation as...** button +
  `DUSTrack.save_annotation_as` method. Opens a Qt `QFileDialog`
  seeded with the video's folder and a suggested filename of
  `<video_stem>_annotations_<layer>.json` for the active layer.
  Falls back to `self.ann.save()` (writes to the layer's existing
  `.fname`) on non-Qt backends.
- `dustrack/dlcinterface.py`: **Swap layers** button +
  `DUSTrack.swap_active_and_overlay` method. Swaps the
  `annotation_layer` (foreground) with `annotation_overlay`
  (background); no-op when no overlay is selected. Positioned as
  the last sidebar button so it sits immediately above the state
  variables widget it manipulates.
- `tests/test_rewire_to_in_project_paths.py` -- 6 tests covering the
  rewire dispatch matrix: `self.fname` switches to the in-project
  video; JSON outside project tree → migrate (`.fname` / `.fstem`
  rewritten, in-memory data persisted); paths already inside the
  project → no-op; `.h5` outside project → skipped (DLC traces only
  live inside projects); empty layers → migrate path-only without
  preemptive `save()`; layers with `.fname == None` → skipped.

### Fixed
- Ctrl+C (Copy to clipboard) on the DUSTrack figure window now
  copies the entire window — sidebar + image pane + trace canvas —
  instead of only the matplotlib portion. Fix lives in
  datanavigator 1.4.0rc2's `GenericBrowser.copy_to_clipboard`
  (switched to `QMainWindow.grab()` from `figure.savefig`); DUSTrack
  picks it up by version pin.
- `dustrack/dlcinterface.py`: Reduce jitter was producing a layer
  named `'<datanavigator.pointtracking'` with empty `.data` whenever
  it was clicked (worked example: cached LK output already present
  in the session, re-clicked from the in-project video). Root cause:
  `_adopt_layer` did `isinstance(ann, VideoAnnotation)` against the
  *dustrack* subclass, but `lk_moving_average_filter` returns the
  parent `datanavigator.VideoAnnotation`, so the check silently fell
  through to `fname = str(obj)` -- the Python default repr of the
  VideoAnnotation, parsed as a Path stem. Fix: relax the isinstance
  to `dnav.VideoAnnotation` and re-promote the freshly-added layer
  to the dustrack subclass post-add (mirrors the bulk promotion
  `__init__` already does for layers loaded at startup).

### Notes
- Programmatic callers that need the return value
  (`tracker.process_with_lk()` in a notebook,
  `tracker.create_dlc_project()` in a script) should run on a
  non-Qt backend, or — if running in Qt — read
  `tracker.annotations[...]` / `tracker._dlcproject` after the
  Done button is clicked. The Qt async path returns ``None`` (or
  ``self`` for Train DLC) by design.

## [1.0.0] - 2026-05-17

Audit-and-polish release. No public API changes. Drops the alpha tag,
re-points the README + bibtex from the arXiv preprint to the published
*Scientific Reports* paper (Namburi et al., 2026,
[doi:10.1038/s41598-026-42795-3](https://doi.org/10.1038/s41598-026-42795-3),
accepted 2026-02-27), and folds in mechanical cleanup along the lines
of datanavigator 1.2.0. The `numpy<2` pin and `<=3.13` upper bound stay
in place — both are slated for relaxation in a CI-gated follow-up.

### Fixed
- `dustrack/dlcinterface.py`: removed a duplicate
  `from skimage import io, img_as_ubyte` that was masked by the first
  import a few lines up.
- `dustrack/dlcinterface.py`: the ImportError path for missing
  `deeplabcut` now calls `warnings.warn(...)` instead of `print(...)`,
  so library import no longer writes unsolicited text to stdout when
  `HAS_DLC=False`.
- `dustrack/dlcinterface.py`: `process_dlc_project` had an
  `assert self._dlcproject is not None` immediately followed by an
  `if self._dlcproject is None: raise ValueError(...)`; the assert was
  unreachable and has been dropped. The `ValueError` branch is now also
  evaluated *before* `plt.close(self.figure)` so a stale-state caller
  doesn't lose its figure.
- `dustrack/dlcinterface.py`: removed four leftover developer-debug
  `print` calls (`from_video` h5 path, `copy_annotations` filename,
  `process` iteration counters, `annotate` video name). User-facing
  status prints in extraction / training / merge paths are unchanged.
- `dustrack/postprocess.py`: `lk_moving_average_filter` docstring no
  longer contradicts the signature — `video_name` is documented as
  optional with the path-argument requirement asserted at call time.

### Changed
- `dustrack/dlcinterface.py`: import-alias hygiene per the portfolio
  convention settled 2026-05-13/14 — `from pyfilemanager import
  FileManager` → `import pyfilemanager`; `import datanavigator` →
  `import datanavigator as dnav`. Stdlib imports (`re`, `warnings`,
  `PureWindowsPath`, `PurePosixPath`) are now grouped with the other
  stdlib imports at the top of the module instead of trailing the
  `try/except ImportError` block. No external API surface change.
- `pyproject.toml`: dropped `Python :: 3.7` and `Python :: 3.8` from
  the classifier list. The dependency stack (DLC3 + datanavigator
  1.3.x + scientific Python) does not actually support either, and the
  classifiers were misleading on PyPI. `requires-python = ">=3.7,
  <=3.13"` is unchanged for this release (CI matrix + bound relaxation
  is the next-band item).
- `README.md`: citation paragraph and bibtex block now point at the
  Nature *Scientific Reports* article rather than the arXiv preprint.
- `docs/conf.py`: `release` bumped to `1.0.0`; stale `# pyfilemanager`
  comment dropped.
- `docs/index.md`: master-file comment header switched from
  `pyfilemanager` (boilerplate copy-paste) to `dustrack`.

## [1.0.0a2] - 2026-05-17

Maintenance release. Fixes a packaging bug in 1.0.0a1 that silently
disabled DeepLabCut integration on every fresh install from PyPI,
swaps the video-reading backend from `decord` to
`datanavigator.VideoReader` (PyAV+TOC), and ships the GUI / workflow
additions accumulated since 1.0.0a1 (ultrasound image enhancement,
dark theme, source-model wiring). No API breakage; the Qt rework
slated for the next release lives on the `1.4.0-qt` branch and is
not part of this release.

### Fixed
- `dustrack/dlcinterface.py`: removed the stale
  `from . import imagesimilarity` import that landed inside the
  `try`-import-`deeplabcut` block in 1.0.0a1. The `imagesimilarity`
  module was never shipped to PyPI — it lives only on the
  `wip/labeled-data-nudging` branch — so the import fell into
  `except ImportError`, silently setting `HAS_DLC=False`, hiding
  `DLCProject`, and disabling the `Create DLC Project` /
  `Train DLC model` / `Reduce jitter` GUI buttons. Fresh PyPI installs
  of 1.0.0a1 could not exercise the DLC pipeline at all.

### Changed
- Video reading now flows through `datanavigator.VideoReader`
  (PyAV+TOC backend) instead of `decord`. The change rides
  datanavigator 1.3.0's decord→PyAV+TOC swap; public reader API is
  preserved (decord-shaped indexing, `get_batch([...]).asnumpy()`,
  `len()`, iteration, `get_avg_fps()`). See the datanavigator 1.3.0
  changelog for the parity-test motivation.
- `datanavigator` pin: `>=1.3.0` → `>=1.3.0,<1.4.0`. Floor matches
  the PyAV+TOC reader requirement; ceiling locks out the upcoming
  datanavigator 1.4.0 Qt API, which the `1.4.0-qt` branch in this
  repo targets as a separate forthcoming release.
- Retired references to the internal NAS IP / hard-coded paths in
  the dustrack module.

### Added
- `enhance_ultrasound_image()` helper in `dustrack/dlcinterface.py`
  (CLAHE + gamma + brightness adjustments for B-mode ultrasound).
- `DUSTrack` GUI: optional dark theme; gamma-correction toggle;
  source-model path support for DLC project creation.
- Documentation expansions in `dustrack/__init__.py` and the GUI
  docstrings; Sphinx now picks up docstrings of optional components.

### Removed
- The `pia02_workflow/` directory has been moved off the `main`
  branch into the `pia02-sandbox` branch (`git subtree split`, 63
  commits of history preserved). The sandbox is technician-led
  piano-study scratch (see `specs/dustrack.md` Open questions in the
  pn-portfolio repo) and was not intended to live on `main`. It will
  eventually graduate to `pn-projects/projects/mithic/pia02/`. This
  is a repository-layout change only — the PyPI wheel never included
  `pia02_workflow/` (flit scopes packaging to the `dustrack/` import
  package), so installed-user behavior is unaffected.
