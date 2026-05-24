# Change Log
All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed
- **Detect blip outliers button no longer gates on >=80% per-label
  coverage.** Real DLC predicted traces frequently have NaN gaps
  where the model bailed, and those drop per-label coverage below
  80% on otherwise-valid layers — the button would stay greyed out
  even after switching the active layer to a dense `dlc_*` trace.
  The detection algorithm itself is well-behaved on sparse input
  (MAD threshold falls back cleanly; modal surfaces `0 blips found`
  for too-sparse cases), so the coverage check was just keeping
  useful work off-screen. The gate now disables only for the
  degenerate cases (no active layer / no labels, or the layer is
  already a `*_blip_corrections` output). Surfaced from a user
  session on the s006/RFA pia02 project.
- **`s` key on `.h5` layers no longer silent no-op.** Pressing `s` on a
  DLC trace / `labeled_data` `.h5` layer previously called
  `VideoAnnotation.save()`, which raises `ValueError("Supply a json file
  name.")` on non-JSON suffixes; the exception was swallowed by
  matplotlib's key-event dispatcher and the user saw nothing happen.
  `DUSTrack.save()` now short-circuits `.h5` cases with a printed
  message naming the layer and pointing at `Save annotation as...`
  (sidebar) for explicit copy-out. 4 new tests in
  `tests/test_save_h5_noop.py`. Surfaces a pia02 mid-session UX gap
  (2026-05-23).

### Added
- **Detect blip outliers — sidebar button + two-stage modal +
  ProgressOverlay (UI wiring on top of the existing `detect_blips`
  / `interpolate_blips` API).** New **Detect blip outliers** button
  in the Workflow group (right after Reduce jitter, before Save
  annotation as...) that pops `BlipOptionsDialog`. Stage 1: tune three
  detection knobs (threshold factor / max blip length / return
  tolerance) + click **Detect** to run detection synchronously in-modal
  (~0.14 s on a 36715-frame pia02 trace); per-label counts +
  thresholds + length histogram populate. Re-Detect with new knobs to
  iterate. Stage 2: **Interpolate** closes the modal and kicks off the
  slow LK pass under a `ProgressOverlay` (one tick per blip via the
  new `progress_callback` kwarg on `interpolate_blips`); the sparse
  blip-corrections layer adopts as active with the source DLC trace
  pinned as overlay (mirrors Reduce jitter's adoption shape).
  Overwrite-confirm overlay fires when a `<stem>_blip_corrections.json`
  already exists alongside the source. mpl-fallback path runs detect +
  interpolate synchronously with module defaults (no modal, no
  overlay), preserving the headless workflow. Workflow gate disables
  the button when the active layer is sparse (per-label coverage
  < 80%, per [[gate-on-data-not-naming]]) or when the active layer is
  already a blip-corrections output. New `BlipReport.min_coverage()`
  helper exposes the gate-relevant statistic on the algorithm side.
  12 new tests in `tests/test_blip_modal.py` (result-text renderer +
  mpl-fallback workflow dispatch + progress-callback semantics +
  coverage helper + gate matrix). Manual Qt smoke harness at
  `tests/qt_learning/31_blip_modal_smoke.py`.
- **`dustrack.detect_blips(ann)` + `dustrack.interpolate_blips(ann, report)`
  + `dustrack.detect_and_interpolate_blips(ann)`** — sparse-blip outlier
  detection on dense-trace annotations (typically DLC `.h5` predicted
  traces) and per-label LK-RSTC re-tracking across detected blips.
  A *blip* is a short run of frames where the labeled point jumps away
  and returns — the signature of a model picking the visually-strongest
  answer for one or a few frames before snapping back to the
  temporally-consistent lane. Per-label robust threshold
  (`med + factor * 1.4826 * MAD`, with midpoint fallback for flat-with-
  spikes traces); return-tolerance scales with run length. Interpolation
  delegates to the existing `lucas_kanade_rstc` helper, anchored to the
  surrounding good frames; sparse `VideoAnnotation` output containing
  only the blip frames for blipped labels (other labels in the same
  frame untouched, per spec). Suitable as DLC training data via the
  NaN-tolerant labeled-data pipeline. Sequential per-blip; reuses one
  `VideoReader` across blips (network-drive open cost). Saves to
  `<source_stem>_blip_corrections.json` next to the source; refuses to
  overwrite an existing file. New module `dustrack/blip.py`; new
  exports `Blip`, `BlipReport`, `detect_blips`, `interpolate_blips`,
  `detect_and_interpolate_blips`. 15 new tests in `tests/test_blip.py`
  (synthetic detection + LK-on-example-video interpolation +
  round-trip). Real-data smoke at `tests/qt_learning/30_blip_demo.py`:
  on `pia02_s001_007_RFA2` (36715 frames × 2 labels), detection runs
  in 0.14s and surfaces 1676 blips; interpolation runs in 137s. Roadmap
  item 5 in *Next (general-model workflow features)*; independent of
  the DINOv3 infrastructure (#1).
- **`dustrack.batch.build_toc(sources, *, extensions, recursive, force,
  show_progress)`** — pre-build the PyAV+TOC sidecar
  (`<video>.dnav-toc`) for every video under `sources`. Thin pass-through
  to `datanavigator.precompute_toc_folder` so DUSTrack callers can stay
  inside the `dustrack` namespace. Accepts a directory, a video file, or
  an iterable mixing both. First DUSTrack open of a video pays the
  per-file TOC build cost (a full sequential demux to record per-packet
  offsets + per-frame timestamps); pre-building means `dustrack.open(...)`
  returns essentially instantly afterward. Use case: warming the pia02
  master corpus at `M:/us_videos_for_tracking2/` (1627 mp4s) before an
  annotation session.
- Both functions also re-exported at the package root
  (`dustrack.build_toc`, `dustrack.convert_to_mono`); the
  `dustrack.batch` submodule is now part of the public surface. New
  tests in `tests/test_batch_toc.py`.
- **`convert_to_mono(show_progress=False)`** — optional tqdm bar over
  the per-file loop, matching `build_toc`'s `show_progress` kwarg.
  Per-file status lines route through `tqdm.write` when the bar is
  active so it stays clean. Off by default to preserve the historical
  print-only behaviour for shell users. 3 new tests in
  `tests/test_convert_to_mono.py`.
- **`convert_to_mono` / `build_toc` gain `progress_callback` and
  `cancel_check` kwargs** — per-file callback (`(idx, total, path,
  status)`) and a between-files cancel hook. These power the new
  batch-process modal without forcing CLI users to install tqdm or
  wire up signals. On `build_toc` the kwargs trigger a per-file loop
  (one `dnav.precompute_toc([fp])` call per video) instead of the
  bulk `precompute_toc_folder` delegation so progress + cancel fire at
  file granularity.
- **Batch-process modal (Qt overlay)** — clickable surface for
  `convert_to_mono` + `build_toc`. Folder picker, two operation
  checkboxes, Run + Cancel, progress bar + last-N status feed,
  QThread worker with a between-files cancel via `threading.Event`.
  Same backdrop + parented-QFrame scaffolding as the welcome / confirm
  overlays. Reachable two ways:
  - **"Batch process..." button on the welcome modal** — secondary
    action below Open/Load. The welcome modal exits with the sentinel
    string `"batch_process"`, the dispatcher in `dustrack._open`
    opens the batch modal on the same seed window, then re-mounts
    the welcome modal when the batch modal closes.
  - **Tools menu on the main DUSTrack window** — `Tools → Batch
    process...`. Lets users with a real session open warm a sibling
    folder without relaunching `dustrack.open()`. No-op when the host
    isn't a QMainWindow (mpl fallback / headless).
  Dispatcher logic extracted into `dustrack._batch_modal.run_batch_jobs`
  for unit-testability; new tests in `tests/test_batch_modal.py`.
  DLC projects: no special-case wiring. Point the folder picker at
  `<project>/videos` to TOC the in-project copies; nested per-session
  subfolders under `videos/` are not walked (DLC's `copy_videos=True`
  produces a flat layout).

## [1.2.0] - 2026-05-23

First minor release on top of 1.1.0. Three months of development
land as a single PyPI cut, consolidating the locally-tagged
`v1.2.0a1` / `v1.2.0a2` / `v1.2.0a3` checkpoints (none published)
plus the `1.2.0rc1` structural refactor band plus a tail of polish.

Per-arc detail is preserved verbatim in the `[1.2.0aN]` /
`1.2.0rc1` narrative sections below; the bullets here are the
executive summary of what changed *visibly* between 1.1.0 and 1.2.0.

### Headline arcs (chronological)

- **`1.2.0a1` (2026-05-20) — datanavigator boundary refactor**: the
  point-tracking UI (`VideoPointAnnotator`), annotation containers
  (`VideoAnnotation`, `VideoAnnotations`, `_TrackedFrameDict`), and
  Lucas-Kanade helpers (`lucas_kanade`, `lucas_kanade_rstc`)
  relocated from `datanavigator` to `dustrack` via `git filter-repo`
  with full history preserved. DUSTrack now owns its labeling UI +
  DLC workflow end-to-end. dnav floor raises to `>=1.5.0`.
- **`1.2.0a2` (2026-05-21) — pia02 workflow features part 1**:
  cold-open performance (2.69× cumulative on the
  `interosseous_pn24-x` 12-bundle session: vectorised DLC-trace
  conversion + shared `VideoReader` across annotation layers); LK
  performance (decode-once RSTC 5.44×, parallel `lk_moving_average_filter`
  with bounded-inflight executor + sum-and-count `save_raw=False`
  mode 10-12× peak memory cut); DLC training UI controls
  (`DLCProject.train_iteration` explicit-args sibling of
  `process()`, three `refine_mode` paths, training options modal);
  seed-from-snapshot-bundle flow (`dustrack.import_seed_bundle_into_project`
  + `dustrack.extract_snapshot_for_seeding` etc., Create-DLC-project
  modal seeding when active manual layer is empty); zero-arg launch
  (`dustrack` console script + Qt picker + cross-session recent-videos /
  recent-folders history); lazy `import deeplabcut` cuts
  `import dustrack` 8.45 → 2.83 s (~3×).
- **`1.2.0a3` (2026-05-22) — pia02 workflow features part 2**:
  in-session multi-video navigation (`Alt+Left` / `Alt+Right` arrow
  swap between every video in a DLC project, per-bundle state
  persistence across swaps, hybrid sync/async hydration with a
  `_HDF5_LOCK`-serialised data half + Qt-thread artist-setup
  poller, 14 ms round-trip swap on 12 hydrated bundles); seed-window
  welcome modal at no-arg launch (`dustrack.open()` with no path
  pops a `OpenVideoOverlay` mounted on a tiny seed-mode DUSTrack
  that swaps in-place to the user's pick via `replace_active_with`);
  rendering / paint trade-off (`flush_events()`-only in
  `DUSTrack.update` + `_show_first_paint_notice` modal at multi-video
  init whose OK-click drains the paint queue, 22.25 ms / 45 fps
  steady-state on the multi-video bench).
- **`1.2.0rc1` (2026-05-22 → 2026-05-23) — structural refactor**:
  `dlcinterface.py` split from ~9700 LOC into focused modules
  (Phase 0 leaf renames: `opticalflow` → `lk_opticalflow`,
  `postprocess` → `lk_filter`, `convert` → `batch`, `_dlc_patch`
  → `dlcpatch`; Phases A-E + follow-ups: extract `_layer_names`,
  `_image_enhance`, `_workflow_gates`, `_view_state`, `_qt_styling`,
  `_close_guard`, `_nav_widget`, `_preflight` + `_preflight_modal`,
  `_seed_bundle_modal`, `_train_modal`, `_overlays`,
  `_file_management`, `_bundle`, `gui`, `_open`, `annotations`,
  `_dlc_paths`; `_DUSTrackBase` collapsed into `DUSTrack`;
  `dlcinterface.py` shrunk to ~1700 LOC, holds only `DLCProject`
  + `DLCData` + the lazy-DLC `__getattr__` proxy + the
  `_RELOCATED_NAMES` shim for back-compat). Import sweep + black +
  `__all__` declaration on the public surface; three latent
  undefined-name bugs fixed (missing `plt` import in `_open`
  seed-modal paths, missing `import_seed_bundle_into_project`
  in seeding flow, missing `DLCProject` type ref in
  `_file_management`).
- **Tail polish (2026-05-23)**: frame-level **Decimate annotations**
  feature (prune incomplete frames in the selected interval then
  drop every other complete frame; `x` keybinding + Niche-group
  button; starter form of the general-model workflow's
  decimation-modal feature pending DINOv3 image-similarity
  infrastructure). Sidebar regroup from five task-flow groups
  (Workflow / Display / Niche / Utilities / Swap) to four
  (Workflow / Display / Niche / Swap): Refresh UI + Keyboard
  shortcuts move into Display (visual controls + UI utilities);
  Decimate + Discard unsaved + Replace existing + Remove layer
  form the Niche cluster (all layer-mutating affordances).

### Dependency floor

`datanavigator>=1.5.0a1` → `>=1.5.0`. dnav 1.5.0 shipped to PyPI
2026-05-23 with the relocated `pointtracking` / `opticalflow`
modules removed; in-process upgrades from a 1.4.0-based env should
upgrade dnav first.

### Public API removals (no shim)

The Phase 0 renames removed the old module names with no
back-compat aliases:

- `dustrack.opticalflow` → import `dustrack.lk_opticalflow`
- `dustrack.postprocess` → import `dustrack.lk_filter`
- `dustrack.convert` → import `dustrack.batch`
- `dustrack._dlc_patch` → import `dustrack.dlcpatch`
- `dustrack.pointtracking` (1.2.0a1 destination) → contents in
  `dustrack.annotations` after Phase E; the parent class
  `VideoPointAnnotator` collapsed into `dustrack.DUSTrack`.

The portfolio sweep at refactor time confirmed every external
caller uses the top-level `dustrack.X` surface (no `from
dustrack.opticalflow import ...` sites in `pn-projects/`,
`immersionToolbox/`, or `datanavigator/tests/`), so the API
break is invisible to in-tree consumers. External users of the
old paths re-pickle / update import sites.

## [1.2.0a3] - unreleased

In-session multi-video navigation (Roadmap *Next 1.2.0* item 3). A
single DUSTrack `QMainWindow` now hosts a queue of bundles, one per
video, with a `◀ k/N ▶` nav row at the top of the sidebar and
`Alt+Left` / `Alt+Right` key bindings. The active session swaps in
place — no figure teardown, no window rebuild. Per-video state
(active layer, overlay, current frame, active label, label range,
frozen axes, trace pane pan/zoom, image-pane zoom + pan, CLAHE +
gamma + brightness, unsaved edits, frames-of-interest) persists
across swaps: swap-back returns to the exact visual state the user
left. The one statevar that broadcasts across bundles is
`number_keys` (select/place mode — UI-mode, video-agnostic).

Strict-single-DLC-project contract for this cut: every video in a
multi-video session must belong to the same DLC project. Two entry
shapes:

- **`dustrack.open(project_folder)`** queues every video in
  `project.config['video_sets']` (project order). Behavior change
  from <=1.2.0a2 which opened only the first video.
- **`dustrack.open([v0, v1, ...])`** queues exactly those videos
  in the given order; every entry must resolve to the same project.

Bare-video lists, mixed-project lists, and `config.yaml` paths all
raise `ValueError` with an actionable message. Phase 1 (no DLC
project) multi-video, mixed-mode queues, and ad-hoc
`add_videos`-on-the-fly are out of scope for 1.2.0a3.

The active bundle (video 0) opens synchronously so the user can
start working immediately; bundles 1..N are hydrated by a daemon
background worker in parallel. Worker pipeline: off-thread data
half (VideoReader open, `<video>_annotations*.json` discovery, DLC
`.h5` reads, VideoAnnotation construction with empty axis lists)
→ thread-safe finalisation queue → Qt-thread poller (`QTimer` on
the QMainWindow, 50 ms) drains the queue and runs the artist-setup
half (per-layer `add_marker_group` on the image pane, trace-axis
binding, immediate `ann.hide(draw=False)` to park). PyTables is
not thread-safe so all HDF5 reads serialise behind a module-level
`_HDF5_LOCK`. End-to-end smoke (12-video pia02 s006 project on the
M: drive): construction 7 s, all-bundles-hydrated 15 s, swap-back
to a previously-visited bundle 14 ms.

Swap mechanics (`DUSTrack.swap_to(index)`):

1. **Snapshot active bundle**: current frame, `_ax_lims` (freeze
   state), image-pane viewport (Tier 2 reads QGraphicsView's
   transform + scrollbars via the new dnav `_QtImagePane.get_view_state`
   surface; Tier 1 falls back to `_ax_image.get_xlim/ylim`),
   frames-of-interest, all 5 statevar selections.
2. **Park leaving bundle**: `ann.hide(draw=False)` on every
   annotation in the leaving container — artists stay attached
   to the shell's axes but `set_visible(False)`'d, so swap-back
   re-shows them instantly (Model D artist parking; no data
   re-read).
3. **Rebind shell**: `self.fname`, `self.data` (VideoReader),
   `self.annotations`, `_current_idx`, `_ax_lims`,
   `frames_of_interest` all point at the arriving bundle.
4. **Show arriving artists**: `ann.show(draw=False)` on every
   annotation.
5. **Restore statevars silently**: rewrites each statevar's
   `states` list to the new bundle's rotation, sets selection
   via direct `_current_state_idx = ...` to bypass the
   `_notify_change` callback chain (otherwise on_change cascades
   would fire mid-restore against partially-rebuilt state).
   `statevariables._text.update()` syncs the Qt sidebar widgets.
6. **Restore image-pane viewport** via the dispatch above.
7. **Single repaint** via `self.update()` + `plt.draw()`.

Only `number_keys` (select/place mode) auto-broadcasts across
every bundle's `selections` dict when the user mutates it via UI
/ keybinding. `annotation_label` and `label_range` are
per-bundle: each video remembers its active label independently,
so "label 1 on video A" doesn't bleed into "video B" on swap-out.
`annotation_layer` / `annotation_overlay` are per-bundle for
data-shape reasons. The silent-bypass restore avoids triggering
the broadcast hook during a swap-in (which would otherwise
clobber every bundle's just-restored value with the active
bundle's).

Save-on-close sweeps every ready bundle, not just the active one
(`_scan_unsaved_layers_all_bundles`). The modal renders unsaved
diffs grouped by video. Cancel still keeps the window open.

Train DLC's post-success `_refresh_dlc_layers` propagates the new
`dlc_*` layers to every ready non-active bundle (the underlying h5
files were written for every video in the project, so swapping to
bundle k+3 immediately shows the fresh inference instead of needing
a session restart). Pending bundles will discover the new files
naturally when the worker reaches them.

### New API

- `DUSTrack.swap_to(index: int) -> bool`,
  `DUSTrack.swap_prev()`, `DUSTrack.swap_next()`.
- `dustrack.open(project_folder)` — multi-video dispatch.
- New `dustrack._bundle._BundleState` dataclass (internal).
- New `dustrack.dlcinterface._BgHydrationWorker` (internal).

### New dnav surface (folded into the open 1.5.0a1 changelog)

- `datanavigator._qt._QtImagePane.get_view_state()` /
  `.set_view_state(state)` — opaque-blob round-trip of the
  QGraphicsView transform + scrollbar positions, for per-bundle
  image-pane viewport persistence. Dnav floor stays at the existing
  `>=1.5.0a1` (the additive methods don't require a version bump).

### Rendering / paint behavior (2026-05-22 follow-up)

Earlier 1.2.0a3 cuts saw `plt.draw()` (deferred via
`canvas.draw_idle()`) silently failing in interactive Qt sessions
— after a click or swap, the data state updated correctly but the
trace pane stayed stale until the user resized the window /
clicked the zoom tool / picked something from a dropdown. Two
root causes, both addressed:

1. **Bg-hydration worker QTimer polled forever** (50 ms interval,
   even with an empty queue once all bundles were hydrated). It
   competed with paint events on the Qt event loop. Worker now
   auto-stops its timer in `drain_finalisation_queue` once every
   bundle is terminal.
2. **The canvas needed a synchronous warm-up before
   `canvas.draw_idle()` would fire reliably.** Two warm-up sites:
   `QTimer.singleShot(0, canvas.draw)` at the end of
   `_init_bundles` (first paint after open — `singleShot(0)` so
   the warm-up fires *after* `plt.show(block=True)` starts the
   event loop, not before) and a synchronous `canvas.draw()` at
   the tail of `swap_to` (first paint after a swap). Once warm,
   subsequent `plt.draw()` calls flush correctly through the
   event loop's natural drain.

Shipped fix is a one-line `self.figure.canvas.flush_events()` in
`DUSTrack.update` after `super().update()`. On QtAgg
`flush_events()` is a thin wrapper over
`QApplication.processEvents()` and forces the queued
`canvas.draw_idle()` to run without scheduling a second full
render the way `canvas.draw()` would.

**Known multi-video limitation**: on first visit to a bundle after
open, the trace pane can stay stale until the user flips videos
once (`Alt+Right` then `Alt+Left`, or the sidebar nav buttons) —
after which `draw_idle` starts firing reliably. Two cheaper
warm-up variants (counter-gated first-update `canvas.draw()`,
plain `flush_events()`) were tried and neither fixed the
interactive bug; an unconditional `canvas.draw()` fixes it but
carries a ~3× per-frame cost (76 ms / 13 fps vs 22.6 ms / 44 fps
on the multi-video bench). Shipped as the lesser of two evils
until the root cause of `draw_idle` starvation is diagnosed —
top entry on the 1.2.0a3 perf-profiling agenda.

Steady-state per-frame bench
(`tests/qt_learning/28_benchmark_multi_video_update.py`, 12-bundle
pia02 s006 session, 220 Line2D on `_ax_trace_x`, continuous frame
walk with 15-frame warmup dropped — methodology mirrors probe 09 /
probe 14):

| | min | median | mean | p95 | fps (median) |
|---|---|---|---|---|---|
| 1.5.0 fast_render single-video (probe 14) | — | **36.0** | 36.0 | 37.2 | 28 |
| 1.2.0a3 multi-video, flush_events (shipped) | 21.2 | **22.6** | 22.9 | 25.0 | **44** |
| 1.2.0a3 multi-video, unconditional canvas.draw() (rejected) | 74.5 | 76.2 | 76.5 | 80.0 | 13 |

Shipped variant is net no-op vs single-video despite the 220-line
trace pane: `_revision`-keyed trace cache hits on every non-first
frame, the only per-update cost is the image decode + frame-marker
reposition + base-class `plt.draw()`.

### Sidebar nav row gains a video-name dropdown (2026-05-22 follow-up)

The central `1 / N` label in the `◀ … ▶` nav row at the top of the
sidebar is now a `QComboBox` listing every bundle as
`"i. <stem>"` (1-based) so the user can both *see which video they
are working on* and *jump directly to any other video in the
session*. Per-item tooltip carries the full path; the combo's own
hover tooltip shows the path of the currently-displayed video.
Hydration progress that used to render as the
`(X ready)` suffix now appears as per-item markers
(`…` pending/hydrating, `✗` failed, blank = ready) so the same
information stays visible without a separate label.

The ◀ / ▶ buttons and `Alt+Left` / `Alt+Right` key bindings work
unchanged; programmatic syncs (`setCurrentIndex`) are wrapped in
`blockSignals` and the dropdown listens on `activated[int]` (user
interaction only) so a sequential nav doesn't re-enter `swap_to`.
`_sync_nav_combo` rebuilds the items list only when the bundle
fnames change (signature-keyed); bg-hydration progress ticks take
the cheap in-place per-item update path.

### Tests

53 new tests across `tests/test_bundle.py`,
`tests/test_swap_to.py`, `tests/test_open_multi_video.py`,
`tests/test_bg_hydration_worker.py`,
`tests/test_broadcast_statevars.py`, plus updates to
`tests/test_open_zero_arg.py` (bare-video multi-video lists now
raise per the strict-single-project contract),
`tests/test_save_on_close.py` (stubs updated to the multi-bundle
sweep API), `tests/test_user_config_recent.py` (stubs use
`_bundles` instead of the legacy `_video_queue`). The 2026-05-22
dropdown follow-up rewrote the four `TestRefreshNavButtons` tests
in `test_swap_to.py` against an in-process `_StubCombo` (replaces
the prior `_StubLabel`) and added one no-rebuild-on-progress-tick
case (5 cases total). End-to-end smoke in
`tests/qt_learning/26_smoke_multi_video.py` (programmatic),
`27_visual_smoke.py` (screenshot harness), and
`28_benchmark_multi_video_update.py` (per-update paint cost).
Full DUSTrack suite: 509 passed, 1 skipped.

Plan archive at `pn-portfolio/plans/20260521_dustrack_1.2.0a3_multi_video_swap.md`
(continued through 2026-05-22).

---

**Seed-window + welcome modal (2026-05-22)**. `dustrack.open()` with
no path (the no-arg CLI form `dustrack`) now constructs a tiny
seed-mode `DUSTrack` against a packaged synthetic video and mounts a
welcome modal on top, instead of popping the legacy `QFileDialog`
directly. The user picks via a "Choose video..." button (forwards to
the same `QFileDialog`) or a clickable "Recent sessions" list fed
from the unified `recent_sessions` history. On pick, the active
bundle swaps in-place via the new `DUSTrack.replace_active_with`;
no window teardown, no figure rebuild.

The transition is powered by three new public methods that
generalize the 1.2.0a3 multi-video machinery:

- **`DUSTrack.add_video(path_or_paths, *, layer_name=None, set_active=False, **kwargs)`** —
  validates, hydrates, and appends one or more new bundles to a live
  tracker. Phase 1 (bare-video) and Phase 2 (single video in a DLC
  project, or multi-video in one project) all routed through the
  same surface. Multi-video adds queue the tail as `PENDING` and
  spawn a `_BgHydrationWorker` (same daemon machinery the
  multi-video launch uses). Returns the new bundles' indices.
- **`DUSTrack.remove_video(index)`** — drops a bundle from the list.
  If the index is the active bundle, swaps to a sibling first
  (next-in-line, falls back to prev at the tail). Refuses to empty
  the list. Surviving bundles are renumbered so `video_index`
  matches the new list position.
- **`DUSTrack.replace_active_with(path_or_paths, **kwargs)`** —
  composition of `add_video(..., set_active=True)` + `remove_video`
  of the old active bundle. Used by the seed-modal flow but
  reusable for any "switch what's in the active slot" UX (planned
  future affordances: an "Open recent" submenu inside an active
  session, an "Add video" sidebar button).

Bundle-state extension: `_BundleState.project` field added (Phase
1 bundles store `None`; Phase 2 store the shared `DLCProject`). The
swap contract grows by one rebind step — `DUSTrack._attach_bundle`
now pushes the arriving bundle's project onto `self._dlcproject` so
Workflow-button gating reads the right value across phase
transitions.

Cross-session history rework: the pre-1.2.0a3 split
(`recent_videos: list[str]` + `recent_folders: list[str]` JSON keys)
collapses into a unified `recent_sessions: list[list[str]]` shape
where each entry is the full bundle list of one session (1-element
for single-video, N-element for multi-video, the unified active
path leading). One-time migration on first read folds the legacy
keys into the new list and drops them from disk. Back-compat
accessors (`record_recent_video` / `get_recent_videos` /
`record_recent_folder` / `get_recent_folders`) project the unified
list down for legacy callers. Cap dropped 25 → 20 to keep the
modal's recent list short enough to scan at a glance.

Close-guard short-circuits on `_is_seed_session = True`: the seed
tracker never prompts to save on close and never writes its
synthetic asset path to `recent_sessions`, even if the user
accidentally interacts with the seed image before dismissing.
Defensive fallback: if the packaged seed video fails to load
(corrupt asset, codec mismatch, headless / no Qt window),
`dustrack.open()` falls through to the legacy direct-picker flow
unchanged.

Asset: `dustrack/_data/seed_video.mp4` (8 frames, 64x64 mid-gray
h264, 1.7 KB) + co-shipped `seed_video.mp4.dnav-toc` sidecar so the
first-launch open skips dnav's TOC scan. Both ship via the existing
flit package-data auto-include. Regeneratable via
`tests/_assets/build_seed_video.py`.

Tests: 1.2.0a3 baseline 436 → 555 (+119 this cut: 33 history
migration + 16 seed-modal dispatch + 12 seed-asset + overlay
rendering + 22 bundle-API). Manual Qt smoke confirmed
`_open_seed_session` + `replace_active_with(picked)` against a copy
of the seed asset.

**Seed-window follow-ups (2026-05-22 same-day)**. Three fixes from
first-use feedback on the welcome modal:

- **Explicit Load button.** The modal's single-click commit (Choose
  video... + recent-row click → instant load) was too easy to fire
  by accident. The Browse button and the recent-row single-click
  now *stage* a selection; a new primary Load button (disabled
  until something is staged) is the commit point. A "Selected: ..."
  preview line shows the staged pick. Double-click / Enter on a
  recent row is preserved as a stage+commit shortcut for users who
  remember the one-click path.
- **Polluted-history prune.** The packaged seed asset path
  (`dustrack/_data/seed_video.mp4`) is now dropped from
  `recent_sessions` on read (one-time disk prune persisted on next
  write). Pre-fix this leaked into history under some test /
  fallback paths and surfaced as a clickable "session" that
  re-opened the synthetic 64x64 seed video on selection. Defense-
  in-depth: the writer (`record_recent_session`) also rejects the
  seed path, so a future caller wiring its own save path can't
  re-introduce the pollution.
- **Post-swap zoom (datanavigator fix).**
  `_QtImagePane.set_image()` now detects image-dimension changes
  and rebuilds the scene rect + re-fits the view. Pre-fix the pane
  fixed its scene rect on the first frame and never updated it,
  so the seed-modal swap from the 64x64 synthetic video to the
  user's pick left the new frame rendered in the old (64x64) scene
  rect — the loaded video looked extremely zoomed in and `r`
  (reset view) didn't help because `reset_view` re-fit to the
  stale rect. Same-dimension calls (every normal per-frame update)
  stay no-op; cross-dimension calls reset the transform, clear the
  `user_adjusted` flag, and refit. Fix lives in dnav 1.5.0a1
  (still unreleased).

Tests: +21 this follow-up (16 modal staging/commit + 5 prune). End-
to-end suite at 571 passed, 1 skipped.

**Seed-window follow-up #2 (2026-05-22 same-day)**. Five fixes from
the next round of first-use feedback:

- **Picker scope expanded to DLC `config.yaml`.** The file dialog
  now accepts `*.mp4 *.avi ... config.yaml` in its default filter row
  (also offers a videos-only and a config-only row). Picking a
  `config.yaml` lands in the new dispatch below.
- **`dustrack.open('.../config.yaml')` now opens multi-video.** Pre-
  fix the config.yaml-scalar form opened only video 0; that was a
  holdover from the pre-multi-video era. New behavior queues every
  video in `config['video_sets']` in YAML-stored order, mirroring the
  project-folder form. `DLCProject.__init__` runs `rebase_to_config`
  on each video_sets key BEFORE we enumerate via `project.video_list`,
  so a project folder that was renamed since the YAML was written
  self-heals on first open. No backwards-compat shim -- callers
  scripting `open(config.yaml)` for single-video should switch to
  the in-project video path. Multi-element lists that mix
  `config.yaml` with videos now raise with a clear pointer to the
  supported shapes. `DUSTrack._validate_bundle_paths` mirrors the
  dispatch so `replace_active_with([config.yaml])` from the seed
  modal does the same multi-video init.
- **Modal: contextual single-button + history toggle.** The
  Load-after-Choose flow was redundant on the file-dialog path
  (the dialog's own Open click already commits intent). Replaced
  with one button whose label flips: `Open` when no history row is
  selected (clicking pops the file dialog and the dialog's return
  commits straight to load); `Load` when a recent row is selected
  (clicking commits the selected row). Recent-row single-click
  toggles selection -- clicking the same row again deselects.
  Double-click / Enter still commits a row in one gesture. The
  helpful message above the button state-flips too (`Pick a video
  or DLC config.yaml` ↔ `Click Load (or double-click) to open it`).
- **Enhance-state first-visit defaults.** Pre-fix
  `_set_enhance_state(None)` returned early on a swap to a bundle
  with no saved enhance state, which meant the arriving bundle
  inherited the leaving bundle's slider positions (set gamma=1.5 on
  V1, swap to V2 first-visit, V2's sliders showed 1.5 instead of
  the construction default). Fix snapshots construction-time
  CLAHE/gamma/brightness into `_initial_enhance_state` once in
  `__init__`; `_set_enhance_state(None)` now resets to that
  snapshot. Returning visits still use the per-bundle saved state.
- **Modal Qt.UserRole-data shortcut.** Double-click / Enter on a
  recent row now reads the path list straight from the item's
  `Qt.UserRole` data instead of doing a `widget.row(item)` -> index
  -> `_recent_sessions[index]` round-trip. The round-trip raced
  with the Qt event queue under parallel pytest-xdist workers and
  occasionally returned `-1`, dropping the commit.

Tests: +21 this follow-up (10 modal contextual/toggle + 6 config.yaml
dispatch + 3 enhance-defaults + 2 misc). Full suite: 580 passed, 1
skipped.

**`_DUSTrackBase` merged into `DUSTrack` (2026-05-22, 1.2.0rc1
refactor tail)**. The 1.2.0rc1 structural refactor (Phases 0-E)
deferred the `_DUSTrackBase` collapse to a dedicated session because
the 37 tests in `test_pointtracking.py` instantiated the base class
directly. This change closes that follow-up:

- All `_DUSTrackBase` methods + `__init__` body absorbed into
  `DUSTrack` in `dustrack/gui.py` (now ~5,650 LOC; was ~5,110 LOC +
  ~1,500 LOC in `pointtracking.py`). DUSTrack's base class is now
  `datanavigator.videos.VideoBrowser` directly.
- `pointtracking.py` **deleted**. Test imports for `VideoAnnotation` /
  `VideoAnnotations` / `_TrackedFrameDict` updated to
  `dustrack.annotations`. Portfolio sweep confirmed no external
  imports.
- **All pickle-compat `sys.modules` aliases dropped from
  `dustrack/__init__.py`** (the prior `opticalflow` / `postprocess` /
  `convert` aliases as well as the never-published `pointtracking`
  alias). Pickle compat for the renamed-leaf-module paths is
  explicitly out of scope; re-pickle if you hit one. The only
  surviving forwarding shim is `dlcinterface.__getattr__`, which
  bounces `DUSTrack` / `open` / DLC-path-classifier names through
  to their new homes for live `from dustrack.dlcinterface import X`
  callers (not a pickle path).
- `DUSTrack.__init__` signature grew the kwargs that used to live on
  the parent: `n_labels`, `titlefunc`, `height_ratios`, `fast_render`.
  `fast_render` defaults to `True` (Qt-native image pane) for the
  normal interactive path; pass `fast_render=False` for headless /
  Agg test contexts that used to construct `_DUSTrackBase` directly.
- `tests/test_pointtracking.py` rewritten: 37 call sites updated
  from `_DUSTrackBase(...)` to `DUSTrack(..., fast_render=False,
  annotation_names="")` (the old `_DUSTrackBase` default was `""`;
  `DUSTrack` defaults to `"iteration-0"`).
- The default-buttons hook (`_add_default_buttons`) was removed
  entirely — the rc2 sidebar already places `Refresh UI` as a styled
  utility-group button next to `Keyboard shortcuts`, so the parent
  hook had no remaining consumer.
- Benchmark `tests/qt_learning/24_benchmark_cold_open.py` rewired to
  patch `gui.DUSTrack.{add_annotation_layers,add_events,
  set_key_bindings,update}` instead of `pointtracking._DUSTrackBase.*`.
- `docs/keyconcepts.md` inheritance diagram updated; CHANGELOG note
  here.

Tests: full `test_pointtracking.py` (90 cases) passes; suite-wide
green expected to hold (the merged class is bit-identical to the
old subclass on every observable surface besides the `fast_render`
default — and that default only matters for headless contexts that
weren't previously using `DUSTrack`).

**1.2.0rc1 structural refactor: `dlcinterface.py` split into focused
modules (2026-05-22 → 2026-05-23).** `dustrack/dlcinterface.py` had
accreted to ~9700 LOC and bundled too many concerns (DUSTrack GUI
class + DLCProject + VideoFileManager + DLCData + overlay factories
+ `open()` dispatch + image-enhancement helpers + first-paint notice
+ DLC interop). The 1.2.0rc1 band split it into focused modules
without changing observable behavior. Eleven commits land the split
plus a hygiene close-out:

- **Phase 0 — leaf module renames**: ``opticalflow.py`` →
  ``lk_opticalflow.py``, ``postprocess.py`` → ``lk_filter.py``,
  ``convert.py`` → ``batch.py``, ``_dlc_patch.py`` → ``dlcpatch.py``.
  No back-compat aliases for the old paths (the prior
  ``sys.modules`` setdefault block in ``__init__.py`` was dropped
  with the ``_DUSTrackBase`` merge above); re-pickle if you hit a
  baked-in old module path.
- **Phase A — extract leaf helpers from ``dlcinterface``**:
  ``_layer_names.py``, ``_image_enhance.py``, ``_workflow_gates.py``,
  ``_view_state.py``, ``_qt_styling.py``, ``_close_guard.py``,
  ``_nav_widget.py``, ``_preflight.py``, ``_preflight_modal.py``,
  ``_seed_bundle_modal.py``, ``_train_modal.py``. Pure-logic vs
  Qt-modal pairs (``_preflight.py`` + ``_preflight_modal.py``,
  ``seed.py`` + ``_seed_bundle_modal.py``) follow the
  alphabetic-pair naming convention so directory sorting surfaces
  them side by side.
- **Phase B — extract ``_overlays.py``** (~1800 LOC): every overlay
  factory (``_make_confirm_overlay_class``,
  ``_make_progress_overlay_class``, ``_make_seed_bundle_picker_class``,
  ``_make_training_options_class``, ``_make_open_video_overlay_class``)
  and the ``_show_first_paint_notice`` helper into one Qt-modal
  toolkit module. Not paired with any single feature; the
  workflow-specific ``*_modal.py`` files compose these factories.
- **Phase C — extract ``_file_management.py`` + expand
  ``_bundle.py``**: ``VideoFileManager`` + canonical-path helpers +
  frame-extraction helpers (``_extract_frames``,
  ``_extract_frames_decord``, ``make_annotation_file_name``,
  ``get_annotation_file_name``, ``merge_annotations_in_folder``,
  ``rebase_to_config``) move to ``_file_management.py``; bundle
  hydration helpers (``hydrate_bundle_data_only``,
  ``finalise_bundle_artists``, ``park_bundle_artists``, etc.) move
  from inline to ``_bundle.py`` as standalone functions over
  ``_BundleState``.
- **Phase D — narrow ``dlcinterface.py`` + extract ``gui.py`` +
  ``_open.py``**: ``DUSTrack`` moves to ``gui.py``; the ``open()``
  dispatch + seed-modal session helpers move to ``_open.py``.
  ``dlcinterface.py`` is now ~1700 LOC, holding only ``DLCProject``
  + ``DLCData`` + the lazy-DLC ``__getattr__`` proxy + the
  ``_RELOCATED_NAMES`` shim that keeps
  ``from dustrack.dlcinterface import X`` callers resolving for the
  relocated names without forcing the new modules to load eagerly
  (which would cycle through ``dlcinterface`` at module-load time).
- **Phase E — extract ``annotations.py`` from ``pointtracking.py``**:
  ``VideoAnnotation`` + ``VideoAnnotations`` + ``_TrackedFrameDict``
  move into ``annotations.py``.
- **Phase F (above) — ``_DUSTrackBase`` merged into ``DUSTrack``**
  and ``pointtracking.py`` deleted. See the earlier "1.2.0rc1
  refactor tail" note.
- **Follow-up — extract ``_dlc_paths.py``**: six pure path
  predicates (``_is_dlc_config_yaml``, ``_is_dlc_project_root``,
  ``_find_dlc_config``, ``_find_video_index``,
  ``_session_inside_dlc_project``, ``_resolve_multi_video_from_list``)
  move out of ``dlcinterface.py`` into their own module; shimmed
  through ``dlcinterface.__getattr__``.
- **Follow-up — ``gui.py`` thin-coordinator pass**: extract DUSTrack
  helpers (bundle-swap state machine, broadcast-statevar hooks,
  navigation widget binding) so the class file is orchestration code
  with per-feature logic in the ``_*`` modules.
- **Hygiene — import sweep + black formatting**: cleared unused
  imports across every module, declared ``__all__`` in
  ``__init__.py`` to make the public surface explicit (17 names),
  applied black to every module per pyproject's ``[tool.black]``.
  Fixed three latent undefined-name bugs the refactor introduced
  (missing ``plt`` import in ``_open`` seed-modal fallback paths,
  missing ``import_seed_bundle_into_project`` in ``gui.py`` seeding
  flow, missing ``DLCProject`` type reference in
  ``_file_management.py``). Kept ``seed.py``'s ``_USER_CONFIG_DIR``
  / ``_USER_CONFIG_PATH`` re-exports with explicit noqa — the test
  fixture in ``tests/test_seed.py`` monkeypatches them.

**Public API: observably backwards-compatible.** Portfolio sweep
across `pn-projects/`, `immersionToolbox/`, and
`datanavigator/tests/` confirmed zero external callers reference
relocated module paths directly; every external import uses the
top-level ``dustrack.X`` surface. The ``dlcinterface.__getattr__``
lazy proxy covers the relocated names that internal tests + bench
scripts still reach via ``from dustrack.dlcinterface import X``.
Three benchmark scripts (``bench_decode_in_ui.py``,
``14_benchmark_fast_render.py``, ``14b_paint_counter.py``) reach
``dustrack.dlcinterface.VideoFileManager`` — works because
``dlcinterface.py`` imports ``VideoFileManager`` at module level
from ``_file_management``. Three autodoc directives in
``docs/api.md`` (``_extract_frames``, ``get_annotation_file_name``,
``merge_annotations_in_folder``) were stale and have been
re-pointed at ``dustrack._file_management``.

Tests: **588 passed, 1 skipped** (the skip is the no-deeplabcut
install path, conditional on ``HAS_DLC`` being False).

## [1.2.0a2] - unreleased

Cold-open optimisation: two independent wins folded together — the
vectorised DLC-trace conversion (drops the per-frame `.loc[frame]`
pandas cross-section), and a single shared `VideoReader` across all
annotation layers (drops the per-layer `utils.Video(vname)` open).
Together they take `g.annotate()` on the pia02 `interosseous_pn24-x`
production benchmark from **7.95 s → 2.96 s** (−5.0 s, **2.69× faster**)
on video 0.

DLC training UI controls: new `DLCProject.train_iteration` (explicit-args
sibling of `process()`) plus a Training options modal that opens
before the existing pre-flight scan when the user clicks Train DLC
model. The modal exposes refine mode (scratch / in-project /
external), source iteration + snapshot picker (or Browse... for an
external `.pt`), training epochs, and a create-labeled-video toggle.
Closes a latent silent-drop bug: `process(refine=<path>)` was a no-op
on DLC2 because DLC2 has no `train_network(snapshot_path=...)`
runtime override; the new path edits pose_cfg's `init_weights`
explicitly. `process()` is unchanged for CLI users.

Zero-argument launch: `dustrack.open()` with no path pops a Qt
multi-select file picker, and a new `dustrack` console-script entry
point in `pyproject.toml` lets users launch the GUI by typing
`dustrack` at any shell. Closes the "user does not need to use the
command line" gap. List-form `open([p0, p1, ...])` is also new:
first path dispatches, rest land on `tracker._video_queue` for the
forthcoming multi-video navigation work.

Cross-session history: the picker remembers the last folder a video
was picked from, and DUSTrack's close-guard writes opened videos to
a `recent_videos` list (multi-video sessions also write the common
parent folder to `recent_folders`). Both stored in
`~/.dustrack/config.json` for the future "pick from history" modal
that bundles with the seed-window work post-multi-video.

Earlier 1.2.0 scope items (dnav 1.5.0 adoption + DLC `.h5` reclaim)
came along with the 1.2.0a1 relocation. The originally-scheduled
`fast_traces=True` Qt-tier was explored on a throwaway branch
2026-05-20 and reverted after the benchmark showed an 8.81×
per-frame regression on the production video; the matplotlib trace
pane stays. See portfolio memo `feedback_qt_traces_benchmark_2026_05_20`.

### Fixed
- **Pre-flight incomplete-frame scan uses project bodyparts**
  (`dlcinterface.py`). ``_scan_incomplete_frames`` now accepts an
  optional ``target_labels=`` argument; ``_scan_unsaved_and_incomplete``
  passes the project's ``config['bodyparts']`` (mapped through
  ``_dlc_bodyparts_to_layer_labels``) when a project is set.
  Closes a bug surfaced by the seed-from-bundle visual smoke: a
  layer where the user touched only label ``"0"`` of a
  ``["point0", "point1"]`` project was treated as complete by the
  legacy "active labels = labels with any annotation" rule, so the
  Save-and-clean modal didn't fire and training proceeded on
  single-bodypart frames. Project-aware mode considers every
  bodypart required regardless of whether the user has annotated
  it yet. Legacy ``target_labels=None`` mode preserved for the
  no-project / empty-bodyparts edge cases.
- **Pre-flight remediation drops incomplete frames directly**
  (`dlcinterface.py`). ``_apply_pre_flight_remediations`` previously
  ran ``ann.remove_empty_labels()`` followed by
  ``ann.keep_overlapping_frames()``. That pair silently failed in
  the project-aware case: a required-but-empty label (user touched
  only ``"0"`` of a ``["point0", "point1"]`` project) got dropped
  by step 1, and step 2 then trivially preserved every incomplete
  frame under the now-single-label schema. Net effect: Save-and-
  clean ran, but the layer still had the incomplete frames, the
  post-clean re-check passed, and DLC failed at
  ``create_training_dataset`` downstream. Fix: drop incomplete
  frames directly via ``ann.remove(label, frame)`` using the
  scan's ``incomplete`` dict as the source of truth -- correct in
  both project-aware and legacy modes, and routes mutations
  through the canonical revision-bumping path.
- **Post-clean re-check for trainable labels** (`dlcinterface.py`).
  After the Save-and-clean pre-flight remediation runs in
  ``process_dlc_project``, the trainable-labels predicate is
  re-evaluated. If cleaning dropped every annotated frame (typical
  post-seed case: user annotated only one of the project's
  bodyparts, every frame got dropped as incomplete) AND no other
  labels exist anywhere in the project, the same
  ``_prompt_no_trainable_labels`` overlay used at click-time fires
  and training is hard-blocked. Without this, DLC's
  ``create_training_dataset`` would fail downstream with a less
  helpful message.
- **Seeding overlay progress bar + radio-button visibility** (`dlcinterface.py`).
  Two visual nits on the 1.2.0a2 modals. (a) The Create-and-seed
  ``ProgressOverlay`` was opened with ``show_progress_bar=False``,
  so the ``analyze_videos`` phase had no bar despite DLC's tqdm
  output matching ``_PROGRESS_PATTERNS``; flipped to ``True`` to
  match the Train DLC overlay. (b) ``QRadioButton::indicator`` on
  the Training options modal inherited Windows' native checked
  rendering (a white inner ring) which was almost invisible
  against the ``rgba(0, 0, 0, 200)`` overlay backdrop -- explicit
  QSS now renders the indicator as a white-bordered circle filled
  with the primary-action blue (``#3a86ff``, matching the Train
  button) when checked.
- **Disabled-feature visibility in Training options modal** (`dlcinterface.py`).
  Sibling of the prior checked-indicator fix. On the dark
  ``rgba(0, 0, 0, 200)`` overlay backdrop, the modal's disabled
  state was nearly indistinguishable from the enabled state --
  ``_set_row_enabled`` only re-coloured ``QLabel`` children, and a
  disabled ``QRadioButton`` (e.g. "Refine from in-project iteration"
  with no trained iterations yet) kept its white text + white
  indicator border. Now: QSS ``:disabled`` rules mute radio +
  checkbox text to ``#777777`` and the radio indicator border to
  ``#666666`` (with a muted ``#4a5a7a`` checked-fill), and
  ``_set_row_enabled`` applies a ``QGraphicsOpacityEffect`` at 0.40
  to the whole sub-row so the native ``QComboBox`` / ``QLineEdit`` /
  ``QSpinBox`` / ``QPushButton`` children dim uniformly instead of
  relying on Windows-native disabled rendering, which is too subtle
  on this backdrop.
- **Empty manual-layer labels after seeding** (`dlcinterface.py`).
  Two compounding issues left the post-seed ``annotation_label``
  dropdown showing the bootstrap default ``"0"`` instead of the
  bundle's bodyparts. (1) ``_DUSTrackBase.__init__`` snapshots
  ``self.ann.labels`` into the dropdown statevariable once and
  never refreshes it when the active layer flips to
  ``iteration-1`` post-seed. (2) ``add_annotation_layers``' union
  pass adds missing labels but never *removes* the session-bootstrap
  ``"0"``, so non-sequential bodyparts like ``["point1", "point3"]``
  would land on a layer with ``["0", "1", "3"]`` rather than
  ``["1", "3"]``. Fix: after seeding (or training, or fresh-from-
  project annotate), reset every *empty* manual layer's labels to
  match the project's bodyparts exactly via
  ``_normalize_empty_manual_layer_labels``, then re-bootstrap the
  ``label_range`` + ``annotation_label`` statevariables from the
  active layer via ``_rebootstrap_label_states``. Bodypart -> label
  mapping (``_dlc_bodyparts_to_layer_labels``) mirrors the
  ``_dlc_trace_to_annotation_dict`` convention: strip the ``"point"``
  prefix when every bodypart is ``point<digit>``; otherwise fall
  back to consecutive indices.

### Changed
- **Create DLC Project on an empty active layer routes through a
  Seed-from-bundle modal** (`dlcinterface.py`). Qt path only. When
  ``not any(self.ann.data.values())`` at click time, a three-step
  sequence runs: (a) confirm-intent overlay ("Browse for seed
  bundle… / Cancel"), (b) QFileDialog folder picker, (c) detected-
  bodyparts preview ("Create and seed / Cancel"). On accept, a new
  ``DLCProject`` is scaffolded, the bundle is installed as
  iteration-0's trained model via ``import_seed_bundle_into_project``,
  and ``analyze_videos(iteration_num=0)`` runs inline (under the
  existing ``ProgressOverlay`` with ``_SEED_PROJECT_PHASES``). On
  Done, ``_refresh_dlc_layers`` loads the predictions as a dense
  ``dlc_iteration-0`` overlay and points the active layer at an
  empty ``iteration-1`` manual layer for refinement. The
  ``seed_bundle_path=`` kwarg on
  :meth:`DUSTrack.create_dlc_project` also lets callers drive the
  same flow programmatically (Qt or non-Qt). The non-empty path is
  unchanged.
- **Train DLC guard: hard-block on no-labels-anywhere, confirm
  otherwise** (`dlcinterface.py`). The empty-active-layer guard
  now distinguishes two situations:
  - **Hard block** when ``not self._has_trainable_labels()`` --
    active layer empty AND no other manual layer in the session
    has data AND no ``*.h5`` exists under ``<project>/labeled-data/``.
    Typical trigger: freshly-seeded project, user clicks Train
    before annotating any iteration-1 frames. ``_prompt_no_trainable_labels``
    surfaces an error overlay instructing the user to annotate
    first or run Apply manual corrections to convert a DLC trace
    into a manual layer.
  - **Confirm modal** when labels exist elsewhere: mid-refinement,
    user clicked Train without adding new labels this pass; training
    is feasible by reusing the existing ``labeled-data/`` or by
    running ``extract_frames`` on the other manual layer. The
    confirm wording explains the situation; default is Cancel.
- **Seed-bundle picker + remembered root**
  (`dlcinterface.py`, `seed.py`). The Create-DLC-Project seeding
  modal opens a list-picker (``SeedBundlePickerDialog``) when a
  seed-bundles root has been configured: every valid bundle in
  the root is listed inline by name + bodyparts + description.
  Actions: ``Use selected`` / ``Browse elsewhere…`` /
  ``Change bundles root…`` / ``Cancel``. The picker falls back
  to the previous Browse-only flow when no root is configured,
  and auto-offers to remember the picked bundle's parent as the
  root after a successful Browse so the next session opens the
  picker directly. Persistence lives at ``~/.dustrack/config.json``
  (key ``seed_bundles_root``). New public API in ``dustrack.seed``:
  ``get_seed_bundles_root``, ``set_seed_bundles_root``,
  ``list_seed_bundles``.
- **Bundle ``description.txt`` support** (`seed.py`).
  ``extract_snapshot_for_seeding`` accepts a ``description=``
  kwarg and writes ``description.txt`` into the bundle;
  ``inspect_seed_bundle`` reads it and surfaces it via the
  ``"description"`` key, which the picker shows alongside
  bodyparts so users can identify bundles without opening the
  folder.
- **`DUSTrack.process_dlc_project` routes through
  `DLCProject.train_iteration` on the Qt path** (`dlcinterface.py`).
  The Qt button click now opens the Training options modal first; on
  Accept, the modal's choices become `train_iteration(**kwargs)` in
  the worker closure under the existing `ProgressOverlay`. Positional
  `*args` / keyword `**kwargs` passed to `process_dlc_project` are now
  **ignored on the Qt path** -- the modal owns the kwarg surface
  per-click. Non-Qt fallback path is unchanged: still routes through
  `DLCProject.process()` with `kwargs.setdefault('create_video',
  False)`. `create_video`'s pre-1.2.0a2 setdefault that applied to
  both paths is now scoped to the non-Qt branch only.

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
- **`dustrack.inspect_seed_bundle(bundle_path)`** + **`dustrack.import_seed_bundle_into_project(dlc_project, bundle_path, iteration=0, shuffle=1)`**
  (`dustrack/seed.py`). The other half of the seeding flow.
  ``inspect_seed_bundle`` validates a bundle and returns its
  resolved file paths + bodyparts (for the modal preview);
  ``import_seed_bundle_into_project`` wires the bundle into an
  existing ``DLCProject`` so DLC sees iteration-N as already
  trained. Side effects: overwrites project bodyparts from the
  bundle, manufactures
  ``dlc-models-pytorch/iteration-N/<modelfolder>/{train,test}/``,
  rewrites the path fields inside ``pytorch_config.yaml`` and
  ``pose_cfg.yaml`` to the destination, and registers the shuffle
  in ``training-datasets/iteration-N/.../metadata.yaml`` (the
  registry DLC's ``TrainingDatasetMetadata.get`` consults at
  inference). After import, ``analyze_videos(iteration_num=N)``
  produces predictions without training; an end-to-end slow test
  pins the contract on a real pia02 video + the interosseous
  bundle.
- **`dustrack.extract_snapshot_for_seeding(snapshot_path, destination_path)`**
  (`dustrack/seed.py`). First piece of the "seed an empty project
  from an external snapshot" flow. Given a `.pt` inside an existing
  DLC3 project's `dlc-models-pytorch/iteration-N/<modelfolder>/train/`,
  copies three files into the destination folder: the snapshot,
  its `pytorch_config.yaml`, and the sibling `test/pose_cfg.yaml`
  (one level up from `train/`). All three are required at inference
  time -- DLC's pytorch `analyze_videos` reads each one via
  `DLCLoader.model_cfg` + `read_plainconfig(...test/pose_cfg.yaml)`
  (`deeplabcut/pose_estimation_pytorch/apis/videos.py:425`).
  `learning_stats.csv`, `train.txt`, and sibling snapshots are
  deliberately omitted. Absolute paths inside `pytorch_config.yaml`
  (`metadata.project_path` / `metadata.pose_config_path`) and
  `pose_cfg.yaml` (`dataset`) are copied verbatim; the eventual
  importer (the upcoming "Create DLC Project" modal when the active
  manual layer is empty) is responsible for rewriting them.
- **`DLCProject.train_iteration(...)`** — explicit-args sibling of
  `DLCProject.process()`. Where `process()` auto-infers state and picks
  sane defaults for CLI ergonomics, `train_iteration` assumes the
  caller has already decided everything (refine source, training
  duration, output options) and performs strict per-`refine_mode`
  validation with no inference or silent fallbacks. Three modes:
  - `refine_mode="scratch"` — no pose_cfg edit; train from a fresh
    `create_training_dataset` pose_cfg.
  - `refine_mode="in_project"` — requires `source_iteration` (int,
    must be trained); optional `source_snapshot` (None = best).
    Routes through `initialize_weights(...)`. Works on both DLC2
    and DLC3.
  - `refine_mode="external"` — requires `external_snapshot_path`
    (file must exist). On DLC3 the path is passed straight to
    `train_network(snapshot_path=...)` (existing pattern); on DLC2
    a new helper `_initialize_weights_from_external_path` edits
    pose_cfg's `init_weights` to the external path string (since
    DLC2's `train_network` has no runtime override). This closes a
    surprise where `process(refine=<str>)` was silently a no-op on
    DLC2 because the DLC2 branch never propagated the string. The
    GUI Training options modal (below) is the primary caller.
- **Training options modal on Train DLC model click**
  (`dlcinterface.py`). New `_make_training_options_class()`
  qtpy-lazy-import factory + `DUSTrack._prompt_training_options(qt_window)`
  method + two pure helpers (`_default_training_options(dlcproject)`,
  `_training_options_to_train_iteration_kwargs(options)`). Modal
  opens BEFORE the existing pre-flight scan (separate concerns:
  training config vs. data-loss prevention). Layout:
  - **Refine mode** radio group: `Start from scratch` /
    `Refine from in-project iteration` / `Refine from external
    snapshot`. The in_project radio is auto-disabled when no
    trained iterations are available (first-time training).
  - **In-project sub-row**: iteration `QComboBox` (lists every
    trained iteration; pre-selects the latest) + snapshot
    `QComboBox` (repopulated on iteration change; first entry is
    `best (auto)` mapping to None so `initialize_weights` picks
    the best).
  - **External sub-row**: `QLineEdit` + `Browse...` button (opens
    a `QFileDialog` filtered to `*.pt` on DLC3 / `*.index` on
    DLC2). The DLC2 path edits pose_cfg explicitly via the new
    `_initialize_weights_from_external_path` helper.
  - **Training duration**: `QSpinBox` (label adapts to "epochs"
    on DLC3 / "iterations" on DLC2; pre-filled with the same 50 /
    500000 default `process()` uses).
  - **Create labeled video on completion**: `QCheckBox` (defaults
    off; the UI ergonomics default vs. `process()`'s CLI-parity
    `create_video=True`).
  - **Train** (primary QSS) / **Cancel** (neutral QSS) buttons,
    matching the rc2 ConfirmOverlay visual vocab.
  Modal shares ConfirmOverlay's dark-translucent backdrop +
  reposition + event-filter scaffolding; inner content `QWidget`
  intentionally carries no `QWidget { ... }` QSS so child
  combos / line edit / spinbox keep their native Windows
  rendering (avoids the cascade trap from
  `feedback_qt_qss_vs_palette`). Non-Qt fallback path is
  unchanged -- still routes through `DLCProject.process()` with
  its `kwargs.setdefault('create_video', False)` ergonomic.
  Visual polish + a future "create DLC project from external
  snapshot" feature deferred to a follow-up session.
- **`tests/test_train_iteration.py`** (34 tests) — validates
  `refine_mode` discrimination + DLC2/DLC3 dispatch + maxiters
  defaults + analyze_videos kwarg forwarding +
  `_initialize_weights_from_external_path` directly (DLC3 edits
  `resume_training_from`, DLC2 edits `init_weights`, `.pt` / `.index`
  extension stripping). Stub `DLCProject` subclass bypasses the
  heavy `__init__` + filesystem deps.
- **`tests/test_training_options_helpers.py`** (16 tests) — pure-helper
  coverage for `_default_training_options` + `_training_options_to_train_iteration_kwargs`
  across DLC2/DLC3 and trained/untrained project states. Pins the
  payload-shape contract (scratch drops `source_*` keys; in_project
  forwards `source_snapshot=None` rather than dropping it so
  `initialize_weights` picks the best snapshot; external drops
  `source_*` keys). The Qt widget itself is covered by manual
  smoke per the EnhanceWidget / ConfirmOverlay precedent.
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

### Added
- **Lazy `import deeplabcut`** (`dustrack/__init__.py`,
  `dustrack/dlcinterface.py`). The DLC import (~7 s on the dlc3rc14
  env) used to run at module-import time inside a top-level
  ``try: import deeplabcut`` block, so every ``import dustrack`` --
  including the CLI ``dustrack`` shell command -- paid the full cost
  before the picker could pop. Refactored into three pieces:
  - ``HAS_DLC`` now resolves via ``importlib.util.find_spec`` (cheap,
    no actual import) so button-gating decisions in
    ``DUSTrack.__init__`` don't need the real DLC.
  - ``_ensure_dlc_loaded()`` (synchronous, idempotent, thread-safe)
    imports DLC on first call and binds the module-level
    ``deeplabcut`` / ``VideoWriter`` / ``ScannerError`` / ``DLC3``
    globals. Called at the top of ``DLCProject.__init__`` and the
    legacy ``_extract_frames`` helper so every DLC-using callsite
    routes through one entry point.
  - ``_ensure_dlc_loaded_async()`` (fire-and-forget daemon thread,
    idempotent) is kicked off from ``dustrack.open()`` after the
    picker returns (or immediately if a path was supplied). DLC
    loads concurrently with DUSTrack construction + user annotation;
    by the time the user clicks Create DLC Project the import is
    typically done.
  - Companion ``register_dlc_load_callback`` + ``_dlc_load_state``
    helpers expose the loader state for the Workflow-button gate
    refresh (250 ms ``QTimer`` poll in ``__init__`` that flips Create
    DLC Project from greyed-out to enabled once the loader resolves)
    and for any future consumer that wants to react to the load
    finishing.

  ``import dustrack`` clean-interpreter median dropped **8.45 s →
  2.83 s** (≈3×, 5.6 s shaved); the remaining ~7 s DLC cost is fully
  hidden behind the picker / GUI construction / user-annotation
  window on the typical session. Workflow-button gate gained two
  new states for Create DLC Project: ``"Loading DeepLabCut…"`` while
  the bg load is in flight, ``"DeepLabCut failed to load."`` for the
  rare ``find_spec``-yes-but-import-failed edge case. Project-
  membership ("Already inside DLC project X") still wins precedence.

  Qt-binding side effects historically supplied by DLC's
  ``__init__`` are now set explicitly at the top of
  ``dustrack/__init__.py`` (gated on ``find_spec("deeplabcut")``,
  via ``setdefault`` so explicit shell overrides still win):
  ``QT_API=pyside6`` (matches DLC's choice; without it qtpy resolves
  to PyQt6 on multi-binding envs and ``_pin_qt_palette``'s light-mode
  pin stops working), plus the OpenMP guard pair
  ``KMP_DUPLICATE_LIB_OK=True`` / ``KMP_INIT_AT_FORK=FALSE`` and
  ``PYSIDE6_OPTION_PYTHON_ENUM=True``. 14 new tests in
  ``tests/test_lazy_dlc_loader.py`` (state-machine, async return
  shape, idempotency, callback fan-out incl. one-failure-doesn't-
  block-others, real ``deeplabcut`` import smoke); 5 new tests in
  ``tests/test_workflow_button_gates.py`` (pending / loading /
  missing states + project-membership-wins-over-loading); existing
  gate tests pinned via an autouse fixture that forces loader state
  to ``"done"``.
- **Cross-session recent-videos + recent-folders history; picker
  remembers last folder** (`dustrack/_config.py`,
  `dustrack/dlcinterface.py`). The no-arg picker now lands at the
  folder the user picked from last time (single source of truth:
  ``_config.get_last_video_picker_dir`` -- parent of the most-recent
  existing entry in ``recent_videos``, falling back to the most-
  recent existing entry in ``recent_folders``, falling back to the
  OS default on a fresh install). DUSTrack's close-guard
  (``_install_close_guard`` tail) writes ``self.fname`` to
  ``recent_videos`` on every successful close; for multi-video
  sessions (``self._video_queue`` non-empty), the common parent
  folder is also written to ``recent_folders``. Both lists are
  dedup-prepended, capped at 25 entries, and filtered to paths-that-
  still-exist on read (stale on-disk entries are kept so a re-mounted
  network drive can recover them). New public-ish accessors:
  ``dustrack._config.get_recent_videos`` / ``get_recent_folders`` /
  ``record_recent_video`` / ``record_recent_folder`` /
  ``get_last_video_picker_dir``. Stage for the post-multi-video
  "pick from history" modal: history collection starts now so the
  future modal opens against a non-empty list. **Refactor**:
  ``_read_user_config`` / ``_write_user_config`` lifted from
  ``seed.py`` into ``_config.py`` as the canonical home for
  cross-session JSON state; ``seed.py`` re-exports the symbols for
  back-compat (and the ``test_seed.py::isolated_user_config``
  fixture patches both modules so existing seed-bundles-root tests
  keep passing). 25 new tests in
  ``tests/test_user_config_recent.py`` (round-trip, dedup, cap,
  stale filtering, last-picker-dir derivation, picker directory
  arg, single-video and multi-video close-guard writes, mixed-
  drives commonpath fallback).
- **Zero-argument launch + `dustrack` console entry**
  (`dustrack/dlcinterface.py`, `dustrack/cli.py`, `pyproject.toml`).
  Closes the "user does not need to use the command line" gap.
  ``dustrack.open()`` with no arguments pops a Qt
  ``QFileDialog.getOpenFileNames`` (videos + All files filter,
  multi-select) and threads the picked list through the existing
  Phase 1 / Phase 2 dispatch; cancel returns ``None`` so the call
  is safe in REPL / script contexts. ``open()`` also now accepts a
  list / tuple of paths: the first dispatches, the rest land on
  ``tracker._video_queue`` (always set, defaults to ``[]``) so the
  forthcoming multi-video navigation work has a queue to consume.
  ``str`` and ``Path`` entries coexist in the list form. New
  ``[project.scripts]`` entry exposes the workflow as a shell
  command: typing ``dustrack`` from any conda env that has the
  package installed pops the picker and blocks until the window
  closes (no Python REPL needed). The picker is files-only in this
  release; folder selection (recurse for videos) arrives alongside
  the multi-video swap-state contract on the Roadmap. Helper:
  ``_prompt_for_videos`` (qtpy-lazy, ``QApplication.instance() or
  QApplication([])`` bootstrap, returns ``None`` on cancel /
  mpl-only install). 12 new tests in
  ``tests/test_open_zero_arg.py`` (picker mocked); modal-loop Qt
  exec stays manual-smoke per the ConfirmOverlay precedent.
- **Workflow-button gating** (`dustrack/dlcinterface.py`). The three
  DLC-aware Workflow-group buttons now disable themselves (visible
  but greyed, with tooltip) when their click handler would otherwise
  fail or scaffold a bogus project:
  - **Create DLC Project** disables when the session is already
    inside a DLC project (either `self._dlcproject` is set, or the
    video path walks up to a `config.yaml + videos/ + labeled-data/`
    triad). Avoids creating a nested project at
    `<project>/videos/<video>/<new-project>/` that nothing downstream
    handles. Tooltip names the project root.
  - **Train DLC model** disables when `self._dlcproject is None`,
    replacing the click-time `ValueError("DLCProject not created. Use
    create_dlc_project() to create it.")` with a discoverable greyed
    button + tooltip.
  - **Apply manual corrections** disables when there's no overlay
    layer set, or when the active layer is already the corrections
    output (`dlccorr`). Mirrors the two `ValueError` paths in
    `apply_manual_corrections`.

  **Reduce jitter is intentionally not gated.** Its real precondition
  is "every frame of the active layer is fully annotated", which is a
  data property; the cheap name-pattern proxy (`_is_dense_layer_name`)
  is correct for rendering style but would false-disable a fully-
  annotated manual layer. Deferred until a cheap data-side check
  exists. The runtime sparse-labels guard in
  `lk_moving_average_filter` is the existing source of truth.

  New module-level helper `_session_inside_dlc_project(dustrack)`
  consolidates the in-project check (short-circuits on `_dlcproject`,
  walks up `fname` via the existing `_find_dlc_config`). New methods
  `DUSTrack._evaluate_workflow_gates()` (pure-state predicate, unit-
  testable) and `_refresh_workflow_button_state()` (Qt-side
  `setEnabled` / `setToolTip` writer). `_qss_for_group` grew a
  `QPushButton:disabled` rule (desaturated bg `#d8d8d8`, dim grey
  italic text `#888888`, dashed border `#b0b0b0`) so disabled buttons
  read as such across all four group palettes; without it the
  enabled-state QSS won over Qt's built-in disabled paint and
  `setEnabled(False)` produced no visual change. Refresh fires at end of
  `__init__`, after `_rewire_to_in_project_paths` (post-Create
  success), after `remove_current_layer` (layers may drop), at the
  tail of `DLCProject.annotate` (Phase-2 open path that sets
  `_dlcproject` *after* DUSTrack `__init__` ran), and on every
  `annotation_layer` / `annotation_overlay` dropdown change via
  dnav 1.4.0rc2+'s `StateVariable.add_on_change`. 16 new tests
  (`tests/test_workflow_button_gates.py` + 5 added to
  `tests/test_open.py::TestSessionInsideDLCProject`); Qt round-trip
  left to manual smoke per the existing posture for
  `ConfirmOverlay` / palette-pin.

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
