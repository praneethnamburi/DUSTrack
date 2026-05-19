# Change Log
All notable changes to this project will be documented in this file.

## [1.1.0rc2] - 2026-05-18 (unreleased)

Second release candidate for the smoother-interaction band. rc1 was
backend-perf-oriented (datanavigator 1.4.0rc1 cache + revision-counter
fixes); rc2 turns to the user-facing rough edges of the DLC pipeline:
**all three DLC-pipeline buttons** (Train DLC model, Reduce jitter,
Create DLC project) now run on a background thread under a shared
modal `ProgressOverlay` instead of freezing the GUI; a unified
**Done** button on the overlay lets the user review the final stdout
(or read the error) before the underlying UI becomes interactive
again. The training overlay no longer auto-dismisses, and the
pre-rc2 `QMessageBox` failure dialogs are folded into the overlay
itself. **Plus (2026-05-19 fold-in from the originally-planned rc3
robustness band)**: save-on-close guard intercepts every way the
window can close (X button, alt-F4, `plt.close()`) and offers
*Save all / Discard / Cancel* on any unsaved diff.

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
- **Two-slider `EnhanceWidget`** -- new
  `_make_enhance_widget_class` factory + new
  `DUSTrack._add_enhance_widget()` that mounts the widget below the
  statevars widget in the rc2 left-column dock. Two `QSlider`
  controls with live numeric labels: **CLAHE clip** maps to
  `[1.0, 4.0]` and **Gamma** maps to `[1.0, 1.5]`. CLAHE grid (`8`)
  stays at the `__init__` default. Slider values update
  `self._clahe_clip` / `self._gamma` and call `self.update()` on
  every value change so the image redraws live. Sliders are integer
  `0..100`; pure-function slider<->param maps
  (`_slider_to_clahe_clip` / `_slider_to_gamma` / inverses) live at
  module scope and are unit-tested in
  `tests/test_enhance_widget_mapping.py`. **Sliders-at-minimum is
  the true bypass**: `_enhance_is_passthrough(clip, gamma)` returns
  `True` when both sliders sit at their leftmost position (clip=1.0
  AND gamma=1.0); the image processor short-circuits and returns
  the raw frame untouched (skipping the CLAHE pass and the
  RGB->gray->RGB roundtrip). Replaces the originally-rc2 `Toggle
  enhance` button -- with slider-driven bypass at min, a separate
  toggle is redundant. Constructor defaults shifted to
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

### Changed
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
