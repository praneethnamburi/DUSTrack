# Codebase organization

DUSTrack is a single Python package, ``dustrack/``, organised so the
interactive GUI class (``DUSTrack``) is a thin coordinator on top of
small, focused helper modules. This page lists every module and what
it's responsible for, so you can find the right file by purpose
without reading the GUI class first.

## Reading guide

* **Entry points are at the top of the alphabet** — ``cli.py``,
  ``dlcinterface.py``, ``gui.py``, ``seed.py``. Start there if you
  want to know what DUSTrack *does*.
* **Internal helpers are prefixed with an underscore** —
  ``_bundle.py``, ``_overlays.py``, ``_preflight.py``, etc. These are
  the building blocks the entry points compose; you generally don't
  import them directly, but you may need to read them when extending
  a feature.
* **Two modules sharing a prefix are a logic/UI pair.** The naming
  convention follows ``lk_filter.py`` + ``lk_opticalflow.py`` (LK
  algorithm pair) and ``_preflight.py`` + ``_preflight_modal.py``
  (logic + Qt modal pair). Sorting the directory alphabetically
  surfaces these pairs side-by-side.
* **``_overlays.py`` is the shared Qt modal toolkit** — generic
  building blocks (confirm overlay, progress overlay, picker dialog,
  training-options dialog) that the workflow-specific ``_*_modal.py``
  files compose. It pairs with no single feature.

## Entry points

### ``__init__.py``
Re-exports the public surface: :func:`dustrack.open`, the
:class:`DUSTrack` GUI class, the :class:`DLCProject` wrapper, the LK
filter, and the seed-bundle helpers.

### ``cli.py``
Console-script entry point invoked by ``dustrack`` / ``dustrack
<video>`` from the command line. Parses argv, calls
:func:`dustrack.open`, drives the event loop.

### ``gui.py``
:class:`DUSTrack` — the interactive video point-annotator. Inherits
from :class:`datanavigator.videos.VideoBrowser`. The class itself is
a coordinator: it owns the figure, the buttons, the statevariables,
and the bundle list, and delegates to the helper modules below for
each feature.

### ``dlcinterface.py``
:class:`DLCProject` — the DeepLabCut project wrapper. Train, infer,
extract frames, manage iterations. Also hosts the file-pattern
helpers that identify DLC projects on disk (``_is_dlc_project_root``,
``_find_dlc_config``, ``_resolve_multi_video_from_list``).

### ``seed.py``
Logic side of the seed-bundle workflow: inspect a bundle folder,
list bundles under a remembered root, install a bundle into a
fresh DLC project as iteration-0. Pairs with
``_seed_bundle_modal.py`` (the Qt UI).

### ``batch.py``
Bulk runner for headless / scripted processing across many videos.
Useful when the GUI isn't needed.

### ``annotations.py``
:class:`VideoAnnotation` and :class:`VideoAnnotations` — the
in-memory representation of a per-frame point-annotation layer. Add
/ remove / save / load primitives plus the DLC h5 trace decoder.

## Core feature modules

### Optical flow (LK pair)

* **``lk_opticalflow.py``** — :func:`lucas_kanade`,
  :func:`lucas_kanade_rstc`. Pure OpenCV LK + Reverse-Sigmoid
  Tracking Correction. Standalone; no GUI dependencies.
* **``lk_filter.py``** — :func:`lk_moving_average_filter`. Sliding-
  window LK smoothing of an annotation layer. Standalone.

The GUI's "Reduce jitter" button (``process_with_lk``) and the
``v`` / ``ctrl+b`` / ``a`` keybindings are thin adapters that call
into these.

### DLC integration

* **``dlcinterface.py``** — :class:`DLCProject` (above).
* **``dlcloader.py``** — Lazy import of ``deeplabcut`` on a
  background thread, so ``import dustrack`` stays fast. Exposes
  :func:`_ensure_dlc_loaded_async`, ``HAS_DLC``, and the load-state
  poll.
* **``dlcpatch.py``** — Opt-in monkey-patches to DLC's inference
  pipeline (multi-thread preproc, autocast). Opt-in, not auto-applied
  on rc14+ where vanilla DLC already captures the gain.

## Internal helpers (alphabetical)

### ``_bundle.py``
:class:`_BundleState` — per-video heavy + snapshot state for the
multi-video swap (1.2.0a3). Also houses :class:`_BgHydrationWorker`
(daemon thread + Qt poller for off-thread hydration) and the
per-bundle hydration helpers (``hydrate_bundle_data_only``,
``hydrate_phase1_bundle_data``, ``finalise_bundle_artists``,
``hydrate_bundle_sync``, ``derive_initial_bundle_selections``,
``park_bundle_artists``, ``show_bundle_artists``,
``notify_bundle_failure``).

### ``_bundle_swap.py``
Swap state machine and bundle list management on top of
``_bundle.py``: :func:`init_bundles`, :func:`swap_to`,
:func:`add_video`, :func:`remove_video`, :func:`replace_active_with`,
plus the statevar snapshot/restore pair
(:func:`capture_statevar_selections`,
:func:`restore_statevar_selections`) and the cross-bundle broadcast
helpers (:func:`install_broadcast_statevar_hooks`,
:func:`broadcast_statevar`).

### ``_close_guard.py``
Window-close safety net: install a QMainWindow ``closeEvent`` hook
that scans every ready bundle for unsaved diffs, shows the
Save / Discard / Cancel modal if any, and appends the session to
the recent-sessions store.

### ``_config.py``
Persistent user config: clips folder, cache folder, recent sessions.
Reads / writes a JSON file under the user's config home.

### ``_file_management.py``
:class:`VideoFileManager` — given a :class:`DLCProject` + video
index, resolve the canonical paths for every annotation layer (DLC
inference h5s, manual iteration JSONs, ``dlccorr``, ``buffer``).
Also hosts ``make_annotation_file_name``,
``get_annotation_file_name``, ``merge_annotations_in_folder``,
``rebase_to_config``, and the frame-extraction helpers
(``_extract_frames``, ``_extract_frames_decord``).

### ``_image_enhance.py``
CLAHE + gamma + brightness pipeline for ultrasound frames, plus
the two-slider :class:`EnhanceWidget` Qt class factory. Pure
image-processing kernel below the widget; the widget is built only
on the Qt path.

### ``_layer_names.py``
Name predicates and filename constructors for annotation layers:
``_is_dense_layer_name``, ``_dlc_bodyparts_to_layer_labels``,
``is_manual_layer_name``, ``is_manual_annotation_layer``,
``get_fname_annotations``. Used by the preflight scan, the
save-on-close sweep, the close-guard, and ``add_annotation_layers``.

### ``_nav_widget.py``
Multi-video navigation row: ``◀ <video dropdown> ▶`` mounted at the
top of the rc2 left-column dock. Build, refresh, sync to the bundle
list, and Alt+Left / Alt+Right key bindings.

### ``_open.py``
:func:`dustrack.open` — the public entry point. Dispatches to one
of: fresh ``DUSTrack`` on a bare video, ``DLCProject.annotate`` on
a video inside a DLC project, multi-video session on a project
folder, or the seed-modal launch session when called with no args.

### ``_overlays.py``
**Shared Qt modal toolkit.** Class factories for every modal
dialog DUSTrack uses (:func:`_make_confirm_overlay_class`,
:func:`_make_progress_overlay_class`,
:func:`_make_seed_bundle_picker_class`,
:func:`_make_training_options_class`,
:func:`_make_open_video_overlay_class`), plus the phase-pattern
tables for the progress overlay's stdout-tail parser, and the
``_Tee`` / ``_QueueWriter`` plumbing for teeing a worker thread's
stdout to both the overlay log and the launching terminal. Not
paired with any single feature — the workflow-specific ``*_modal.py``
files compose these factories.

### ``_preflight.py`` + ``_preflight_modal.py`` (pair)
* **``_preflight.py``** — Pure logic for "is this layer in a
  trainable state?". Scan incomplete frames, diff in-memory vs disk,
  write recovery sidecars, format breakdowns. Testable from
  synthetic dicts.
* **``_preflight_modal.py``** — Qt UI: the unified "Save and clean"
  modal, the no-trainable-labels block, the empty-active-layer
  confirm, and the remediation orchestrator.

### ``_qt_styling.py``
Qt palette + QSS helpers for the rc2 sidebar palette (pastel
analogous band across the Workflow / Display / Niche / Utilities /
Swap groups), plus the ``_pin_qt_palette`` reproducibility shim.

### ``_seed_bundle_modal.py``
Qt UI for the seed-bundle pick / confirm / Browse flow. Pairs with
``seed.py`` (logic side).

### ``_train_modal.py``
Qt UI for the Train DLC options dialog (refine_mode, source
iteration picker, external snapshot Browse, epochs/iterations,
create-labeled-video toggle). One modal; pairs conceptually with
``DLCProject.train_iteration`` in ``dlcinterface.py``.

### ``_view_state.py``
Per-bundle viewport + enhancement snapshot/restore: image-pane
zoom/pan (Tier 1 mpl + Tier 2 Qt dispatch), trace-axes xlim/ylim,
CLAHE/gamma/brightness slider values. Three get/set pairs, used
on every swap-out / swap-in.

### ``_workflow_gates.py``
Workflow-button enable/disable gates (Create DLC Project, Train
DLC model, Apply manual corrections). The :func:`evaluate_workflow_gates`
function is pure — testable with a mock tracker; the
:func:`refresh_workflow_button_state` function applies the gate
dict to the live Qt buttons.

### ``__version__.py``
Single source of truth for ``dustrack.__version__``. Read at import
time and (in 3.0.0rc14+) cross-checked against
``importlib.metadata.version("dustrack")`` because some PyPI wheels
shipped with a stale version constant.

## How a click flows through the codebase

A concrete example to anchor the modules in actual call paths.

### Clicking **Train DLC model**

1. ``gui.DUSTrack.process_dlc_project`` (the button handler) is
   invoked.
2. It calls ``self._prompt_training_options(qt_window)``, which
   delegates to :func:`._train_modal.prompt_training_options`. The
   user picks refine_mode + source + epochs.
3. It calls ``self._scan_unsaved_and_incomplete()``, which delegates
   to :func:`._preflight.scan_unsaved_and_incomplete`. This sweeps
   every manual layer for in-memory-vs-disk diffs (via
   :func:`._preflight.diff_ann_vs_disk`) and project-aware
   incomplete-frame detection (via
   :func:`._preflight.scan_incomplete_frames`).
4. If issues exist, it calls
   :func:`._preflight_modal.prompt_unified_pre_flight` for the
   "Save and clean" modal. On accept, it routes through
   :func:`._preflight_modal.apply_pre_flight_remediations`, which
   writes per-layer recovery sidecars and drops the incomplete
   frames.
5. It calls ``self._run_with_overlay(...)`` (still on the class) to
   spawn the training thread under a progress overlay built by
   :func:`._overlays._make_progress_overlay_class`.
6. The worker calls ``self._dlcproject.train_iteration(**kwargs)``
   in ``dlcinterface.py``.
7. On success, ``self._refresh_dlc_layers()`` re-discovers the new
   prediction layers via :class:`._file_management.VideoFileManager`
   and adds them to the session.

The class method ``process_dlc_project`` itself is now ~50 lines of
orchestration; every step above lives in its own module so it can
be tested in isolation.

### Clicking the multi-video ▶ arrow

1. ``gui.DUSTrack.swap_next`` is invoked by the
   ``QToolButton.clicked`` signal.
2. ``swap_next`` calls ``self.swap_to(self._active_index + 1)``,
   which delegates to :func:`._bundle_swap.swap_to`.
3. :func:`swap_to` snapshots the leaving bundle (via
   :func:`._bundle_swap.snapshot_active_bundle`), parks its artists
   (via :func:`._bundle.park_bundle_artists`), rebinds the shell
   onto the arriving bundle (via :func:`._bundle_swap.attach_bundle`),
   shows the arriving artists (via
   :func:`._bundle.show_bundle_artists`), restores statevars (via
   :func:`._bundle_swap.restore_statevar_selections`), restores the
   image + trace + enhance viewports (via the three get/set pairs in
   :mod:`._view_state`), and refreshes the nav buttons (via
   :func:`._nav_widget.refresh_nav_buttons`).
4. ``self.update()`` repaints once and the swap is done.

No mixin involved at any step. Every helper takes the tracker as an
argument and reads / writes shell state through it; the class file
just glues the pieces together.

## Tests

Each helper module has at least one corresponding test file under
``tests/``. Notable pairs:

| Module                    | Tests                                    |
|---------------------------|------------------------------------------|
| ``_bundle.py``            | ``test_bundle.py``, ``test_bg_hydration_worker.py`` |
| ``_bundle_swap.py``       | ``test_swap_to.py``, ``test_bundle_api.py`` |
| ``_close_guard.py``       | ``test_save_on_close.py``                |
| ``_image_enhance.py``     | ``test_enhance_widget_mapping.py``       |
| ``_layer_names.py``       | ``test_is_dense_layer_name.py``, ``test_canonical_layer_name.py`` |
| ``_open.py``              | ``test_open.py``, ``test_open_multi_video.py``, ``test_open_zero_arg.py`` |
| ``_overlays.py``          | ``test_open_video_overlay.py``           |
| ``_preflight.py``         | ``test_train_dlc_preflight.py``          |
| ``_preflight_modal.py``   | ``test_train_dlc_preflight.py``          |
| ``_seed_bundle_modal.py`` | ``test_seed_window.py``                  |
| ``annotations.py``        | ``test_pointtracking.py``                |
| ``dlcloader.py``          | ``test_lazy_dlc_loader.py``              |
| ``lk_opticalflow.py``     | ``test_opticalflow.py``                  |
| ``seed.py``               | ``test_seed.py``                         |

Run the suite with ``pytest`` from the repo root.
