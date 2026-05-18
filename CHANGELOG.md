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
itself.

### Changed
- `dustrack/dlcinterface.py`: button-column separators promoted from
  single to **double** (dnav 1.4.0rc2's new
  `Buttons.add_separator(style="double")`) to mark the major
  functional groups in the rc2 sidebar: shortcuts | DLC pipeline
  | trace + display controls. A trailing double separator now also
  closes the display-controls group (after "Toggle enhance"); the
  state-variables section gets its own trailing double separator
  for free via dnav's `_QtStatevarsWidget`. Visual rhythm matches
  what the rc2 stacked statevars layout introduced; users no longer
  have to scan for group boundaries in a long flat list of buttons.
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
- `dustrack/dlcinterface.py`: `DUSTrack.process_with_lk`
  (**Reduce jitter**) joins the overlay path on a Qt backend (no
  more UI freeze during long LK-RSTC passes). The overlay log
  shows the tqdm output and the progress bar is driven by parsing
  tqdm's `N/M` markers; phase label flips between "Submitting
  tracking jobs" and "Processing tracking results". On non-Qt
  backends the pre-existing synchronous behavior is retained,
  including the `VideoAnnotation` return value. On the Qt path the
  smoothed layer is added (and selected) when the user clicks Done.
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
  refresh failed" path.
- `dustrack/dlcinterface.py`: DLC's stdout/stderr during overlay
  work are teed to `sys.__stdout__` (the original terminal file
  descriptor) rather than the possibly-wrapped `sys.stdout`, so
  launching from a shell reliably shows progress in the terminal
  even when the call is initiated from a GUI button handler.
  Output also feeds the in-app overlay log in real time.

### Added
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
