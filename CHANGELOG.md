# Change Log
All notable changes to this project will be documented in this file.

## [1.1.0rc2] - 2026-05-18 (unreleased)

Second release candidate for the smoother-interaction band. rc1 was
backend-perf-oriented (datanavigator 1.4.0rc1 cache + revision-counter
fixes); rc2 turns to the user-facing rough edges, starting with the
DLC training round-trip.

### Changed
- `dustrack/dlcinterface.py`: `DUSTrack.process_dlc_project` no longer
  closes the figure and re-opens it after training. On a Qt backend
  (the default for `DUSTrack(..., fast_render=True)`, which is the
  default), training now runs on a background thread under a modal
  "Training in progress" overlay parented to the QMainWindow. The
  overlay shows the current pipeline phase (extract / train / evaluate
  / analyze / labeled-video), a progress bar driven by parsed
  `Epoch X/Y` and iteration markers in DLC's stdout, and a scrolling
  tail of the last few hundred log lines. On successful completion,
  the overlay dismisses and the newly-produced DLC trace layers are
  added to the live DUSTrack via `add_annotation_layers` -- no
  relaunch -- with the freshest `dlc_*` layer set as the annotation
  overlay and drawn as a line plot. The pre-rc2 close-and-reopen path
  is retained as the fallback when no QMainWindow can be located
  (non-Qt backend, headless run, etc.).
- `dustrack/dlcinterface.py`: DLC's stdout/stderr during training are
  now teed to `sys.__stdout__` (the original terminal file descriptor)
  rather than the possibly-wrapped `sys.stdout`, so launching from a
  shell now reliably shows training progress in the terminal even when
  the call is initiated from the GUI button handler. Output also feeds
  the in-app overlay log in real time.

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
- `dustrack/dlcinterface.py`: module-level `_Tee`, `_QueueWriter`, and
  lazily-built `_make_training_overlay_class()` -- the plumbing for the
  off-thread training run + overlay. The Qt class builder mirrors
  datanavigator's `_make_qt_text_overlay_class` pattern so importing
  dustrack on a no-Qt-binding machine doesn't touch qtpy.

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
