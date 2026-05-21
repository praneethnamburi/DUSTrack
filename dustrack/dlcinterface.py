"""
Main DUSTrack module, including an interface to manage DeepLabCut (DLC) projects.
"""
from __future__ import annotations

import fnmatch
import functools
import os
import queue
import re
import shutil
import sys
import threading
import traceback
import warnings
from pathlib import Path, PureWindowsPath, PurePosixPath
from typing import Literal, Mapping, Optional, Union

import numpy as np
import pandas as pd
import cv2 as cv
import pyfilemanager
import pysampled
from skimage import io, img_as_ubyte

import matplotlib.pyplot as plt
import datanavigator as dnav
from datanavigator import VideoReader, cpu

from .postprocess import lk_moving_average_filter
from .pointtracking import VideoAnnotation, VideoAnnotations, _DUSTrackBase
from .seed import (
    get_seed_bundles_root,
    import_seed_bundle_into_project,
    inspect_seed_bundle,
    list_seed_bundles,
    set_seed_bundles_root,
)
from . import _config

try:
    import deeplabcut
    from deeplabcut.utils.auxfun_videos import VideoWriter
    from ruamel.yaml.scanner import ScannerError
    DLC3 = deeplabcut.__version__.startswith('3.')
    HAS_DLC = True
except ImportError:
    warnings.warn(
        'deeplabcut is not installed. You can still use the optical flow functions with DUSTrack.',
        stacklevel=2,
    )
    HAS_DLC = False


EXPERIMENTER = _config.EXPERIMENTER


def enhance_ultrasound_image(image, clahe_clip=2.0, clahe_grid=8, gamma=1.0, brightness=0):
    """
    Enhance ultrasound image for better visibility.

    Args:
        image: Input image (RGB or grayscale)
        clahe_clip: CLAHE clip limit (higher = more contrast)
        clahe_grid: CLAHE tile grid size
        gamma: Gamma correction (>1 = brighter midtones, <1 = darker)
        brightness: Brightness offset (-255 to 255)

    Returns:
        Enhanced RGB image for matplotlib display.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    else:
        gray = image

    # Apply CLAHE
    clahe = cv.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    enhanced = clahe.apply(gray)

    # Apply gamma correction
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        enhanced = cv.LUT(enhanced, table)

    # Apply brightness
    if brightness != 0:
        enhanced = np.clip(enhanced.astype(np.int16) + brightness, 0, 255).astype(np.uint8)

    # Convert back to RGB for matplotlib
    return cv.cvtColor(enhanced, cv.COLOR_GRAY2RGB)


# Layer-name patterns that indicate "dense" tracking output (data on
# every frame, like a model prediction or a smoothed trajectory) --
# the default rendering for these is a line plot, vs the dnav default
# of "dot" which is right for sparse manual annotations. Kept here as
# data, not a hardcoded predicate, so adding a new smoothing recipe
# (e.g. a second post-processing filter that writes
# <stem>_kalman_<param>.json) is a one-line tuple edit. See
# :func:`_is_dense_layer_name`.
_DENSE_LAYER_PREFIXES = ("dlc_", "dlccorr")
_DENSE_LAYER_SUBSTRINGS = ("lkmovavg",)


def _dlc_bodyparts_to_layer_labels(bodyparts: list[str]) -> list[str]:
    """Convert DLC ``bodyparts`` to DUSTrack annotation-layer ``labels``.

    Mirrors :meth:`VideoAnnotation._dlc_trace_to_annotation_dict`
    (the h5-trace loader) and inverts ``DLCProject.__init__``'s
    ``[f'point{x}' for x in annotation_names]`` synthesis at
    project-creation time:

    - If every bodypart strips cleanly to a digit after removing
      the ``"point"`` prefix, the labels are the bare digits
      (``["point0", "point1"]`` -> ``["0", "1"]``; ``["point1",
      "point3"]`` -> ``["1", "3"]``).
    - Otherwise (e.g. ``["nose", "ear"]`` from a non-DUSTrack
      project), the labels are consecutive indices starting at 0.

    Single source of truth for "given a project's bodyparts, what
    labels should a new manual annotation layer carry?".
    """
    prefix = "point"
    stripped = [bp.removeprefix(prefix) for bp in bodyparts]
    if stripped and all(s.isdigit() for s in stripped):
        return stripped
    return [str(i) for i in range(len(bodyparts))]


def _is_dense_layer_name(name: str) -> bool:
    """True if ``name`` is a layer that should render as a line plot
    by default (DLC inference, the ``dlccorr`` manual-corrections
    splice, or any LK-RSTC jitter-reduced output).

    The LK output of a non-DLC source (e.g. ``dlccorr``) lands at a
    name like ``dlccorr_lkmovavg_0.500`` via
    :meth:`VideoFileManager.canonical_layer_name`'s ``_annotations``
    branch -- dense like a DLC trace, but it doesn't start with
    ``dlc_``. The substring match catches it without widening the
    prefix list. ``dlccorr`` itself is dense because it's the
    overlay's per-frame DLC trace with the active layer's sparse
    manual edits spliced in -- per-frame coverage is inherited from
    the overlay.
    """
    return (
        any(name.startswith(p) for p in _DENSE_LAYER_PREFIXES)
        or any(s in name for s in _DENSE_LAYER_SUBSTRINGS)
    )


def _qss_for_group(spec: dict) -> str:
    """Build the per-group QSS string from a ``_SIDEBAR_PALETTE`` entry.

    Lifted to module-level so :func:`_make_group_styler` can close over
    it without dragging the whole ``DUSTrack`` class into the styler
    closure. Inputs are color-hex strings keyed by ``bg/fg/border/
    hover/pressed``.

    The ``:disabled`` rule paints workflow buttons whose gate
    predicate has refused (see ``DUSTrack._refresh_workflow_button_state``).
    Without it, the ``QPushButton`` selector above wins over Qt's
    built-in disabled styling and the button keeps its enabled
    look. Three cues compound for visibility across all four group
    palettes: a uniform desaturated bg, dim italic text, and a dashed
    border. No perf cost -- QSS is parsed once per button at add-time
    and Qt swaps style on enable/disable without re-parsing.
    """
    return (
        f"QPushButton {{ background-color: {spec['bg']}; "
        f"color: {spec['fg']}; border: 1px solid {spec['border']}; "
        f"padding: 4px; }} "
        f"QPushButton:hover {{ background-color: {spec['hover']}; }} "
        f"QPushButton:pressed {{ background-color: {spec['pressed']}; }} "
        f"QPushButton:disabled {{ background-color: #d8d8d8; "
        f"color: #888888; font-style: italic; "
        f"border: 1px dashed #b0b0b0; }}"
    )


def _make_group_styler(spec: dict):
    """Factory for a per-button styler closed over a palette ``spec``.

    Returned closure is registered on a :class:`Buttons` container via
    :meth:`datanavigator.assets.Buttons.register_style` and runs once
    per button at add-time inside ``_finalize_button``. No-op on the
    mpl fallback (``_qt_btn`` is absent there) -- pre-refactor
    behavior matched: the per-group palette only ever landed on the
    Qt path.
    """
    qss = _qss_for_group(spec)

    def _styler(b) -> None:
        qbtn = getattr(b, "_qt_btn", None)
        if qbtn is not None:
            qbtn.setStyleSheet(qss)

    return _styler


def _pin_qt_palette(dark: bool) -> None:
    """Pin the ``QApplication`` palette so DUSTrack looks the same
    regardless of Qt binding and Windows system theme.

    Why: PySide6 6.5+ on Windows honors the OS color scheme by default;
    PyQt6 does not. With both bindings now in play across portfolio
    envs (DLC mandates PySide6 via ``deeplabcut/gui/__init__.py:14``
    setting ``QT_API=pyside6``, while matplotlib/older envs prefer
    PyQt6), the same DUSTrack code would otherwise paint light on one
    machine and dark on another -- including dnav's built-in stylers,
    which sample the live palette via
    :func:`datanavigator.styles._is_dark_mode`. We force a Fusion-
    styled palette keyed off the explicit ``dark_mode`` kwarg so the
    appearance is deterministic; dnav's heuristic samples this pinned
    palette and stays in sync.

    No-op on the mpl-only path (qtpy import fails).
    """
    try:
        from qtpy.QtWidgets import QApplication
        from qtpy.QtGui import QPalette, QColor
    except ImportError:
        return
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    pal = QPalette()
    if dark:
        pal.setColor(QPalette.Window,          QColor(45, 45, 45))
        pal.setColor(QPalette.WindowText,      QColor(220, 220, 220))
        pal.setColor(QPalette.Base,            QColor(30, 30, 30))
        pal.setColor(QPalette.AlternateBase,   QColor(45, 45, 45))
        pal.setColor(QPalette.Text,            QColor(220, 220, 220))
        pal.setColor(QPalette.Button,          QColor(60, 60, 60))
        pal.setColor(QPalette.ButtonText,      QColor(220, 220, 220))
        pal.setColor(QPalette.ToolTipBase,     QColor(45, 45, 45))
        pal.setColor(QPalette.ToolTipText,     QColor(220, 220, 220))
        pal.setColor(QPalette.Highlight,       QColor(70, 110, 180))
        pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    else:
        # Explicit light palette. Do NOT use ``app.style().standardPalette()``
        # -- in Qt 6.5+ Fusion's standard palette follows the OS color
        # scheme, so on a Windows-dark-mode machine it returns dark
        # colors and the whole point of the pin is lost.
        pal.setColor(QPalette.Window,          QColor(240, 240, 240))
        pal.setColor(QPalette.WindowText,      QColor(0, 0, 0))
        pal.setColor(QPalette.Base,            QColor(255, 255, 255))
        pal.setColor(QPalette.AlternateBase,   QColor(245, 245, 245))
        pal.setColor(QPalette.Text,            QColor(0, 0, 0))
        pal.setColor(QPalette.Button,          QColor(240, 240, 240))
        pal.setColor(QPalette.ButtonText,      QColor(0, 0, 0))
        pal.setColor(QPalette.ToolTipBase,     QColor(255, 255, 220))
        pal.setColor(QPalette.ToolTipText,     QColor(0, 0, 0))
        pal.setColor(QPalette.Highlight,       QColor(70, 110, 180))
        pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(pal)


# Phase / progress detection on DLC's stdout. We don't depend on any
# single DLC version's exact format -- if nothing matches, the overlay
# stays in indeterminate-busy mode and the status label shows the last
# recognised phase. Patterns ordered most-specific first.
_TRAINING_PHASES = [
    (re.compile(r"extract_frames|extracting frame", re.IGNORECASE), "Extracting frames"),
    (re.compile(r"create_training_dataset|creating training", re.IGNORECASE), "Creating training dataset"),
    (re.compile(r"initialize.*weights|loading.*snapshot", re.IGNORECASE), "Initializing weights"),
    (re.compile(r"started training|train_network|begin training", re.IGNORECASE), "Training network"),
    (re.compile(r"evaluate_network|evaluating", re.IGNORECASE), "Evaluating snapshots"),
    (re.compile(r"analyze_videos|analyzing video", re.IGNORECASE), "Analyzing videos"),
    (re.compile(r"create_labeled_video|labeled video", re.IGNORECASE), "Creating labeled video"),
]
# LK-RSTC jitter reduction emits tqdm bars whose desc= strings are the
# only stable signal for which half of the loop we're in (submit vs.
# collect). Match the desc prefix so we don't depend on tqdm's exact
# bar / spinner glyphs.
_JITTER_PHASES = [
    # Post-2026-05-21 the parallel path uses a single fused bar
    # ("Processing tracking jobs") instead of separate Submitting /
    # Processing phases. Old labels kept as fallbacks so a stale
    # installation that still runs the two-phase code path renders
    # cleanly under the Qt overlay.
    (re.compile(r"Processing tracking jobs", re.IGNORECASE), "Processing tracking jobs"),
    (re.compile(r"Submitting jobs", re.IGNORECASE), "Submitting tracking jobs"),
    (re.compile(r"Processing results", re.IGNORECASE), "Processing tracking results"),
    (re.compile(r"Processing sequentially", re.IGNORECASE), "Processing sequentially"),
]
# DLC project creation chatters about copying videos and writing the
# config; useful as phase labels even when the operation completes in
# under a second.
_CREATE_PROJECT_PHASES = [
    (re.compile(r"Created.*\bproject\b|new project", re.IGNORECASE), "Project skeleton created"),
    (re.compile(r"adding.*video|copying.*video", re.IGNORECASE), "Copying video"),
    (re.compile(r"config.*yaml|writing.*config", re.IGNORECASE), "Writing config"),
    (re.compile(r"labeled-data|extract", re.IGNORECASE), "Preparing labeled-data folders"),
]
_SEED_PROJECT_PHASES = _CREATE_PROJECT_PHASES + [
    (re.compile(r"installing seed bundle", re.IGNORECASE), "Installing seed bundle"),
    (re.compile(r"analyze_videos|analyzing video", re.IGNORECASE), "Analyzing videos"),
]
_PROGRESS_PATTERNS = [
    re.compile(r"[Ee]poch\s+(\d+)\s*/\s*(\d+)"),
    re.compile(r"iteration[:\s=]+(\d+)\s*/\s*(\d+)", re.IGNORECASE),
    re.compile(r"\b(\d+)\s*/\s*(\d+)\s*\[", re.IGNORECASE),  # tqdm-style "  3/100 ["
]


class _Tee:
    """Fan-out writer: forwards every write+flush to multiple sinks.

    Used to route DLC's stdout/stderr during training to (1)
    ``sys.__stdout__`` so the user sees output in the launching terminal
    even when the button-click handler is running inside the Qt event
    loop, and (2) a thread-safe queue drained by the GUI to update the
    in-app overlay log.

    Why ``sys.__stdout__`` rather than the current ``sys.stdout``: in
    some launch contexts (IPython, certain IDE consoles) ``sys.stdout``
    is a buffered proxy that delays flushes until the event loop yields;
    ``sys.__stdout__`` is the original file descriptor and prints
    immediately.
    """

    def __init__(self, *streams):
        self._streams = [s for s in streams if s is not None]

    def write(self, s):
        for st in self._streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass
        return len(s) if isinstance(s, str) else 0

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self):
        return False


class _QueueWriter:
    """File-like sink that pushes each write into a queue (non-blocking)."""

    def __init__(self, q: queue.Queue):
        self._q = q

    def write(self, s):
        if s:
            self._q.put(s)
        return len(s) if isinstance(s, str) else 0

    def flush(self):
        pass

    def isatty(self):
        return False


def _make_progress_overlay_class():
    """Build :class:`ProgressOverlay` lazily so importing ``dustrack``
    never touches qtpy (and therefore never fails on a no-Qt-binding
    machine). Mirrors :func:`datanavigator._qt._make_qt_text_overlay_class`.
    """
    from qtpy.QtCore import QEvent, QObject, Qt
    from qtpy.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QLabel,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QVBoxLayout,
    )

    class ProgressOverlay(QObject):
        """Semi-transparent modal-feeling overlay parented to the DUSTrack
        QMainWindow. Shows a title, phase / status line, an optional
        progress bar (indeterminate until we parse a known progress
        format), and a scrolling tail of stdout. On completion the
        overlay transitions to a "done" state with a Done button so the
        user can review the final output before the underlying UI
        becomes interactive again.

        The overlay is a child ``QFrame`` of the main window, sized to
        cover it, and re-positioned on resize via an event filter. The
        ``QFrame`` itself accepts focus and swallows mouse events,
        preventing clicks from reaching the underlying buttons / canvas
        without us having to disable them individually.
        """

        def __init__(
            self,
            main_window,
            *,
            title: str = "Working",
            initial_phase: str = "Starting up",
            hint: str = "",
            show_progress_bar: bool = True,
        ):
            super().__init__(main_window)
            self._mw = main_window
            self._show_progress_bar = show_progress_bar
            self._done_callback = None

            self._frame = QFrame(main_window)
            self._frame.setObjectName("dustrack_progress_overlay")
            self._frame.setStyleSheet(
                "#dustrack_progress_overlay { background-color: rgba(0, 0, 0, 200); }"
                "QLabel { color: white; }"
                "QPlainTextEdit { "
                "  background-color: rgba(0, 0, 0, 220); "
                "  color: #cccccc; "
                "  font-family: 'Consolas', 'Courier New', monospace; "
                "  font-size: 10pt; "
                "  border: 1px solid #555555; "
                "}"
                "QProgressBar { "
                "  border: 1px solid #888888; background-color: #1a1a1a; "
                "  color: white; text-align: center; height: 18px; "
                "}"
                "QProgressBar::chunk { background-color: #3a86ff; }"
                "QPushButton { "
                "  background-color: #3a86ff; color: white; "
                "  border: 1px solid #2a76ef; padding: 6px 24px; "
                "  font-size: 11pt; font-weight: bold; "
                "}"
                "QPushButton:hover { background-color: #4a96ff; }"
                "QPushButton:pressed { background-color: #2a76ef; }"
            )
            self._frame.setFocusPolicy(Qt.StrongFocus)

            layout = QVBoxLayout(self._frame)
            layout.setAlignment(Qt.AlignCenter)
            layout.addStretch(1)

            self._title = QLabel(title)
            self._title.setAlignment(Qt.AlignCenter)
            self._title.setStyleSheet("font-size: 22pt; font-weight: bold;")
            layout.addWidget(self._title)

            self._phase = QLabel(initial_phase)
            self._phase.setAlignment(Qt.AlignCenter)
            self._phase.setStyleSheet("font-size: 12pt;")
            layout.addWidget(self._phase)

            self._progress = QProgressBar()
            self._progress.setRange(0, 0)  # indeterminate (busy) until we parse
            self._progress.setFixedWidth(480)
            self._progress.setAlignment(Qt.AlignCenter)
            self._progress.setSizePolicy(self._progress.sizePolicy())
            row = QVBoxLayout()
            row.setAlignment(Qt.AlignCenter)
            row.addWidget(self._progress, alignment=Qt.AlignCenter)
            layout.addLayout(row)
            if not show_progress_bar:
                self._progress.hide()

            self._log = QPlainTextEdit()
            self._log.setReadOnly(True)
            self._log.setMaximumBlockCount(400)
            self._log.setFixedWidth(720)
            self._log.setFixedHeight(220)
            layout.addWidget(self._log, alignment=Qt.AlignCenter)

            self._hint = QLabel(
                hint
                or "Output is also streamed to the launching terminal."
            )
            self._hint.setAlignment(Qt.AlignCenter)
            self._hint.setStyleSheet("color: #aaaaaa; font-size: 9pt;")
            layout.addWidget(self._hint)

            # Done button is created up front but hidden until mark_done()
            # so the layout doesn't shift when the work completes.
            button_row = QHBoxLayout()
            button_row.setAlignment(Qt.AlignCenter)
            self._done_button = QPushButton("Done")
            self._done_button.setFixedWidth(160)
            self._done_button.clicked.connect(self._on_done_clicked)
            self._done_button.hide()
            button_row.addWidget(self._done_button)
            layout.addLayout(button_row)

            layout.addStretch(1)

            main_window.installEventFilter(self)

            self._frame.show()
            self._reposition()
            self._frame.raise_()
            self._frame.setFocus()

        def eventFilter(self, obj, event):  # noqa: N802 (Qt API)
            if obj is self._mw and event.type() == QEvent.Resize:
                self._reposition()
            return False

        def _reposition(self):
            self._frame.setGeometry(0, 0, self._mw.width(), self._mw.height())
            self._frame.raise_()

        def append_log(self, line: str):
            line = line.rstrip("\r\n")
            if line:
                self._log.appendPlainText(line)

        def set_phase(self, text: str):
            self._phase.setText(text)

        def set_progress(self, current: int, total: int):
            if not self._show_progress_bar:
                return
            if total > 0:
                if self._progress.maximum() == 0:
                    self._progress.setRange(0, total)
                elif self._progress.maximum() != total:
                    self._progress.setRange(0, total)
                self._progress.setValue(min(current, total))
                self._progress.setFormat(f"{current} / {total} (%p%)")

        def mark_done(self, success: bool, summary: str, on_done=None):
            """Switch to the "complete" state: update title + phase to the
            given ``summary`` and reveal the Done button. ``on_done`` runs
            on the GUI thread when the user clicks Done (after the overlay
            dismisses itself). Safe to call from the GUI thread only.
            """
            self._done_callback = on_done
            self._title.setText("Complete" if success else "Failed")
            self._title.setStyleSheet(
                "font-size: 22pt; font-weight: bold; "
                f"color: {'#7cdb7c' if success else '#ff7c7c'};"
            )
            self._phase.setText(summary)
            if self._show_progress_bar:
                if success:
                    # Fill the bar to convey "complete". If we never had
                    # a determinate range, set a 1/1 so the chunk paints.
                    if self._progress.maximum() == 0:
                        self._progress.setRange(0, 1)
                        self._progress.setValue(1)
                    else:
                        self._progress.setValue(self._progress.maximum())
                    self._progress.setFormat("Complete")
                else:
                    self._progress.setRange(0, 1)
                    self._progress.setValue(0)
                    self._progress.setFormat("Failed")
            self._hint.setText(
                "Review the output above, then click Done to continue."
            )
            self._done_button.show()
            self._done_button.setFocus()

        def _on_done_clicked(self):
            cb = self._done_callback
            self._done_callback = None
            self.dismiss()
            if cb is not None:
                try:
                    cb()
                except Exception as exc:  # noqa: BLE001
                    sys.__stderr__.write(
                        f"Post-overlay callback raised: {exc}\n"
                    )

        def dismiss(self):
            try:
                self._mw.removeEventFilter(self)
            except Exception:
                pass
            self._frame.hide()
            self._frame.deleteLater()

    return ProgressOverlay


def _make_confirm_overlay_class():
    """Build :class:`ConfirmOverlay` lazily, mirroring
    :func:`_make_progress_overlay_class`'s qtpy-import-on-demand pattern.

    Sibling to :class:`ProgressOverlay`: shares the backdrop-frame +
    reposition + event-filter + dark-translucent scaffolding, but is
    synchronous (``exec_()`` runs a local ``QEventLoop`` and returns
    the clicked button's label string) rather than async. Replaces the
    pre-rc2 ``QMessageBox`` sites (``_prompt_unified_pre_flight``,
    ``_prompt_save_on_close``) and hosts the new rc2 confirms
    (``discard_unsaved_annotations``, ``remove_current_layer``).
    """
    from qtpy.QtCore import QEvent, QEventLoop, QObject, Qt
    from qtpy.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
    )

    # Severity tint reuses ProgressOverlay's done/failed palette
    # (#7cdb7c / #ff7c7c, see ProgressOverlay.mark_done) so the two
    # overlays share a visual vocabulary.
    _SEVERITY_TITLE_COLOR = {
        "info": "white",
        "warning": "#7cdb7c",
        "destructive": "#ff7c7c",
    }
    _ROLE_QSS = {
        "primary": (
            "QPushButton { background-color: #3a86ff; color: white; "
            "  border: 1px solid #2a76ef; padding: 6px 24px; "
            "  font-size: 11pt; font-weight: bold; }"
            "QPushButton:hover { background-color: #4a96ff; }"
            "QPushButton:pressed { background-color: #2a76ef; }"
        ),
        "destructive": (
            "QPushButton { background-color: #ff7c7c; color: white; "
            "  border: 1px solid #df5c5c; padding: 6px 24px; "
            "  font-size: 11pt; font-weight: bold; }"
            "QPushButton:hover { background-color: #ff9c9c; }"
            "QPushButton:pressed { background-color: #df5c5c; }"
        ),
        "neutral": (
            "QPushButton { background-color: #555555; color: white; "
            "  border: 1px solid #444444; padding: 6px 24px; "
            "  font-size: 11pt; }"
            "QPushButton:hover { background-color: #666666; }"
            "QPushButton:pressed { background-color: #444444; }"
        ),
    }

    class ConfirmOverlay(QObject):
        """Modal confirm overlay parented to the DUSTrack QMainWindow.

        Synchronous: :meth:`exec_` runs a local :class:`QEventLoop`
        and returns the label string of the clicked button. Mirrors
        :class:`ProgressOverlay`'s parented-QFrame + reposition +
        event-filter + dark-translucent scaffolding.

        Example::

            result = ConfirmOverlay(
                qt_window,
                title="Remove layer",
                message="...severity-aware body...",
                buttons=[
                    ("Remove layer", "destructive"),
                    ("Cancel", "neutral"),
                ],
                default="Cancel",
                severity="warning",
            ).exec_()
            if result == "Remove layer":
                ...

        ``severity`` picks the title color
        (``info`` / ``warning`` / ``destructive``);
        ``buttons`` is an ordered list of ``(label, role)`` pairs where
        ``role`` styles the button (``primary`` / ``destructive`` /
        ``neutral``); ``default`` names the button that receives focus
        (and is the implicit Enter / Esc target -- ``Cancel`` is the
        usual choice for destructive flows).
        """

        def __init__(
            self,
            main_window,
            *,
            title: str,
            message: str,
            buttons,  # list[tuple[label, role]]
            default: str | None = None,
            severity: str = "info",
        ):
            super().__init__(main_window)
            self._mw = main_window
            self._result: str | None = None
            self._loop = QEventLoop()

            self._frame = QFrame(main_window)
            self._frame.setObjectName("dustrack_confirm_overlay")
            title_color = _SEVERITY_TITLE_COLOR.get(severity, "white")
            self._frame.setStyleSheet(
                "#dustrack_confirm_overlay { background-color: rgba(0, 0, 0, 200); }"
                "QLabel { color: white; }"
                f"#dustrack_confirm_title {{ color: {title_color}; "
                "  font-size: 22pt; font-weight: bold; }"
                "#dustrack_confirm_message { font-size: 12pt; }"
            )
            self._frame.setFocusPolicy(Qt.StrongFocus)

            layout = QVBoxLayout(self._frame)
            layout.setAlignment(Qt.AlignCenter)
            layout.addStretch(1)

            title_lbl = QLabel(title)
            title_lbl.setObjectName("dustrack_confirm_title")
            title_lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(title_lbl)

            message_lbl = QLabel(message)
            message_lbl.setObjectName("dustrack_confirm_message")
            message_lbl.setAlignment(Qt.AlignCenter)
            message_lbl.setWordWrap(True)
            # Cap the message width so long bodies wrap rather than
            # stretching the overlay to the full window width.
            message_lbl.setMaximumWidth(720)
            layout.addWidget(message_lbl, alignment=Qt.AlignCenter)

            button_row = QHBoxLayout()
            button_row.setAlignment(Qt.AlignCenter)
            self._buttons = []
            default_btn = None
            for label, role in buttons:
                btn = QPushButton(label)
                btn.setMinimumWidth(160)
                # Per-button QSS rather than parent QSS so each role's
                # palette stays scoped; QSS on a QPushButton replaces
                # native gradient (see memory feedback_qt_qss_vs_palette).
                btn.setStyleSheet(_ROLE_QSS.get(role, _ROLE_QSS["neutral"]))
                btn.clicked.connect(lambda _checked=False, lbl=label: self._on_clicked(lbl))
                button_row.addWidget(btn)
                self._buttons.append(btn)
                if default is not None and label == default:
                    default_btn = btn
            layout.addLayout(button_row)
            layout.addStretch(1)

            main_window.installEventFilter(self)

            self._frame.show()
            self._reposition()
            self._frame.raise_()
            # Focus the default button if named; otherwise focus the
            # frame so Esc routes through eventFilter and the buttons
            # are reachable via Tab.
            if default_btn is not None:
                default_btn.setFocus()
            else:
                self._frame.setFocus()

        def eventFilter(self, obj, event):  # noqa: N802 (Qt API)
            if obj is self._mw and event.type() == QEvent.Resize:
                self._reposition()
            return False

        def _reposition(self):
            self._frame.setGeometry(0, 0, self._mw.width(), self._mw.height())
            self._frame.raise_()

        def _on_clicked(self, label: str) -> None:
            self._result = label
            self._dismiss()
            self._loop.quit()

        def _dismiss(self) -> None:
            try:
                self._mw.removeEventFilter(self)
            except Exception:
                pass
            self._frame.hide()
            self._frame.deleteLater()

        def exec_(self) -> str | None:
            """Block until the user clicks a button; return its label.

            Returns ``None`` only if the overlay is dismissed
            externally (e.g. main window close); every normal flow
            returns the clicked button's label string.
            """
            self._loop.exec_()
            return self._result

    return ConfirmOverlay


# Pure-function slider-to-param maps for the EnhanceWidget. Extracted
# at module scope so they're unit-testable without qtpy import.
_CLAHE_CLIP_MIN = 1.0
_CLAHE_CLIP_MAX = 4.0
_GAMMA_MIN = 1.0
_GAMMA_MAX = 2.0  # extended 2026-05-19 (was 1.5) for darker ultrasound footage
_SLIDER_TICKS = 100  # integer slider range; sliders use 0..100


def _slider_to_clahe_clip(slider_value: int) -> float:
    """Map slider integer 0..100 to CLAHE clip limit in [1.0, 4.0]."""
    t = max(0, min(_SLIDER_TICKS, int(slider_value))) / _SLIDER_TICKS
    return _CLAHE_CLIP_MIN + t * (_CLAHE_CLIP_MAX - _CLAHE_CLIP_MIN)


def _slider_to_gamma(slider_value: int) -> float:
    """Map slider integer 0..100 to gamma in [1.0, 1.5]."""
    t = max(0, min(_SLIDER_TICKS, int(slider_value))) / _SLIDER_TICKS
    return _GAMMA_MIN + t * (_GAMMA_MAX - _GAMMA_MIN)


def _clahe_clip_to_slider(clip: float) -> int:
    """Inverse of :func:`_slider_to_clahe_clip` -- seed slider from current param."""
    span = _CLAHE_CLIP_MAX - _CLAHE_CLIP_MIN
    if span <= 0:
        return 0
    t = (float(clip) - _CLAHE_CLIP_MIN) / span
    return max(0, min(_SLIDER_TICKS, round(t * _SLIDER_TICKS)))


def _gamma_to_slider(gamma: float) -> int:
    """Inverse of :func:`_slider_to_gamma` -- seed slider from current param."""
    span = _GAMMA_MAX - _GAMMA_MIN
    if span <= 0:
        return 0
    t = (float(gamma) - _GAMMA_MIN) / span
    return max(0, min(_SLIDER_TICKS, round(t * _SLIDER_TICKS)))


def _apply_gamma_only(image, gamma: float):
    """Apply a gamma LUT to ``image`` without touching CLAHE or
    grayscale conversion.

    Per-slider bypass for the EnhanceWidget: when the user moves the
    gamma slider off zero with the clip slider still at zero, we
    don't want to spin up the full CLAHE pipeline (which forces an
    RGB->gray->RGB roundtrip + a CLAHE histogram pass at clip=1.0
    that *isn't* a true identity). This helper operates per-channel
    on the input directly via ``cv.LUT``, so moving gamma off 1.0
    transitions smoothly from raw -- the inverse-gamma LUT at
    gamma=1.0+epsilon differs from the identity LUT by less than 1
    unit in any bin, so the rendered frame is visually continuous
    with the bypassed state.

    Mirrors the gamma branch inside :func:`enhance_ultrasound_image`
    but operates on the original (possibly RGB) array rather than
    the grayscale intermediate.
    """
    import cv2 as cv
    inv_gamma = 1.0 / float(gamma)
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype("uint8")
    return cv.LUT(image, table)


def _enhance_is_passthrough(clahe_clip: float, gamma: float) -> bool:
    """True if the enhancement pipeline should be bypassed entirely.

    Returns ``True`` when both sliders sit at their minimum (the
    "no enhancement" position): ``clahe_clip <= 1.0`` AND
    ``gamma <= 1.0``. At that point :func:`enhance_ultrasound_image`
    would still run a CLAHE pass at clip=1.0 (minimal but non-zero
    effect) and would convert RGB->gray->RGB unconditionally, so the
    "bypass" semantic is implemented by short-circuiting at the
    image processor level rather than by tuning the parameter
    extremes. Replaces the pre-2026-05-19 ``_enhance_enabled``
    toggle that the deleted "Toggle enhance" button flipped.
    """
    return float(clahe_clip) <= _CLAHE_CLIP_MIN and float(gamma) <= _GAMMA_MIN


# Anchor points for the auto-enhance heuristic. Tuned 2026-05-19
# against real ultrasound footage in four passes; each pass was the
# user's "too aggressive" reaction to the previous one:
#
# - Pass 1 (LOW=40 / HIGH=180, DARK=50 / MID=130).
# - Pass 2 (LOW=0 / HIGH=120, DARK=20 / MID=90).
# - Pass 3 (LOW=0 / HIGH=100, DARK=20 / MID=75).
# - Pass 4 (current, LOW=0 / HIGH=75, DARK=0 / MID=25): bundled with
#   the gamma-max extension to 2.0. Calibrated against the S-corpus
#   DUSTrack clip where the user-target was clip~1.6, gamma~1.3 and
#   pass-3 was producing clip=2.17, gamma=1.5(capped). Inferred
#   stats: dyn-range ~61, p50 ~20. With pass-4 anchors and
#   gamma_max=2.0 those stats land at clip~1.56, gamma~1.20 --
#   below user target but close enough that the user dials up from
#   there. Anchors deliberately make "typical" ultrasound
#   (p50~60, dyn~80) a near-bypass (clip~1.0, gamma=1.0); Auto
#   only kicks in noticeably for dark + low-contrast frames.
# Adjust here, not in callers.
_AUTO_DYN_RANGE_LOW = 0.0     # at this dyn range, suggest clip=max
_AUTO_DYN_RANGE_HIGH = 75.0   # at this dyn range, suggest clip=min
_AUTO_MEDIAN_DARK = 0.0       # at this median, suggest gamma=max
_AUTO_MEDIAN_MID = 25.0       # at this median, suggest gamma=min


def _auto_enhance_params(image) -> tuple[float, float]:
    """Heuristic ``(clip, gamma)`` from the current frame.

    One-shot inference, called by :class:`EnhanceWidget`'s ``Auto``
    button. Reads the grayscale histogram of ``image`` and maps
    two robust statistics to slider-range parameters:

    - **CLAHE clip** is driven by the 5th-to-95th percentile dynamic
      range. Narrow dynamic range (a flat-looking frame) suggests
      pushing clip high; wide dynamic range (already-contrasty
      frame) suggests leaving clip low. Current anchors (pass 4):
      ``dyn=0 -> clip=_CLAHE_CLIP_MAX``,
      ``dyn=75 -> clip=_CLAHE_CLIP_MIN``.
    - **Gamma** is driven by the 50th percentile (median). A dark
      frame (low median) suggests pushing gamma high to lift the
      midtones; a balanced frame suggests gamma=1.0. Current
      anchors (pass 4):
      ``p50=0 -> gamma=_GAMMA_MAX``,
      ``p50=25 -> gamma=_GAMMA_MIN``.

    Both outputs are clamped to their slider ranges
    (``[_CLAHE_CLIP_MIN, _CLAHE_CLIP_MAX]`` and
    ``[_GAMMA_MIN, _GAMMA_MAX]``) so a degenerate frame (all-zero,
    all-white) can't drive the sliders past their ends.

    Pure function -- accepts a uint8 RGB or grayscale numpy array,
    returns ``(clip, gamma)`` floats. Lives at module scope so
    :file:`tests/test_enhance_widget_mapping.py` can unit-test the
    heuristic without standing up a Qt main loop.
    """
    import cv2 as cv  # localized import: keeps the test path importable on no-cv2 envs
    if image.ndim == 3:
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    else:
        gray = image
    p5, p50, p95 = (float(x) for x in np.percentile(gray, [5, 50, 95]))

    dyn_range = p95 - p5
    t_clip = (_AUTO_DYN_RANGE_HIGH - dyn_range) / (
        _AUTO_DYN_RANGE_HIGH - _AUTO_DYN_RANGE_LOW
    )
    t_clip = max(0.0, min(1.0, t_clip))
    clip = _CLAHE_CLIP_MIN + t_clip * (_CLAHE_CLIP_MAX - _CLAHE_CLIP_MIN)

    t_gamma = (_AUTO_MEDIAN_MID - p50) / (_AUTO_MEDIAN_MID - _AUTO_MEDIAN_DARK)
    t_gamma = max(0.0, min(1.0, t_gamma))
    gamma = _GAMMA_MIN + t_gamma * (_GAMMA_MAX - _GAMMA_MIN)

    return float(clip), float(gamma)


def _make_enhance_widget_class():
    """Build :class:`EnhanceWidget` lazily, mirroring
    :func:`_make_progress_overlay_class`'s qtpy-import-on-demand pattern.

    Two-slider control mounted in the rc2 left-column dock below the
    statevars widget. CLAHE clip (1.0 -> 4.0) and gamma (1.0 -> 1.5).
    Brightness and CLAHE grid (8) stay at their constructor defaults --
    both ride below the "useful slider range" threshold for routine
    review. The widget itself owns no enable/disable state: both
    sliders at their minimum (clip=1.0 AND gamma=1.0) is the bypass
    -- :func:`_enhance_is_passthrough` short-circuits the image
    processor and returns the raw frame untouched.
    """
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider,
        QVBoxLayout, QWidget,
    )

    class EnhanceWidget(QWidget):
        """Two-slider widget for ultrasound image enhancement.

        Slider 1 -- CLAHE clip:  1.0 -> 4.0, default 2.0.
        Slider 2 -- Gamma:       1.0 -> 1.5, default 1.2.

        Sliders update ``self._owner._clahe_clip`` / ``_gamma`` directly
        on every value change and trigger ``self._owner.update()`` so
        the image redraws live.
        """

        def __init__(self, owner, parent=None):
            super().__init__(parent)
            self._owner = owner

            # Slightly darker bg than the parent dock so the enhance
            # section reads as a distinct group from the statevars
            # widget above it. Theme-adaptive via palette.darker().
            self.setAutoFillBackground(True)
            pal = self.palette()
            base = pal.color(self.backgroundRole())
            pal.setColor(self.backgroundRole(), base.darker(110))
            self.setPalette(pal)

            self.setFocusPolicy(Qt.NoFocus)
            self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

            outer = QVBoxLayout(self)
            outer.setContentsMargins(4, 4, 4, 4)
            outer.setSpacing(4)

            section_label = QLabel("Image enhance:", self)
            outer.addWidget(section_label)

            # CLAHE clip slider row.
            self._clip_label = QLabel(
                f"Clip: {float(owner._clahe_clip):.2f}", self,
            )
            outer.addWidget(self._clip_label)
            self._clip_slider = QSlider(Qt.Horizontal, self)
            self._clip_slider.setRange(0, _SLIDER_TICKS)
            self._clip_slider.setValue(_clahe_clip_to_slider(owner._clahe_clip))
            self._clip_slider.setFocusPolicy(Qt.NoFocus)
            self._clip_slider.valueChanged.connect(self._on_clip_changed)
            outer.addWidget(self._clip_slider)

            # Gamma slider row.
            self._gamma_label = QLabel(
                f"Gamma: {float(owner._gamma):.2f}", self,
            )
            outer.addWidget(self._gamma_label)
            self._gamma_slider = QSlider(Qt.Horizontal, self)
            self._gamma_slider.setRange(0, _SLIDER_TICKS)
            self._gamma_slider.setValue(_gamma_to_slider(owner._gamma))
            self._gamma_slider.setFocusPolicy(Qt.NoFocus)
            self._gamma_slider.valueChanged.connect(self._on_gamma_changed)
            outer.addWidget(self._gamma_slider)

            # One-shot trigger row: [None | Auto].
            # - None: snap both sliders to leftmost (passthrough).
            # - Auto: infer clip + gamma from the current frame's
            #   grayscale histogram, set the sliders once, redraw
            #   once. Subsequent frame navigations don't re-trigger
            #   Auto -- slider values stay put until the user
            #   (or another button click) moves them.
            button_row = QHBoxLayout()
            button_row.setContentsMargins(0, 0, 0, 0)
            button_row.setSpacing(4)
            self._none_button = QPushButton("None", self)
            self._none_button.setFocusPolicy(Qt.NoFocus)
            self._none_button.clicked.connect(self._on_none_clicked)
            button_row.addWidget(self._none_button)
            self._auto_button = QPushButton("Auto", self)
            self._auto_button.setFocusPolicy(Qt.NoFocus)
            self._auto_button.clicked.connect(self._on_auto_clicked)
            button_row.addWidget(self._auto_button)
            outer.addLayout(button_row)

        def _on_clip_changed(self, value: int) -> None:
            clip = _slider_to_clahe_clip(value)
            self._owner._clahe_clip = clip
            self._clip_label.setText(f"Clip: {clip:.2f}")
            self._owner.update()

        def _on_gamma_changed(self, value: int) -> None:
            gamma = _slider_to_gamma(value)
            self._owner._gamma = gamma
            self._gamma_label.setText(f"Gamma: {gamma:.2f}")
            self._owner.update()

        def _apply_param_pair(self, clip: float, gamma: float) -> None:
            """Set both sliders to (clip, gamma); one redraw at the end.

            Shared tail for the ``None`` and ``Auto`` button handlers.
            Slider signals are blocked during the dual ``setValue`` so
            the two ``valueChanged`` callbacks don't each fire
            ``owner.update()``; the redraw happens once. Slider
            positions are integer-quantized, so the actually-applied
            values are read back off the sliders to keep the labels
            + the owner's enhancement params in sync with the slider
            UI state of truth.
            """
            self._clip_slider.blockSignals(True)
            self._gamma_slider.blockSignals(True)
            self._clip_slider.setValue(_clahe_clip_to_slider(clip))
            self._gamma_slider.setValue(_gamma_to_slider(gamma))
            self._clip_slider.blockSignals(False)
            self._gamma_slider.blockSignals(False)

            actual_clip = _slider_to_clahe_clip(self._clip_slider.value())
            actual_gamma = _slider_to_gamma(self._gamma_slider.value())
            self._owner._clahe_clip = actual_clip
            self._owner._gamma = actual_gamma
            self._clip_label.setText(f"Clip: {actual_clip:.2f}")
            self._gamma_label.setText(f"Gamma: {actual_gamma:.2f}")
            self._owner.update()

        def _on_none_clicked(self) -> None:
            """Snap both sliders to leftmost (= passthrough).

            Convenience: undoes any Auto/manual enhancement in one
            click. The image processor's
            :func:`_enhance_is_passthrough` short-circuit fires after
            the redraw, so the next frame renders raw.
            """
            self._apply_param_pair(_CLAHE_CLIP_MIN, _GAMMA_MIN)

        def _on_auto_clicked(self) -> None:
            """One-shot auto-enhance from the current raw frame.

            Reads ``owner.data[owner._current_idx]`` (the same raw
            frame the image processor sees), runs
            :func:`_auto_enhance_params`, and applies the result.
            """
            owner = self._owner
            try:
                raw = owner.data[owner._current_idx].asnumpy()
            except Exception:
                # No frame available (e.g. video reader torn down):
                # surface nothing, leave sliders alone. Auto is best-
                # effort UI; a failure here shouldn't crash the
                # session.
                return
            clip, gamma = _auto_enhance_params(raw)
            self._apply_param_pair(clip, gamma)

    return EnhanceWidget


# ---------------------------------------------------------------------------
# Training options modal (Train DLC model click flow, 1.2.0a2)
# ---------------------------------------------------------------------------
#
# The GUI Train button opens a Training options modal *before* the
# existing pre-flight scan. The modal lets the user pick refine_mode +
# source iteration/snapshot (or external snapshot path) + epochs +
# create_video explicitly, then hands the choices to
# :meth:`DLCProject.train_iteration` (the explicit-args sibling of
# ``process()``).
#
# Two pure helpers + one factory:
# - :func:`_default_training_options` reads project state and returns
#   the initial dialog state dict.
# - :func:`_training_options_to_train_iteration_kwargs` translates the
#   user-modified dialog state to ``train_iteration`` kwargs.
# - :func:`_make_training_options_class` builds the Qt dialog lazily.


def _default_training_options(dlcproject):
    """Initial state dict for the Training options modal.

    Pure-Python, no Qt deps -- testable in isolation by passing a stub
    that exposes the same surface as a real :class:`DLCProject`
    (``all_snapshots``). Reads the module-level ``DLC3`` flag for the
    training-duration default.

    Defaults:

    * ``refine_mode``: ``"in_project"`` if any iteration is trained;
      otherwise ``"scratch"`` (the only mode that works without a
      source).
    * ``source_iteration``: the latest trained iteration; ``None`` if
      none are trained.
    * ``source_snapshot``: ``None`` (lets :meth:`initialize_weights`
      pick the best snapshot).
    * ``external_snapshot_path``: empty string -- the user picks via
      Browse... in the modal.
    * ``maxiters``: 50 (DLC3) or 500000 (DLC2), matching
      :meth:`DLCProject.process` / :meth:`DLCProject.train_iteration`
      defaults.
    * ``create_video``: ``False`` -- UI ergonomics default.

    Also returns ``trained_iterations`` (sorted) and
    ``snapshots_by_iteration`` (dict) so the dialog can populate the
    combos without going back to the project. ``is_dlc3`` is a
    snapshot of the module flag so the dialog's spinbox label
    ("epochs" vs "iterations") and the Browse filter (``.pt`` vs
    ``.index``) don't have to import this module.

    Returns:
        dict: Initial state for the Training options modal.
    """
    trained = sorted(
        i for i, snaps in dlcproject.all_snapshots.items() if snaps
    )
    snapshots_by_iteration = {
        i: list(dlcproject.all_snapshots[i]) for i in trained
    }
    has_trained = bool(trained)
    return {
        "refine_mode": "in_project" if has_trained else "scratch",
        "source_iteration": trained[-1] if has_trained else None,
        "source_snapshot": None,  # None == "best" (initialize_weights default)
        "external_snapshot_path": "",
        "maxiters": 50 if DLC3 else 500000,
        "create_video": False,
        # Combo population helpers; not forwarded to train_iteration.
        "trained_iterations": trained,
        "snapshots_by_iteration": snapshots_by_iteration,
        "is_dlc3": bool(DLC3),
    }


def _training_options_to_train_iteration_kwargs(options):
    """Translate Training options dialog state to
    :meth:`DLCProject.train_iteration` kwargs.

    Pure-Python translation. ``options`` is the dict returned by the
    modal's ``exec_()`` after the user clicks Train -- a copy of
    :func:`_default_training_options`'s output with the user-picked
    values applied. Discriminates by ``options['refine_mode']`` and
    only forwards keys that ``train_iteration``'s validator accepts
    for that mode (``source_*`` keys would be rejected in scratch
    mode, etc.).

    Returns:
        dict: kwargs ready to splat into ``train_iteration(...)``.
    """
    mode = options["refine_mode"]
    common = {
        "refine_mode": mode,
        "maxiters": options["maxiters"],
        "create_video": options["create_video"],
    }
    if mode == "scratch":
        return common
    if mode == "in_project":
        return {
            **common,
            "source_iteration": options["source_iteration"],
            "source_snapshot": options["source_snapshot"],
        }
    if mode == "external":
        return {
            **common,
            "external_snapshot_path": options["external_snapshot_path"],
        }
    raise ValueError(f"unknown refine_mode in options: {mode!r}")


def _make_training_options_class():
    """Build :class:`TrainingOptionsDialog` lazily, mirroring
    :func:`_make_confirm_overlay_class`'s qtpy-import-on-demand pattern.

    Modal dialog parented to the DUSTrack QMainWindow, shown when the
    user clicks Train DLC model. Lets the user pick refine_mode +
    source (in-project iteration/snapshot OR external snapshot path) +
    epochs + create_video, returning the choices as a dict on Train
    or ``None`` on Cancel.

    Shares ConfirmOverlay's dark-translucent backdrop + reposition +
    event-filter scaffolding but holds richer form widgets (radio
    buttons, combos, line edit + Browse, spinbox, checkbox). The
    inner content QWidget deliberately carries NO ``QWidget { ... }``
    QSS so child QComboBox / QLineEdit / QSpinBox keep their native
    rendering (per memory ``feedback_qt_qss_vs_palette``).
    """
    from qtpy.QtCore import QEvent, QEventLoop, QObject, Qt
    from qtpy.QtWidgets import (
        QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFrame,
        QHBoxLayout, QLabel, QLineEdit, QPushButton, QRadioButton,
        QSpinBox, QVBoxLayout, QWidget,
    )

    # Reuse ConfirmOverlay's role QSS so Train/Cancel match the visual
    # vocab of the other rc2 modals.
    _ROLE_QSS = {
        "primary": (
            "QPushButton { background-color: #3a86ff; color: white; "
            "  border: 1px solid #2a76ef; padding: 6px 24px; "
            "  font-size: 11pt; font-weight: bold; }"
            "QPushButton:hover { background-color: #4a96ff; }"
            "QPushButton:pressed { background-color: #2a76ef; }"
        ),
        "neutral": (
            "QPushButton { background-color: #555555; color: white; "
            "  border: 1px solid #444444; padding: 6px 24px; "
            "  font-size: 11pt; }"
            "QPushButton:hover { background-color: #666666; }"
            "QPushButton:pressed { background-color: #444444; }"
        ),
    }

    class TrainingOptionsDialog(QObject):
        """Synchronous modal for the Train DLC model pre-flight.

        Example::

            dialog = TrainingOptionsDialog(qt_window, initial_state=...)
            options = dialog.exec_()
            if options is None:
                ...  # user cancelled
            else:
                kwargs = _training_options_to_train_iteration_kwargs(options)
                self._dlcproject.train_iteration(**kwargs)

        ``initial_state`` is the dict returned by
        :func:`_default_training_options`; the dialog seeds itself
        from it. On *Train*, ``exec_()`` returns a new dict with the
        user-modified values applied. On *Cancel*, returns ``None``.
        """

        def __init__(self, main_window, *, initial_state: dict):
            super().__init__(main_window)
            self._mw = main_window
            self._result = None
            self._loop = QEventLoop()
            self._state = dict(initial_state)

            self._frame = QFrame(main_window)
            self._frame.setObjectName("dustrack_training_options_overlay")
            self._frame.setStyleSheet(
                "#dustrack_training_options_overlay { "
                "  background-color: rgba(0, 0, 0, 200); "
                "}"
                "QLabel { color: white; }"
                "#dustrack_training_title { color: white; "
                "  font-size: 22pt; font-weight: bold; }"
                "QRadioButton { color: white; font-size: 11pt; }"
                "QCheckBox { color: white; font-size: 11pt; }"
                # Radio indicator -- on the dark overlay backdrop the
                # native Windows checked dot is barely visible (white
                # inner ring on a white indicator background). Render
                # the indicator as a white-bordered circle, filled
                # with the primary-action blue when checked. Matches
                # the Train button's #3a86ff so the "selected mode"
                # affordance is unmistakable.
                "QRadioButton::indicator { width: 14px; height: 14px; }"
                "QRadioButton::indicator:unchecked { "
                "  background-color: transparent; "
                "  border: 2px solid white; "
                "  border-radius: 8px; "
                "}"
                "QRadioButton::indicator:checked { "
                "  background-color: #3a86ff; "
                "  border: 2px solid white; "
                "  border-radius: 8px; "
                "}"
            )
            self._frame.setFocusPolicy(Qt.StrongFocus)

            outer = QVBoxLayout(self._frame)
            outer.setAlignment(Qt.AlignCenter)
            outer.addStretch(1)

            title_lbl = QLabel("Training options")
            title_lbl.setObjectName("dustrack_training_title")
            title_lbl.setAlignment(Qt.AlignCenter)
            outer.addWidget(title_lbl)

            # Inner content card: deliberately no QSS on QWidget {} so
            # child native controls (QComboBox / QLineEdit / QSpinBox)
            # keep their Windows-native rendering (avoids the QSS-
            # cascade trap from feedback_qt_qss_vs_palette).
            content = QWidget()
            content.setMaximumWidth(640)
            content_layout = QVBoxLayout(content)
            content_layout.setSpacing(12)

            # --- Refine mode radios ---
            self._refine_group = QButtonGroup(self)
            self._scratch_radio = QRadioButton("Start from scratch")
            self._in_project_radio = QRadioButton(
                "Refine from in-project iteration"
            )
            self._external_radio = QRadioButton(
                "Refine from external snapshot"
            )
            for i, rb in enumerate(
                (self._scratch_radio, self._in_project_radio, self._external_radio)
            ):
                self._refine_group.addButton(rb, i)
                content_layout.addWidget(rb)
                rb.toggled.connect(self._on_radio_toggled)

            # --- In-project source picker (indented sub-row) ---
            self._in_project_row = QWidget()
            ip_layout = QHBoxLayout(self._in_project_row)
            ip_layout.setContentsMargins(28, 0, 0, 0)
            ip_layout.addWidget(QLabel("Iteration:"))
            self._iter_combo = QComboBox()
            for it in initial_state.get("trained_iterations", []):
                self._iter_combo.addItem(f"iteration-{it}", userData=it)
            self._iter_combo.currentIndexChanged.connect(self._on_iteration_changed)
            ip_layout.addWidget(self._iter_combo, stretch=1)
            ip_layout.addWidget(QLabel("Snapshot:"))
            self._snap_combo = QComboBox()
            ip_layout.addWidget(self._snap_combo, stretch=1)
            content_layout.addWidget(self._in_project_row)

            # --- External source picker (indented sub-row) ---
            self._external_row = QWidget()
            ex_layout = QHBoxLayout(self._external_row)
            ex_layout.setContentsMargins(28, 0, 0, 0)
            ex_layout.addWidget(QLabel("Path:"))
            self._external_path_edit = QLineEdit(
                initial_state.get("external_snapshot_path", "")
            )
            ex_layout.addWidget(self._external_path_edit, stretch=1)
            browse_btn = QPushButton("Browse…")
            browse_btn.clicked.connect(self._on_browse_clicked)
            ex_layout.addWidget(browse_btn)
            content_layout.addWidget(self._external_row)

            # --- Training duration ---
            duration_row = QHBoxLayout()
            duration_row.addWidget(
                QLabel(
                    "Training epochs:" if initial_state.get("is_dlc3", True)
                    else "Training iterations:"
                )
            )
            self._maxiters_spin = QSpinBox()
            self._maxiters_spin.setRange(1, 10_000_000)
            self._maxiters_spin.setValue(int(initial_state["maxiters"]))
            duration_row.addWidget(self._maxiters_spin)
            duration_row.addStretch(1)
            content_layout.addLayout(duration_row)

            # --- Create labeled video toggle ---
            self._create_video_chk = QCheckBox(
                "Create labeled video on completion"
            )
            self._create_video_chk.setChecked(
                bool(initial_state["create_video"])
            )
            content_layout.addWidget(self._create_video_chk)

            # --- Train / Cancel buttons ---
            button_row = QHBoxLayout()
            button_row.setAlignment(Qt.AlignCenter)
            self._train_btn = QPushButton("Train")
            self._train_btn.setMinimumWidth(160)
            self._train_btn.setStyleSheet(_ROLE_QSS["primary"])
            self._train_btn.clicked.connect(self._on_train_clicked)
            self._cancel_btn = QPushButton("Cancel")
            self._cancel_btn.setMinimumWidth(160)
            self._cancel_btn.setStyleSheet(_ROLE_QSS["neutral"])
            self._cancel_btn.clicked.connect(self._on_cancel_clicked)
            button_row.addWidget(self._train_btn)
            button_row.addWidget(self._cancel_btn)
            content_layout.addLayout(button_row)

            outer.addWidget(content, alignment=Qt.AlignCenter)
            outer.addStretch(1)

            # Seed radio group from initial_state.
            mode = initial_state["refine_mode"]
            {
                "scratch": self._scratch_radio,
                "in_project": self._in_project_radio,
                "external": self._external_radio,
            }[mode].setChecked(True)

            # Pre-select the default iteration.
            default_iter = initial_state.get("source_iteration")
            if default_iter is not None:
                idx = self._iter_combo.findData(default_iter)
                if idx >= 0:
                    self._iter_combo.setCurrentIndex(idx)

            # Disable in_project radio if no trained iterations are
            # available; first-time training has to start from scratch
            # (or external on either DLC version since the helper
            # supports both).
            if not initial_state.get("trained_iterations"):
                self._in_project_radio.setEnabled(False)
                self._in_project_radio.setToolTip(
                    "No trained iterations available yet."
                )

            # Trigger enable/disable cascade once after seeding.
            self._on_radio_toggled()

            main_window.installEventFilter(self)

            self._frame.show()
            self._reposition()
            self._frame.raise_()
            self._train_btn.setFocus()

        # -- Helpers ---------------------------------------------------

        def _refresh_snapshot_combo(self):
            """Repopulate the snapshot combo for the currently-selected
            iteration. The first entry is "best (auto)" mapping to
            ``None`` so :meth:`initialize_weights` picks the best.
            """
            self._snap_combo.blockSignals(True)
            try:
                self._snap_combo.clear()
                self._snap_combo.addItem("best (auto)", userData=None)
                cur_iter = self._iter_combo.currentData()
                snapshots = self._state.get("snapshots_by_iteration", {}).get(
                    cur_iter, []
                )
                for snap in snapshots:
                    self._snap_combo.addItem(str(snap), userData=snap)
            finally:
                self._snap_combo.blockSignals(False)

        def _set_row_enabled(self, row, enabled):
            row.setEnabled(enabled)
            for child in row.findChildren(QLabel):
                child.setStyleSheet(
                    "color: white;" if enabled else "color: #888888;"
                )

        # -- Slots -----------------------------------------------------

        def _on_radio_toggled(self, *_):
            in_project = self._in_project_radio.isChecked()
            external = self._external_radio.isChecked()
            self._set_row_enabled(self._in_project_row, in_project)
            self._set_row_enabled(self._external_row, external)
            if in_project:
                self._refresh_snapshot_combo()

        def _on_iteration_changed(self, *_):
            self._refresh_snapshot_combo()

        def _on_browse_clicked(self):
            ext = ".pt" if self._state.get("is_dlc3", True) else ".index"
            file_filter = f"DLC snapshot (*{ext});;All files (*.*)"
            path, _selected_filter = QFileDialog.getOpenFileName(
                self._mw,
                "Choose external snapshot",
                "",
                file_filter,
            )
            if path:
                self._external_path_edit.setText(path)

        def _on_train_clicked(self):
            if self._scratch_radio.isChecked():
                mode = "scratch"
            elif self._in_project_radio.isChecked():
                mode = "in_project"
            else:
                mode = "external"
            self._result = {
                **self._state,
                "refine_mode": mode,
                "source_iteration": (
                    self._iter_combo.currentData() if mode == "in_project"
                    else None
                ),
                "source_snapshot": (
                    self._snap_combo.currentData() if mode == "in_project"
                    else None
                ),
                "external_snapshot_path": (
                    self._external_path_edit.text() if mode == "external"
                    else ""
                ),
                "maxiters": int(self._maxiters_spin.value()),
                "create_video": self._create_video_chk.isChecked(),
            }
            self._dismiss()
            self._loop.quit()

        def _on_cancel_clicked(self):
            self._result = None
            self._dismiss()
            self._loop.quit()

        # -- Lifecycle (mirror ConfirmOverlay) -------------------------

        def eventFilter(self, obj, event):  # noqa: N802 (Qt API)
            if obj is self._mw and event.type() == QEvent.Resize:
                self._reposition()
            return False

        def _reposition(self):
            self._frame.setGeometry(0, 0, self._mw.width(), self._mw.height())
            self._frame.raise_()

        def _dismiss(self):
            try:
                self._mw.removeEventFilter(self)
            except Exception:
                pass
            self._frame.hide()
            self._frame.deleteLater()

        def exec_(self):
            """Block until the user clicks Train or Cancel.

            Returns the result dict on Train; ``None`` on Cancel.
            """
            self._loop.exec_()
            return self._result

    return TrainingOptionsDialog


def _make_seed_bundle_picker_class():
    """Build :class:`SeedBundlePickerDialog` lazily, mirroring
    :func:`_make_training_options_class`'s qtpy-import-on-demand
    pattern. Shown when the user clicks Create DLC Project on an
    empty active manual layer and a seed-bundles root has been
    remembered (so we have a list of candidate bundles ready
    instead of forcing a Browse).

    Returns one of:
      - ``("use", info_dict)`` -- user picked a bundle from the list.
      - ``("browse",)`` -- user wants to Browse to a bundle elsewhere
        (caller falls through to ``QFileDialog``).
      - ``("set_root", new_root)`` -- user picked a new bundles root
        and the dialog should re-open against that root.
      - ``None`` -- user cancelled.
    """
    from qtpy.QtCore import QEvent, QEventLoop, QObject, Qt
    from qtpy.QtWidgets import (
        QFileDialog, QFrame, QHBoxLayout, QLabel, QListWidget,
        QListWidgetItem, QPushButton, QVBoxLayout, QWidget,
    )

    _ROLE_QSS = {
        "primary": (
            "QPushButton { background-color: #3a86ff; color: white; "
            "  border: 1px solid #2a76ef; padding: 6px 24px; "
            "  font-size: 11pt; font-weight: bold; }"
            "QPushButton:hover { background-color: #4a96ff; }"
            "QPushButton:pressed { background-color: #2a76ef; }"
        ),
        "neutral": (
            "QPushButton { background-color: #555555; color: white; "
            "  border: 1px solid #444444; padding: 6px 24px; "
            "  font-size: 11pt; }"
            "QPushButton:hover { background-color: #666666; }"
            "QPushButton:pressed { background-color: #444444; }"
        ),
    }

    class SeedBundlePickerDialog(QObject):
        def __init__(self, main_window, *, root, bundles):
            super().__init__(main_window)
            self._mw = main_window
            self._result = None
            self._loop = QEventLoop()
            self._bundles = bundles

            self._frame = QFrame(main_window)
            self._frame.setObjectName("dustrack_seed_picker_overlay")
            self._frame.setStyleSheet(
                "#dustrack_seed_picker_overlay { "
                "  background-color: rgba(0, 0, 0, 200); "
                "}"
                "QLabel { color: white; }"
                "#dustrack_seed_picker_title { color: white; "
                "  font-size: 22pt; font-weight: bold; }"
                "#dustrack_seed_picker_subtitle { color: #cccccc; "
                "  font-size: 10pt; }"
                "QListWidget { background-color: white; "
                "  font-size: 11pt; }"
            )
            self._frame.setFocusPolicy(Qt.StrongFocus)

            outer = QVBoxLayout(self._frame)
            outer.setAlignment(Qt.AlignCenter)
            outer.addStretch(1)

            title_lbl = QLabel("Choose seed bundle")
            title_lbl.setObjectName("dustrack_seed_picker_title")
            title_lbl.setAlignment(Qt.AlignCenter)
            outer.addWidget(title_lbl)

            subtitle = QLabel(f"From: {root}")
            subtitle.setObjectName("dustrack_seed_picker_subtitle")
            subtitle.setAlignment(Qt.AlignCenter)
            outer.addWidget(subtitle)

            content = QWidget()
            content.setMaximumWidth(720)
            content.setMinimumWidth(560)
            content_layout = QVBoxLayout(content)
            content_layout.setSpacing(8)

            self._list = QListWidget()
            self._list.setMinimumHeight(220)
            for b in bundles:
                desc = b.get("description") or "(no description)"
                bodyparts = b.get("bodyparts") or []
                label = (
                    f"{b['name']}\n"
                    f"  bodyparts: {bodyparts}\n"
                    f"  {desc}"
                )
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, b)
                self._list.addItem(item)
            self._list.setCurrentRow(0)
            self._list.itemDoubleClicked.connect(
                lambda _item: self._on_use_clicked()
            )
            content_layout.addWidget(self._list)

            # Buttons row 1: primary action + escape hatch.
            row1 = QHBoxLayout()
            row1.setAlignment(Qt.AlignCenter)
            self._use_btn = QPushButton("Use selected")
            self._use_btn.setMinimumWidth(160)
            self._use_btn.setStyleSheet(_ROLE_QSS["primary"])
            self._use_btn.clicked.connect(self._on_use_clicked)
            self._browse_btn = QPushButton("Browse elsewhere…")
            self._browse_btn.setMinimumWidth(160)
            self._browse_btn.setStyleSheet(_ROLE_QSS["neutral"])
            self._browse_btn.clicked.connect(self._on_browse_clicked)
            self._cancel_btn = QPushButton("Cancel")
            self._cancel_btn.setMinimumWidth(160)
            self._cancel_btn.setStyleSheet(_ROLE_QSS["neutral"])
            self._cancel_btn.clicked.connect(self._on_cancel_clicked)
            row1.addWidget(self._use_btn)
            row1.addWidget(self._browse_btn)
            row1.addWidget(self._cancel_btn)
            content_layout.addLayout(row1)

            # Row 2: change-root affordance, smaller / less prominent.
            row2 = QHBoxLayout()
            row2.setAlignment(Qt.AlignCenter)
            self._set_root_btn = QPushButton("Change bundles root…")
            self._set_root_btn.setMinimumWidth(220)
            self._set_root_btn.setStyleSheet(_ROLE_QSS["neutral"])
            self._set_root_btn.clicked.connect(self._on_set_root_clicked)
            row2.addWidget(self._set_root_btn)
            content_layout.addLayout(row2)

            outer.addWidget(content, alignment=Qt.AlignCenter)
            outer.addStretch(1)

            main_window.installEventFilter(self)

            self._frame.show()
            self._reposition()
            self._frame.raise_()
            self._use_btn.setFocus()

        # -- Slots ----------------------------------------------------

        def _on_use_clicked(self):
            item = self._list.currentItem()
            if item is None:
                return
            info = item.data(Qt.UserRole)
            self._result = ("use", info)
            self._dismiss()
            self._loop.quit()

        def _on_browse_clicked(self):
            self._result = ("browse",)
            self._dismiss()
            self._loop.quit()

        def _on_set_root_clicked(self):
            new_root = QFileDialog.getExistingDirectory(
                self._mw,
                "Choose seed bundles root folder",
                "",
                QFileDialog.ShowDirsOnly,
            )
            if not new_root:
                return  # user cancelled the folder picker, leave dialog open
            self._result = ("set_root", new_root)
            self._dismiss()
            self._loop.quit()

        def _on_cancel_clicked(self):
            self._result = None
            self._dismiss()
            self._loop.quit()

        # -- Lifecycle (mirror ConfirmOverlay) ------------------------

        def eventFilter(self, obj, event):  # noqa: N802 (Qt API)
            if obj is self._mw and event.type() == QEvent.Resize:
                self._reposition()
            return False

        def _reposition(self):
            self._frame.setGeometry(0, 0, self._mw.width(), self._mw.height())
            self._frame.raise_()

        def _dismiss(self):
            try:
                self._mw.removeEventFilter(self)
            except Exception:
                pass
            self._frame.hide()
            self._frame.deleteLater()

        def exec_(self):
            self._loop.exec_()
            return self._result

    return SeedBundlePickerDialog


class DUSTrack(_DUSTrackBase):
    """
    Interactive video point annotator with DeepLabCut integration.
    
    DUSTrack provides a GUI for manual annotation of points in videos, with integrated
    support for creating DeepLabCut projects, training models, and post-processing
    results using optical flow algorithms.
    
    Features:
        - Manual point annotation with event marking
        - Real-time trajectory visualization
        - DeepLabCut project creation and training
        - Optical flow-based jitter reduction
        - Multiple annotation layer management
    
    Attributes:
        _dlcproject (DLCProject): Associated DeepLabCut project instance.
        _ax_lims (dict): Stores axis limits when plot axes are frozen.
    
    Example:
        >>> # Most users should call :func:`dustrack.open` instead of
        >>> # constructing DUSTrack directly -- it dispatches to either
        >>> # this class (bare video) or :meth:`DLCProject.annotate`
        >>> # (resume in a DLC project) based on the path you pass.
        >>> # Direct construction below is for advanced / scripted use.
        >>>
        >>> # Basic usage -- annotation_names defaults to "iteration-0",
        >>> # the canonical seed name for the DLC pipeline (the next
        >>> # DLC training round lands as iteration-1).
        >>> tracker = DUSTrack('video.mp4')
        >>>
        >>> # Explicit layer name -- saved as {video_name}_annotations_pn.json
        >>> tracker = DUSTrack('video.mp4', "pn")
        >>>
        >>> # With multiple annotation layers
        >>> tracker = DUSTrack('video.mp4', {
        ...     'manual': 'manual_labels.json',
        ...     'dlc_iter1': 'dlc_predictions.h5'
        ... })
    """

    # Name of the spliced-corrections layer produced by
    # :meth:`apply_manual_corrections`. Parallels the magic ``"buffer"``
    # layer name :meth:`DLCProject.annotate` creates. The string matches
    # the ``_dlccorr`` substring filter in :func:`_extract_frames` so any
    # file written under this name is automatically excluded from DLC
    # training input -- this is a terminal output, not a refinement
    # source.
    CORRECTIONS_LAYER_NAME = "dlccorr"

    def __init__(self, vid_name, annotation_names="iteration-0", *args,
                 clahe_clip=1.0, clahe_grid=8, gamma=1.0, brightness=0,
                 dark_mode=False, **kwargs):
        # Store enhancement settings. Defaults now correspond to "no
        # enhancement" so the EnhanceWidget sliders start at their
        # leftmost position and DUSTrack opens with the raw frame;
        # the user dials enhancement in via the sliders. Replaced the
        # pre-2026-05-19 "Toggle enhance" button: with slider-driven
        # bypass at min, a separate toggle is redundant. ``brightness``
        # default dropped from ``+10`` to ``0`` so nudging either
        # slider one tick off zero doesn't visually jump a +10 offset
        # in alongside CLAHE/gamma -- the transition off the bypass
        # is now purely the user-driven slider values.
        self._clahe_clip = clahe_clip
        self._clahe_grid = clahe_grid
        self._gamma = gamma
        self._brightness = brightness
        self._dark_mode = dark_mode

        # Create image processor function with per-slider bypass to
        # smooth the transition at the left end of each slider:
        #
        # - Both at min -> raw image, no cvtColor, no CLAHE.
        # - clip at min, gamma > min -> gamma LUT applied per-channel
        #   on the RGB frame; no CLAHE, no cvtColor roundtrip. At
        #   gamma=1.0+epsilon the LUT is near-identity so the
        #   transition off bypass is visually continuous.
        # - clip > min -> full CLAHE pipeline (which inherently
        #   includes the RGB->gray->RGB roundtrip and the gamma
        #   branch inside enhance_ultrasound_image). The clip-slider
        #   transition off zero is still a step (CLAHE startup is
        #   not a fade-in operation) but only happens once now,
        #   instead of being triggered by either slider.
        def image_processor(im):
            skip_clahe = self._clahe_clip <= _CLAHE_CLIP_MIN
            skip_gamma = self._gamma <= _GAMMA_MIN
            if skip_clahe and skip_gamma:
                out = im
            elif skip_clahe:
                out = _apply_gamma_only(im, self._gamma)
            else:
                out = enhance_ultrasound_image(
                    im, self._clahe_clip, self._clahe_grid,
                    self._gamma, self._brightness
                )
            # dnav 1.5.0a2 auto-detects monochrome sources and returns
            # (H, W) gray frames. Coerce to 3-channel RGB here so the
            # downstream image pane (matplotlib or Qt-native) sees the
            # same shape regardless of source pix_fmt. The hot perf
            # paths (LK / postprocess / opticalflow) short-circuit on
            # 2D upstream of this, so this coerce only fires once per
            # interactive frame navigation -- negligible.
            if out.ndim == 2:
                out = cv.cvtColor(out, cv.COLOR_GRAY2RGB)
            return out

        kwargs['image_process_func'] = image_processor
        # DUSTrack defaults to datanavigator 1.5.0+ Tier 2 (Qt-native
        # video pane, ~3x speedup on real videos). Override with
        # ``DUSTrack(..., fast_render=False)`` only if a subclass needs
        # matplotlib Axes on the image region (no in-tree subclass
        # does today; this is forward-looking).
        kwargs.setdefault('fast_render', True)
        # Pin the QApplication palette before any widget is built so
        # appearance is reproducible across Qt bindings + OS themes.
        # See :func:`_pin_qt_palette` for the rationale.
        _pin_qt_palette(dark_mode)
        super().__init__(vid_name, annotation_names, *args, **kwargs)

        for ann in self.annotations:
            ann.__class__ = VideoAnnotation

        self._dlcproject = None
        self._ax_lims = {'state': False, 'x': [None, None], 'y_trace_x': [None, None], 'y_trace_y': [None, None]}

        # Apply dark theme if enabled
        if dark_mode:
            self._apply_dark_theme()

        # rc2 sidebar order: workflow | display | niche-op | utilities | swap.
        # Double separators mark major group boundaries; a *pair* of
        # double separators sets off Swap layers as a distinct trailing
        # action. dnav's _QtStatevarsWidget appends its own trailing
        # double separator below the statevars section, so we don't add
        # one after Swap layers.
        #
        # rc2 styling: per-group palette is plumbed through dnav's
        # ``Buttons.register_style`` / ``style_tag=`` API -- one styler
        # registered per palette key, then each ``add()`` / ``add_multi``
        # declares its tag inline. The pre-rc2 ``_btns_*`` collection
        # lists + ``_style_sidebar_buttons`` batch pass are gone; styling
        # auto-applies at add-time inside ``_finalize_button``. The
        # statevars widget palette still wants direct palette
        # manipulation (QSS on a parent flattens child QComboBoxes), so
        # that one piece lives in ``_paint_statevars_widget`` below.
        for _group, _spec in self._SIDEBAR_PALETTE.items():
            self.buttons.register_style(_group, _make_group_styler(_spec))

        # --- Workflow group: end-to-end annotation pipeline -------------
        if HAS_DLC:
            self.buttons.add(text="Create DLC Project", action_func=self.create_dlc_project, style_tag="workflow")
            self.buttons.add(text="Train DLC model", action_func=self.process_dlc_project, style_tag="workflow")
            self.buttons.add(text="Apply manual corrections", action_func=self.apply_manual_corrections, style_tag="workflow")
            self.buttons.add(text="Reduce jitter", action_func=self.process_with_lk, style_tag="workflow")
        self.buttons.add(text="Save annotation as...", action_func=self.save_annotation_as, style_tag="workflow")
        self.buttons.add_separator(style="double")

        # --- Display / trace controls -----------------------------------
        # Image enhancement is driven by the EnhanceWidget sliders
        # (mounted below statevars by _add_enhance_widget). Sliders at
        # min = bypass; no separate Toggle enhance button.
        self.buttons.add(text="Discard unsaved annotations", action_func=self.discard_unsaved_annotations, style_tag="display")
        self.buttons.add_multi(
            dict(text="Trace: line", action_func=(lambda s, ev: s.ann.set_plot_type("line")).__get__(self), style_tag="display"),
            dict(text="Trace: dot",  action_func=(lambda s, ev: s.ann.set_plot_type("dot")).__get__(self), style_tag="display"),
        )
        self.buttons.add_multi(
            dict(text="Freeze plot axes",   action_func=self.freeze_plot_axes, style_tag="display"),
            dict(text="Unfreeze plot axes", action_func=self.unfreeze_plot_axes, style_tag="display"),
        )
        self.buttons.add_separator(style="double")

        # Niche operation; flagged for a future decision -- should this
        # button be replaced by a keyboard-only shortcut to reclaim the
        # vertical slot? Track usage before removing.
        self.buttons.add(text="Replace existing from overlay", action_func=self.copy_existing_annotations_from_overlay, style_tag="niche")
        self.buttons.add(text="Remove layer", action_func=self.remove_current_layer, style_tag="niche")
        self.buttons.add_separator(style="double")

        # --- Utilities + Swap layers -----------------------------------
        # Refresh UI is normally installed by _DUSTrackBase's
        # _add_default_buttons hook; DUSTrack overrides that hook to
        # no-op (see _add_default_buttons below) so this slot keeps the
        # button next to Keyboard shortcuts as a "utility" pair just
        # above Swap layers.
        self.buttons.add(text="Refresh UI", action_func=self.refresh, style_tag="utilities")
        self.buttons.add(text="Keyboard shortcuts", action_func=(lambda s, ev: s.show_key_bindings()).__get__(self), style_tag="utilities")
        self.buttons.add_separator(style="double")

        self.buttons.add(text="Swap annotation layers", action_func=self.swap_active_and_overlay, style_tag="swap")

        # Statevars widget palette -- can't ride the Buttons styling
        # path because QSS on a QWidget parent flattens child
        # QComboBoxes (Windows-native dropdown rendering goes away).
        self._paint_statevars_widget()

        # Enhance two-slider widget: Qt-only; mpl fallback gets the
        # constructor defaults baked in via __init__ kwargs. Same
        # convention as the rest of DUSTrack's Qt-specific UI
        # (confirm modals, progress overlay, dock sidebar).
        self._enhance_widget = None
        self._add_enhance_widget()

        # rc2 safeguard: intercept window close so the user is asked
        # before losing in-memory annotation diffs. The Train pre-flight
        # already catches this path; this hook covers every other way
        # the window can close (X button, alt-F4, plt.close()).
        self._install_close_guard()

        self.statevariables._text._pos = dnav.utils._parse_pos("bottom left")

        # Workflow-button gating: re-evaluate enabled state whenever the
        # active or overlay dropdown changes. The single refresh helper
        # is also called once below (initial state) and at every other
        # site that mutates _dlcproject / fname / annotation state.
        # ``add_on_change`` is a dnav 1.4.0rc2+ surface; this class
        # already floors on >=1.4.0, so the attribute is guaranteed.
        for _sv_name in ("annotation_layer", "annotation_overlay"):
            if _sv_name in self.statevariables.names:
                self.statevariables[_sv_name].add_on_change(
                    self._refresh_workflow_button_state
                )
        self._refresh_workflow_button_state()

        if self.__class__.__name__ == "DUSTrack":
            plt.show(block=False)
            self.update()
            plt.setp(self._ax_trace_x.get_xticklabels(), visible=False)
            plt.draw()

    def _add_default_buttons(self) -> None:
        """Override the parent's default-button hook to a no-op.

        _DUSTrackBase installs ``Refresh UI`` immediately at
        end of ``__init__``; DUSTrack's rc2 sidebar instead places it
        next to ``Keyboard shortcuts`` as a utility pair just above
        ``Swap layers`` (see button-add block in ``__init__``).
        """
        return

    # rc2 sidebar palette (pastel analogous band; cool -> warm -> neutral).
    # Each group gets a coordinated bg/fg/border/hover/pressed triplet so
    # group transitions read at a glance without saturation-clash. Swap
    # layers + the statevars widget share the same pale silver `#e0e4e8`
    # -- visual pair at the bottom of the column. Single dark-slate
    # text color (`#2c3e50`) is the same for every group so the eye
    # isn't asked to retune contrast row-to-row.
    _SIDEBAR_STATEVARS_BG = "#e0e4e8"
    _SIDEBAR_TEXT_COLOR = "#2c3e50"
    _SIDEBAR_PALETTE = {
        "workflow": {  # powder blue -- primary pipeline, coolest end
            "bg": "#cfdef3", "fg": "#2c3e50",
            "border": "#a8c0dd", "hover": "#bccfea", "pressed": "#a8c0dd",
        },
        "display": {   # pale mint -- cool green, analogous step from blue
            "bg": "#d4ebd4", "fg": "#2c3e50",
            "border": "#aed4ae", "hover": "#c1dfc1", "pressed": "#aed4ae",
        },
        "niche": {     # pale apricot -- warm shift, "use sparingly"
            "bg": "#f5d9c0", "fg": "#2c3e50",
            "border": "#d9b88a", "hover": "#eaca9f", "pressed": "#d9b88a",
        },
        "utilities": {  # pale sand -- neutral warm
            "bg": "#ece6d5", "fg": "#2c3e50",
            "border": "#d4cdb8", "hover": "#e0d9c5", "pressed": "#d4cdb8",
        },
        "swap": {      # pale silver -- matches statevars
            "bg": "#e0e4e8", "fg": "#2c3e50",
            "border": "#c0c5cb", "hover": "#d0d4d9", "pressed": "#c0c5cb",
        },
    }

    def _add_enhance_widget(self) -> None:
        """Mount the two-slider :class:`EnhanceWidget` below the
        statevars widget in the rc2 left-column dock.

        Inserted into ``_dnav_left_column.outer_layout`` immediately
        after the statevars slot so the visual stack is
        ``buttons | statevars | enhance | stretch``. No-op on the
        mpl fallback path (no Qt main window / left column).
        """
        qt_window = self._find_qt_window()
        if qt_window is None:
            return
        col = getattr(qt_window, "_dnav_left_column", None)
        if col is None:
            return
        EnhanceWidget = _make_enhance_widget_class()
        widget = EnhanceWidget(self, parent=col.host)
        # statevars slot is at col.statevars_slot_index; insert right
        # after it (or right after the buttons widget if statevars
        # never got built).
        insert_at = (col.statevars_slot_index + 1) if col.statevars_widget is not None else col.statevars_slot_index
        col.outer_layout.insertWidget(insert_at, widget)
        self._enhance_widget = widget

    def _paint_statevars_widget(self) -> None:
        """Paint the statevars widget bg/fg to match the rc2 sidebar palette.

        Lives outside the ``Buttons.register_style`` machinery because
        the statevars widget needs *palette* manipulation, not QSS:
        QSS on a ``QWidget`` selector cascades into child
        ``QComboBox``\ es and replaces the native Windows dropdown
        paint with a flat CSS box, which is the opposite of what we
        want. ``QPalette.setColor`` respects native widget styling
        and only the parent bg/fg shifts. No-op when the Qt main
        window / left column can't be located (mpl-fallback path).
        """
        qt_window = self._find_qt_window()
        if qt_window is None:
            return
        col = getattr(qt_window, "_dnav_left_column", None)
        if col is None or col.statevars_widget is None:
            return
        from qtpy.QtGui import QColor
        sv = col.statevars_widget
        pal = sv.palette()
        pal.setColor(sv.backgroundRole(), QColor(self._SIDEBAR_STATEVARS_BG))
        pal.setColor(sv.foregroundRole(), QColor(self._SIDEBAR_TEXT_COLOR))
        sv.setPalette(pal)
        sv.setAutoFillBackground(True)
        # Clear any stylesheet that may have been set on a prior pass
        # so the palette change actually takes effect (QSS wins over
        # palette).
        sv.setStyleSheet("")

    def _refresh_workflow_button_state(self) -> None:
        """Enable / disable Workflow-group buttons based on session state.

        Three gates run here:

        - **Create DLC Project** is disabled when the session already
          sits inside a DLC project (either ``self._dlcproject`` is
          set, or the video path resolves into a project tree). DLC's
          ``copy_videos`` would otherwise scaffold a nested project at
          ``<project>/videos/`` whose downstream paths nothing handles.
        - **Train DLC model** is disabled when ``self._dlcproject`` is
          None. Replaces the click-time ``ValueError("DLCProject not
          created. ...")`` with a greyed-out button + tooltip.
        - **Apply manual corrections** is disabled when there's no
          overlay layer set, or when the active layer is already the
          corrections output. Mirrors the two
          :class:`ValueError` paths in :meth:`apply_manual_corrections`.

        ``Reduce jitter`` is intentionally *not* gated here: its real
        precondition is "every frame in the active layer is fully
        annotated", which is a data property rather than a name-pattern
        property. The cheap name-based proxy
        (:func:`_is_dense_layer_name`) is correct for rendering style
        but would false-disable a fully-annotated manual layer; the
        gate is deferred until a cheap data-side check exists. See
        :func:`lk_moving_average_filter`'s sparse-labels guard for the
        canonical run-time check.

        Qt-only: walks ``self.buttons`` and writes ``setEnabled`` +
        ``setToolTip`` on each Button's ``_qt_btn`` attribute. The
        helper no-ops on any button whose Qt handle is missing (the
        legacy mpl-fallback path; no longer supported as a first-class
        deployment, kept working for users on pinned older versions).
        """
        if not HAS_DLC:
            # The Workflow group's DLC buttons aren't added when
            # deeplabcut is missing; nothing to gate.
            return
        gates = self._evaluate_workflow_gates()
        for label, (enabled, tooltip) in gates.items():
            if label not in self.buttons:
                continue
            btn = self.buttons[label]
            qt_btn = getattr(btn, "_qt_btn", None)
            if qt_btn is None:
                continue
            qt_btn.setEnabled(enabled)
            qt_btn.setToolTip(tooltip)

    def _evaluate_workflow_gates(self) -> dict:
        """Compute ``{button_label: (enabled, tooltip)}`` for the gated buttons.

        Pulled out of :meth:`_refresh_workflow_button_state` so the
        gate semantics can be unit-tested without standing up a Qt
        window. Reads only ``self._dlcproject`` /
        ``self._current_overlay`` / ``self.ann`` / ``self.fname`` --
        the same state the click handlers themselves consult.
        """
        gates: dict = {}

        # --- Create DLC Project --------------------------------------
        proj_root = _session_inside_dlc_project(self)
        if proj_root is None:
            gates["Create DLC Project"] = (True, "")
        else:
            gates["Create DLC Project"] = (
                False,
                f"Already inside DLC project {proj_root.name!r} — "
                "use Train DLC model to extend it.",
            )

        # --- Train DLC model -----------------------------------------
        if self._dlcproject is None:
            gates["Train DLC model"] = (
                False,
                "Create a DLC project first.",
            )
        else:
            gates["Train DLC model"] = (True, "")

        # --- Apply manual corrections --------------------------------
        ann = getattr(self, "ann", None)
        ann_name = getattr(ann, "name", None) if ann is not None else None
        if self._current_overlay is None:
            gates["Apply manual corrections"] = (
                False,
                "Set an overlay layer (typically a 'dlc_*' trace) "
                "first.",
            )
        elif ann_name == self.CORRECTIONS_LAYER_NAME:
            gates["Apply manual corrections"] = (
                False,
                "Switch the active layer back to your manual "
                f"annotations — {self.CORRECTIONS_LAYER_NAME!r} is the "
                "output, not the input.",
            )
        else:
            gates["Apply manual corrections"] = (True, "")

        return gates

    def _apply_dark_theme(self):
        """Apply dark theme to the GUI for better ultrasound visibility."""
        bg_color = '#1a1a1a'
        ax_color = '#2a2a2a'
        text_color = 'white'

        # Figure background
        self.figure.patch.set_facecolor(bg_color)

        # Image axis (routes to either mpl set_facecolor (Tier 1) or
        # the Qt image pane background brush (Tier 2 / fast_render)).
        self.set_image_background_color(ax_color)

        # Trace axes
        for ax in [self._ax_trace_x, self._ax_trace_y]:
            ax.set_facecolor(ax_color)
            ax.tick_params(colors=text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            for spine in ax.spines.values():
                spine.set_color(text_color)

    def discard_unsaved_annotations(self, event=None):
        """Drop in-memory edits on the active layer and reload from disk.

        Inverse of ``save``: confirms via :class:`ConfirmOverlay`, then
        calls :meth:`VideoAnnotation.reload` on ``self.ann`` and
        triggers ``self.update()`` so the trace pane + scatter
        re-render. The confirm body branches on whether the layer's
        backing file exists:

        - File exists -> "Reload from disk" semantic.
        - File missing -> "Reset to empty" semantic.

        Refuses on the ``dlccorr`` splice and any layer matching
        :func:`_is_dense_layer_name` -- those are regenerated from
        the active + overlay layers (via ``apply_manual_corrections``
        or ``process_with_lk``), not authored by hand, so "discard
        and reload" has no meaningful semantic. Points the user at
        ``Remove layer`` instead.
        """
        qt_window = self._find_qt_window()
        if qt_window is None:
            # mpl fallback: no overlay; just reload silently. Same
            # convention as the rest of DUSTrack's Qt-specific UI
            # (confirm modals, progress overlay, dock sidebar).
            self.ann.reload()
            self.update()
            return

        ConfirmOverlay = _make_confirm_overlay_class()
        layer_name = self._current_layer

        if _is_dense_layer_name(layer_name):
            ConfirmOverlay(
                qt_window,
                title="Cannot discard derived layer",
                message=(
                    f"Layer {layer_name!r} is regenerated from other "
                    "layers (DLC prediction / manual-correction splice "
                    "/ jitter-reduced output) -- there is no "
                    "hand-authored state to discard.\n\n"
                    "Use Remove layer to drop it from the session "
                    "instead."
                ),
                buttons=[("OK", "neutral")],
                default="OK",
                severity="info",
            ).exec_()
            return

        fname = getattr(self.ann, "fname", None)
        file_exists = bool(fname) and Path(fname).exists()
        if file_exists:
            body = (
                f"Reload layer {layer_name!r} from disk?\n\n"
                f"File: {fname}\n\n"
                "In-memory edits since the last save will be lost."
            )
        else:
            body = (
                f"Reset layer {layer_name!r} to empty?\n\n"
                "No file exists on disk for this layer yet. "
                "All in-memory annotations will be lost."
            )
        result = ConfirmOverlay(
            qt_window,
            title="Discard unsaved annotations",
            message=body,
            buttons=[
                ("Discard", "destructive"),
                ("Cancel", "neutral"),
            ],
            default="Cancel",
            severity="destructive",
        ).exec_()
        if result == "Discard":
            self.ann.reload()
            self.update()

    def _increase_contrast(self, event=None):
        """Increase CLAHE contrast (clip limit)."""
        self._clahe_clip = min(self._clahe_clip + 0.5, 10.0)
        self.update()

    def _decrease_contrast(self, event=None):
        """Decrease CLAHE contrast (clip limit)."""
        self._clahe_clip = max(self._clahe_clip - 0.5, 1.0)
        self.update()

    def _increase_brightness(self, event=None):
        """Increase image brightness (gamma)."""
        self._gamma = min(self._gamma + 0.1, 3.0)
        self.update()

    def _decrease_brightness(self, event=None):
        """Decrease image brightness (gamma)."""
        self._gamma = max(self._gamma - 0.1, 0.3)
        self.update()

    def freeze_plot_axes(self, event=None):
        """
        Lock the axis limits of trajectory plots to current view.
        
        Useful for comparing tracking quality across different frames without
        automatic axis rescaling distracting from the comparison.
        
        Args:
            event: Mouse/keyboard event (unused, for button compatibility).
        """
        self._ax_lims['state'] = True
        self._ax_lims['x'] = self._ax_trace_x.get_xlim()
        self._ax_lims['y_trace_x'] = self._ax_trace_x.get_ylim()
        self._ax_lims['y_trace_y'] = self._ax_trace_y.get_ylim()
        self.update()

    def unfreeze_plot_axes(self, event=None):
        """
        Restore automatic axis scaling for trajectory plots.
        
        Args:
            event: Mouse/keyboard event (unused, for button compatibility).
        """
        self._ax_lims['state'] = False
        self._ax_lims['x'] = [None, None]
        self._ax_lims['y_trace_x'] = [None, None]
        self._ax_lims['y_trace_y'] = [None, None]
        self.update()


    def create_dlc_project(self, event=None, name=None, path=None,
                           experimenter=_config.EXPERIMENTER,
                           seed_bundle_path=None) -> DLCProject:
        """
        Create a new DeepLabCut project using current annotations as training labels.

        rc2 (1.1.0rc2): on a Qt backend, project creation runs off the
        GUI thread under a modal overlay (no progress bar -- it's a
        fast op, but the overlay surfaces DLC's stdout and a Done
        button so the user can confirm the project location and any
        warnings before continuing). On non-Qt backends the call runs
        synchronously and returns the new :class:`DLCProject`.

        1.2.0a2: if the active manual layer is empty when the user
        clicks Create DLC Project on the Qt path, a multi-step
        Seed-from-bundle modal opens
        (:meth:`_prompt_seed_bundle`). Picking a valid bundle routes
        through a seeding flow that: (a) creates the project as
        usual, (b) calls :func:`import_seed_bundle_into_project` to
        install the bundle's snapshot as iteration-0's trained model
        (overwriting the project's bodyparts with the bundle's), and
        (c) runs ``analyze_videos(iteration_num=0)`` to produce a
        dense reference layer the user can refine into iteration-1.
        ``seed_bundle_path`` may also be passed programmatically to
        bypass the modal.

        Args:
            event: Mouse/keyboard event (unused, for button compatibility).
            name (str, optional): Project name. Defaults to "{video_name}_{annotation_layer}".
            path (str, optional): Directory for project. Defaults to video's parent directory.
            experimenter (str, optional): Experimenter name. Defaults to config value.
            seed_bundle_path (str, optional): Path to a seed bundle
                folder (as produced by
                :func:`extract_snapshot_for_seeding`). When supplied,
                the seeding flow runs unconditionally (Qt or non-Qt),
                bypassing the empty-layer-triggered modal.

        Returns:
            DLCProject: The newly created project instance on the sync
            path. ``None`` on the Qt async path -- read
            ``self._dlcproject`` after the Done button is clicked.

        Note:
            Project names must contain an underscore for proper DLC configuration handling.
        """
        if not HAS_DLC:
            raise ImportError('deeplabcut is not installed. Cannot create DLC project.')

        qt_window = self._find_qt_window()
        active_layer_empty = not any(self.ann.data.values())

        # Qt path with empty active layer + no explicit bundle: open
        # the seeding modal sequence. The user can still cancel out
        # at every step (intent -> folder pick -> confirm). Cancel
        # leaves the UI intact (returns None).
        if (
            active_layer_empty
            and seed_bundle_path is None
            and qt_window is not None
        ):
            seed_bundle_path = self._prompt_seed_bundle(qt_window)
            if seed_bundle_path is None:
                return None  # user cancelled at some step

        # An empty active manual layer still needs an on-disk JSON
        # for DLCProject's constructor: ``copy_annotations`` reads
        # ``<video_stem>_annotations_<suffix>.json`` and the
        # constructor's all-same-labels assert iterates the resulting
        # files. The empty save writes ``{}`` (or ``{label: {}}``);
        # both pass through fine.
        self.ann.save()
        if name is None:
            name = f"{self.name}_{self.ann.name}"
        if path is None:
            path = str(Path(self.fname).parent)

        def _build_project():
            return DLCProject(
                path=path,
                videos=[self.fname],
                name=name,
                experimenter=experimenter,
                annotation_suffix=self.ann.name,
            )

        # Seeding path: run project creation + bundle import +
        # iteration-0 inference inside a single work_fn. The bundle's
        # bodyparts override the empty-derived bodyparts from the
        # constructor (see import_seed_bundle_into_project), and the
        # iteration-0 modelfolder is manufactured so DLC sees
        # iteration-0 as already trained.
        if seed_bundle_path is not None:

            def _build_seed_and_analyze():
                project = _build_project()
                print("Installing seed bundle...")
                import_seed_bundle_into_project(project, seed_bundle_path)
                print("Running analyze_videos on iteration-0 ...")
                project.analyze_videos(
                    iteration_num=0, create_video=False,
                )
                return project

            if qt_window is None:
                # Programmatic / non-Qt path: run synchronously.
                project = _build_seed_and_analyze()
                self._dlcproject = project
                self._rewire_to_in_project_paths()
                self._refresh_dlc_layers()
                self._refresh_workflow_button_state()
                return project

            def _on_seed_success(project: DLCProject):
                self._dlcproject = project
                self._rewire_to_in_project_paths()
                # Pulls predictions from videos/iteration-0/ into a
                # dense overlay layer and points the active layer at
                # an empty iteration-1 manual.
                self._refresh_dlc_layers()
                self._refresh_workflow_button_state()

            self._run_with_overlay(
                qt_window,
                work_fn=_build_seed_and_analyze,
                on_success=_on_seed_success,
                title="Creating + seeding DLC project",
                initial_phase="Scaffolding project",
                hint=(
                    "Output is also streamed to the launching terminal. "
                    "Predictions will load when you click Done."
                ),
                # analyze_videos emits a tqdm bar ("  3/100 [") that
                # the _PROGRESS_PATTERNS' tqdm-style regex consumes;
                # the same shape the Train DLC overlay uses for
                # consistency.
                show_progress_bar=True,
                phase_patterns=_SEED_PROJECT_PHASES,
                success_summary=(
                    f"Project '{name}' seeded from bundle. "
                    "Iteration-0 predictions loaded."
                ),
            )
            return None

        # Standard (non-seeded) path.
        if qt_window is None:
            self._dlcproject = _build_project()
            self._rewire_to_in_project_paths()
            self._refresh_workflow_button_state()
            return self._dlcproject

        def _on_success(project: DLCProject):
            self._dlcproject = project
            self._rewire_to_in_project_paths()
            # Create now disables (session sits inside the new project)
            # and Train enables (_dlcproject is set).
            self._refresh_workflow_button_state()

        self._run_with_overlay(
            qt_window,
            work_fn=_build_project,
            on_success=_on_success,
            title="Creating DLC project",
            initial_phase="Saving annotations and scaffolding project",
            hint="Output is also streamed to the launching terminal.",
            show_progress_bar=False,
            phase_patterns=_CREATE_PROJECT_PHASES,
            success_summary=f"Project '{name}' created.",
        )
        return None

    def _rewire_to_in_project_paths(self):
        """Repoint the live session's video + annotation paths to the
        in-project copies so subsequent writes (apply_manual_corrections,
        process_with_lk, save_annotation_as) land inside the project
        rather than next to the original video.

        Invoked once, on the GUI thread, immediately after
        :meth:`create_dlc_project` succeeds. DLC has already copied the
        video and the active layer's annotations into ``<project>/videos/``;
        this method takes care of the in-memory rewiring:

        - ``self.fname`` -> the in-project video copy. Downstream helpers
          like :func:`make_annotation_file_name` and
          ``Path(self.fname).parent`` then naturally write inside the
          project.
        - Each annotation layer whose ``.fname`` lives outside the
          project tree is migrated: ``.fname`` and ``.fstem`` are
          rewritten to the project's ``videos/`` folder; if the layer
          has in-memory data, it's saved at the new path so future
          ``save()`` calls have a destination on disk.
        - ``self.data`` (the video reader) is intentionally left
          pointing at the original file. The DLC-side ``copy_videos``
          guarantees byte-identical content, so frame seeking continues
          to work; rebuilding the reader mid-session would invalidate
          the Qt image pane handle.
        """
        assert self._dlcproject is not None
        in_project_video = str(self._dlcproject.video_list[0])
        videos_dir = Path(in_project_video).parent
        # Derive ``project_root`` from the in-project video path
        # (``videos_dir.parent``) rather than ``self._dlcproject.path``.
        # ``DLCProject.path`` is whatever the caller passed to the
        # constructor -- for a brand-new project that's the WORKING
        # DIRECTORY (parent of the actual project dir), because
        # ``deeplabcut.create_new_project`` creates the project at
        # ``<working_dir>/<name>-<experimenter>-<date>/``. The migration
        # check below ``ann_path.relative_to(project_root)`` then matches
        # a false-positive "yes, already inside" for any annotation file
        # sitting next to the original video, because the working dir IS
        # a prefix of that path -- silently skipping the migration and
        # stranding those layers at their original (outside-project)
        # locations. Affected the train pre-flight in particular: it
        # saved cleaned annotations to the wrong path, leaving the
        # project's copy stale and feeding the stale file into
        # ``extract_frames`` -> ``labeled_data``.
        project_root = videos_dir.parent

        self.fname = in_project_video

        for ann in self.annotations:
            if ann.fname is None:
                continue
            ann_path = Path(ann.fname)
            try:
                ann_path.relative_to(project_root)
                continue  # already inside project tree
            except ValueError:
                pass
            # DLC h5 traces only ever exist inside a project, so any
            # JSON outside the project (manual, dlccorr, buffer, an
            # iteration-N stub) is what we want to migrate.
            if ann_path.suffix.lower() != ".json":
                continue
            new_path = videos_dir / ann_path.name
            ann.fname = str(new_path)
            ann.fstem = new_path.stem
            if any(ann.data.values()):
                ann.save()

    def process_dlc_project(self, event=None, *args, **kwargs):
        """
        Train the DeepLabCut model without leaving the annotation UI.

        rc2 (1.1.0rc2): training runs off the GUI thread, the DUSTrack
        window stays open under a "Training in progress" overlay
        (progress bar + scrolling stdout tail), and on success the new
        DLC prediction layers are added to the live session via
        :meth:`add_annotation_layers` -- no relaunch. The overlay
        transitions to a "Complete" / "Failed" state with a Done
        button so the user can review the final stdout before the
        predictions swap in (or read the error before retrying);
        failure paths fold into the overlay rather than popping a
        separate :class:`QMessageBox`.

        1.2.0a2: a **Training options modal** runs before everything
        else (Qt path only). It surfaces
        :meth:`DLCProject.train_iteration`'s arg surface in the UI:
        refine_mode (scratch / in-project / external), source
        iteration/snapshot picker for in-project, Browse... for
        external ``.pt``, training epochs (DLC3) / iterations (DLC2),
        and a create-labeled-video toggle. Cancel returns to the UI
        without kicking off training. On accept, the underlying call
        routes through :meth:`DLCProject.train_iteration` (the
        explicit-args sibling of :meth:`DLCProject.process`) -- so the
        DLC2 silent-drop bug on ``refine=<str>`` is gone, and external
        snapshots work on both DLC versions (DLC3 via
        ``train_network(snapshot_path=...)``, DLC2 via a pose_cfg
        ``init_weights`` edit).

        A **unified pre-flight** runs before the overlay starts,
        scanning *every* manual annotation layer in the session
        (file-pattern detection -- ``.json`` alongside the video,
        ``<video_stem>_annotations*.json``, minus ``dlccorr`` /
        ``buffer`` / ``dlc*``; agnostic to which layer is active,
        the overlay, or a placeholder, and to whether the user
        renamed ``iteration-N`` to something else). For each manual
        layer we report (a) in-memory diffs vs the on-disk JSON
        and (b) frames missing one or more bodyparts. If any layer
        has either kind of issue, a single modal lists the
        per-layer breakdown and offers two actions: *Save and clean*
        (write per-layer recovery sidecars for the dropped frames,
        drop the incomplete frames from every affected layer, then
        save every affected layer, then train) and *Cancel*
        (return to the UI). Layers without issues are not touched;
        DLC trace layers and ``dlccorr`` are not in scope. Recovery
        sidecars are written as
        ``<fstem>.dustrack-dropped-incomplete-<YYYYMMDDTHHMMSS>``;
        the composite extension avoids ``.json`` so the
        annotation-discovery glob does not re-ingest them on
        subsequent training runs.

        DLC's stdout/stderr are also teed to the launching terminal
        (``sys.__stdout__``) so callers who launched from a shell can
        watch progress there as before.

        On non-Qt backends (or if the QMainWindow can't be located), the
        method falls back to the pre-rc2 behavior: close the figure,
        run training synchronously on the calling thread, and return a
        fresh DUSTrack via :meth:`DLCProject.annotate`. The pre-flight
        modal is skipped on this path (no GUI to host it).

        Args:
            event: Mouse/keyboard event (unused, for button compatibility).
            *args: Forwarded to :meth:`DLCProject.process` on the
                non-Qt fallback path; **ignored on the Qt path** (the
                Training options modal owns the kwarg surface there).
            **kwargs: Same -- non-Qt fallback only. ``create_video``
                defaults to ``False`` on the non-Qt path
                (vs. ``True`` for direct :meth:`DLCProject.process`
                calls) so the annotate -> train -> review -> annotate
                loop doesn't write a labeled mp4 each pass; the Qt
                path's create-labeled-video checkbox owns this on its
                own.

        Returns:
            DUSTrack: ``self`` on the Qt path (training is asynchronous;
            the same DUSTrack will refresh in place when the user
            clicks Done). Also ``self`` if the user cancels either
            the Training options modal or the pre-flight modal -- the
            UI is left intact. On the fallback path, the
            freshly-launched DUSTrack from :meth:`DLCProject.annotate`.

        Raises:
            ImportError: If ``deeplabcut`` isn't installed.
            ValueError: If no DLCProject has been created yet.
        """
        if not HAS_DLC:
            raise ImportError('deeplabcut is not installed. Cannot process DLC project.')
        if self._dlcproject is None:
            raise ValueError('DLCProject not created. Use create_dlc_project() to create it.')

        qt_window = self._find_qt_window()
        if qt_window is None:
            # Non-Qt fallback: no Training options modal possible, so
            # route through ``DLCProject.process()`` (auto-infer + sane
            # defaults). The Qt path uses ``train_iteration`` below with
            # explicit args supplied by the modal.
            kwargs.setdefault('create_video', False)
            plt.close(self.figure)
            self._dlcproject.process(*args, **kwargs)
            return self._dlcproject.annotate()

        # Qt path: empty-active-layer guard runs first so the user
        # can bail before configuring any training options. Two
        # sub-cases, distinguished by whether *any* labels exist in
        # the project (other manual layers OR previously-extracted
        # ``labeled-data/`` from prior iterations):
        #
        # - No labels anywhere (the freshly-seeded iteration-1 case):
        #   training would fail downstream because
        #   ``create_training_dataset`` has nothing to write. Hard-
        #   block with an error overlay; the user must annotate
        #   frames (or use Apply manual corrections to copy a DLC
        #   trace into a manual layer) before training.
        # - Labels exist elsewhere (mid-refinement: user clicked
        #   Train without adding new labels this pass): training is
        #   feasible by reusing the existing ``labeled-data/`` and/or
        #   running ``extract_frames`` on the other manual layer.
        #   Confirm intent with the existing modal.
        if not any(self.ann.data.values()):
            if not self._has_trainable_labels():
                self._prompt_no_trainable_labels(qt_window)
                return self  # hard block -- UI left intact
            if not self._prompt_empty_layer_train_confirm(qt_window):
                return self  # user cancelled -- UI left intact

        # Qt path: prompt for training options FIRST, then run the
        # existing pre-flight scan. The Training options modal owns the
        # full ``DLCProject.train_iteration`` kwarg surface (refine_mode
        # + in-project source picker OR external Browse + epochs +
        # create_video), so positional ``*args`` / ``**kwargs`` passed
        # to this method are ignored on the Qt path -- the user picks
        # per-click.
        training_kwargs = self._prompt_training_options(qt_window)
        if training_kwargs is None:
            return self  # user cancelled the Training options modal

        # Pre-flight: scan every manual annotation layer for
        # in-memory-vs-disk diffs AND/OR frames missing one or more
        # bodyparts. Single modal lets the user save & clean
        # everything in one click or return to the UI; layers
        # without issues are not touched. DLC traces / dlccorr /
        # buffer / non-iteration placeholders all fall out of scope
        # by file pattern, so a user who happens to have a DLC
        # trace active at click time doesn't trigger an h5 save
        # crash or a silent-overwrite of an unrelated layer.
        issues = self._scan_unsaved_and_incomplete()
        if issues:
            if not self._prompt_unified_pre_flight(qt_window, issues):
                return self  # user cancelled -- UI left intact
            self._apply_pre_flight_remediations(issues)
            # Cleaning can drop every frame of a layer (typical
            # post-seed case: user annotated only one label out of
            # the project bodyparts, every annotated frame is
            # incomplete, all frames get dropped). Re-evaluate the
            # trainable-labels predicate after the remediation; if
            # nothing is left to train on, hard-block here with the
            # same overlay :meth:`_has_trainable_labels` uses at the
            # click-time entry guard. Without this, the cleaning
            # leaves a project that DLC will fail on at
            # create_training_dataset time with a confusing message.
            if not self._has_trainable_labels():
                self._prompt_no_trainable_labels(qt_window)
                return self

        def _train():
            self._dlcproject.train_iteration(**training_kwargs)

        def _on_success(_unused):
            # Refresh layers BEFORE the user clicks Done so the new
            # predictions are already loaded when the overlay
            # dismisses. If the refresh raises, the helper folds the
            # exception into the Done overlay (mark_done success=False
            # with the refresh error -- training itself succeeded).
            self._refresh_dlc_layers()

        self._run_with_overlay(
            qt_window,
            work_fn=_train,
            on_success=_on_success,
            title="Training in progress",
            initial_phase="Starting up",
            hint=(
                "Output is also streamed to the launching terminal. "
                "Predictions will load when you click Done."
            ),
            show_progress_bar=True,
            phase_patterns=_TRAINING_PHASES,
            success_summary="Training complete. Predictions loaded.",
        )
        return self

    @staticmethod
    def _scan_incomplete_frames(
        data: dict, target_labels: list[str] | None = None,
    ) -> dict:
        """Find frames missing one or more required bodyparts in an
        annotation ``data`` dict (``{label: {frame: [x, y]}}``).

        Two modes:

        - ``target_labels=None`` (legacy, project-unaware): "required"
          = labels that have at least one annotation. Empty labels
          are treated as UI placeholders so they don't fail every
          frame. This is the right behavior for a session without a
          DLC project, where the user's declared label set may be the
          default ``[" 0"]`` bootstrap with no project bodyparts to
          anchor against.
        - ``target_labels=<list>`` (project-aware): "required" =
          exactly that list. Used when a ``DLCProject`` exists and
          its ``config['bodyparts']`` are the load-bearing label
          set -- training fails if a frame is missing any of them,
          regardless of whether the user has touched that label
          anywhere in the layer yet. Closes the case where a
          seeded project has bodyparts ``["point0", "point1"]``
          but the user annotated only the ``"0"`` label, and
          pre-flight wrongly reported the layer as complete.

        Returns ``{frame: [missing_label, ...]}`` for incomplete
        frames, frame-sorted with missing-labels lists in the same
        order as the required-label list. Empty dict iff every
        annotated frame has every required label.

        Pure data-in / data-out; testable from synthetic dicts.
        """
        if target_labels is None:
            required = [L for L, frames in data.items() if frames]
        else:
            required = list(target_labels)
        if not required:
            return {}
        all_frames: set = set()
        for L, frames in data.items():
            all_frames.update(frames.keys())
        incomplete: dict = {}
        for frame in sorted(all_frames):
            missing = [L for L in required if frame not in data.get(L, {})]
            if missing:
                incomplete[frame] = missing
        return incomplete

    @staticmethod
    def _build_dropped_incomplete_payload(data: dict, incomplete_frames: dict) -> dict:
        """Build the JSON payload for the dropped-incomplete sidecar.

        Each entry is ``{label: [x, y]}`` for the labels that *were*
        present at the dropped frame (the missing ones are the
        incompleteness). Frame keys are stringified for JSON compat.
        """
        payload: dict = {}
        for frame in incomplete_frames:
            present = {}
            for L, frames in data.items():
                if frame in frames:
                    present[L] = [float(x) for x in frames[frame]]
            payload[str(frame)] = present
        return payload

    @staticmethod
    def _build_dropped_incomplete_sidecar_name(ann_fname, ts: str) -> str:
        """``<fstem>.dustrack-dropped-incomplete-<ts>`` in the
        annotation's directory.

        Intentionally avoids `.json` so DUSTrack's annotation-discovery
        glob (``{video_stem}*_annotations*.json`` in
        :meth:`DLCProject.extract_frames`) does not re-ingest the
        sidecar on a subsequent training run.
        """
        p = Path(ann_fname)
        return str(p.parent / f"{p.stem}.dustrack-dropped-incomplete-{ts}")

    @staticmethod
    def _format_incomplete_breakdown(incomplete_frames: dict, max_rows: int = 200) -> str:
        """Multi-line per-bodypart breakdown for the pre-flight modal's
        detailed-text panel. Truncates very wide reports.
        """
        rows = []
        total = len(incomplete_frames)
        for i, (frame, missing) in enumerate(sorted(incomplete_frames.items())):
            if i >= max_rows:
                rows.append(f"... ({total - max_rows} more frames)")
                break
            rows.append(f"Frame {frame}: missing {', '.join(missing)}")
        return "\n".join(rows)

    def _save_dropped_incomplete_sidecar(self, ann, incomplete_frames: dict):
        """Persist the dropped-frame contents next to the given layer.

        Returns the sidecar path on success, ``None`` if the layer
        has no on-disk filename (in-memory only).
        """
        import datetime
        import json
        if ann.fname is None:
            return None
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        sidecar = self._build_dropped_incomplete_sidecar_name(ann.fname, ts)
        payload = self._build_dropped_incomplete_payload(ann.data, incomplete_frames)
        # Path.write_text rather than builtin open() because the
        # module-level open() (workflow entry point) shadows
        # builtins.open inside this file -- see _load_layer_disk_data.
        Path(sidecar).write_text(json.dumps(payload, indent=2))
        return sidecar

    @staticmethod
    def _is_manual_layer_name(
        ann_name: str,
        special_names: tuple = ("dlccorr", "buffer"),
    ) -> bool:
        """Name-only predicate for "is this a manual annotation layer?".

        Pure string check on the layer name -- excludes ``dlccorr``
        (terminal output of apply_manual_corrections), ``buffer``
        (workspace scratch), and any layer whose name starts with
        ``"dlc"`` (DLC trace + process_with_lk LK outputs). Symmetric
        with the name-side of :meth:`_is_manual_annotation_layer`,
        which adds an on-disk file-pattern check on top.

        Lives separately so callers that care about incomplete-frame
        scanning -- which only needs the in-memory ``ann.data`` --
        can include layers that aren't yet saved to disk (``ann.fname
        is None``). The Train pre-flight uses this for inclusion and
        then guards the disk-diff portion on ``ann.fname`` being set;
        save-on-close uses the stricter file-aware predicate because
        a layer with no disk file has nothing to diff against.
        """
        if ann_name in special_names:
            return False
        if ann_name.startswith("dlc"):
            return False
        return True

    @staticmethod
    def _is_manual_annotation_layer(
        video_fname,
        ann_fname,
        ann_name: str,
        special_names: tuple = ("dlccorr", "buffer"),
    ) -> bool:
        """Identify a manual annotation ``.json`` layer that feeds
        :meth:`DLCProject.extract_frames`.

        Rule: ``.json`` file alongside the video, matching the
        ``<video_stem>_annotations*.json`` pattern, AND the layer name
        passes :meth:`_is_manual_layer_name`. Excludes ``dlccorr`` /
        ``buffer`` / ``dlc*`` by name.

        File-based detection -- doesn't rely on the
        ``iteration-N`` naming convention, so a layer the user
        renamed to ``iter1`` or seeded with experimenter initials
        (e.g. ``pn``) is still picked up.
        """
        if ann_fname is None or video_fname is None:
            return False
        fname_path = Path(ann_fname)
        if fname_path.suffix != ".json":
            return False
        video_path = Path(video_fname)
        if fname_path.parent != video_path.parent:
            return False
        video_stem = video_path.stem
        stem = fname_path.stem
        if stem != f"{video_stem}_annotations" and not stem.startswith(
            f"{video_stem}_annotations_"
        ):
            return False
        return DUSTrack._is_manual_layer_name(ann_name, special_names)

    @staticmethod
    def _normalize_layer_data(data: dict) -> dict:
        """Canonical form for diff comparison: int frame keys, float
        ``[x, y]`` values, empty labels filtered.

        Empty-label filtering exists for *diff* symmetry: a label
        with no frames contributes no entries to added / removed /
        modified regardless of whether it's present on one side
        only. With dnav 1.4.0rc2's first-class-label schema, both
        on-disk JSON and in-memory data may legitimately carry
        ``"label": {}`` entries (whereas pre-rc2,
        :meth:`VideoAnnotation.save` pruned them on the way out); the
        diff still works correctly because both inputs are filtered
        the same way here.
        """
        out: dict = {}
        for label, frames in data.items():
            if not frames:
                continue
            out[label] = {
                int(frame): [float(x) for x in xy]
                for frame, xy in frames.items()
            }
        return out

    @staticmethod
    def _load_layer_disk_data(ann_fname) -> dict:
        """Read the on-disk JSON for a layer and return its data in
        canonical form. Empty dict if the file does not exist or
        cannot be parsed (treated as "fully unsaved").

        Uses ``Path.read_text`` rather than builtin ``open()`` because
        the module defines a top-level ``open`` (the workflow entry
        point) that shadows ``builtins.open`` inside this file -- same
        convention as ``DLCProject._read_trackermap``.
        """
        import json
        if ann_fname is None:
            return {}
        p = Path(ann_fname)
        if not p.exists():
            return {}
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return DUSTrack._normalize_layer_data(raw)

    @staticmethod
    def _diff_ann_vs_disk(mem_data: dict, disk_data: dict) -> dict:
        """Compare two canonical data dicts. Returns
        ``{"added": [...], "removed": [...], "modified": [...]}``
        where each list contains ``(label, frame)`` tuples in label-
        then frame-sorted order.

        Both inputs are assumed normalized via
        :meth:`_normalize_layer_data`.
        """
        added: list = []
        removed: list = []
        modified: list = []
        all_labels = set(mem_data) | set(disk_data)
        for label in sorted(all_labels):
            mem_frames = mem_data.get(label, {})
            disk_frames = disk_data.get(label, {})
            for frame in sorted(set(mem_frames) - set(disk_frames)):
                added.append((label, frame))
            for frame in sorted(set(disk_frames) - set(mem_frames)):
                removed.append((label, frame))
            for frame in sorted(set(mem_frames) & set(disk_frames)):
                if mem_frames[frame] != disk_frames[frame]:
                    modified.append((label, frame))
        return {"added": added, "removed": removed, "modified": modified}

    def _scan_unsaved_layers(self) -> dict:
        """Across every manual annotation layer, return
        ``{layer_name: diff}`` for layers whose in-memory data differs
        from disk. Sibling of :meth:`_scan_unsaved_and_incomplete`
        scoped to the data-loss concern only -- the close-event guard
        does not care about incomplete-frame quality (that surfaces
        next time the user trains).
        """
        unsaved: dict = {}
        for ann in self.annotations:
            if not self._is_manual_annotation_layer(
                self.fname, ann.fname, ann.name
            ):
                continue
            mem_data = self._normalize_layer_data(ann.data)
            disk_data = self._load_layer_disk_data(ann.fname)
            diff = self._diff_ann_vs_disk(mem_data, disk_data)
            if any(diff.values()):
                unsaved[ann.name] = diff
        return unsaved

    @staticmethod
    def _format_unsaved_summary(unsaved: dict) -> str:
        """Per-layer +added / -removed / ~modified counts for the
        save-on-close modal's informative text.
        """
        lines = []
        for layer_name, diff in unsaved.items():
            a = len(diff.get("added", []))
            r = len(diff.get("removed", []))
            m = len(diff.get("modified", []))
            pieces = []
            if a:
                pieces.append(f"+{a} added")
            if r:
                pieces.append(f"-{r} removed")
            if m:
                pieces.append(f"~{m} modified")
            lines.append(f"  {layer_name!r}: " + ", ".join(pieces))
        return "\n".join(lines)

    def _scan_unsaved_and_incomplete(self) -> dict:
        """Across every manual annotation layer in the session, find
        in-memory-vs-disk diffs AND/OR incomplete frames. Returns
        ``{layer_name: {"diff": ..., "incomplete": ...}}`` for
        layers with at least one issue; layers with neither are
        omitted.

        Inclusion is name-based (see :meth:`_is_manual_layer_name`)
        so layers that haven't been saved yet -- ``ann.fname is None``,
        the state after a user opens a fresh video, annotates a
        partial layer, and clicks Train without saving first -- are
        still scanned for incompleteness. The disk-diff portion is
        guarded on ``ann.fname`` being set AND matching the
        ``<video_stem>_annotations*.json`` pattern; without a disk
        file there's nothing to diff against. Pre-2026-05-21 the
        inclusion check required a file match too, which silently
        skipped unsaved-but-incomplete layers on first-time training.

        Project-aware incomplete scan (1.2.0a2): when ``self._dlcproject``
        is set, derives the required-label set from
        ``config['bodyparts']`` (mapped through
        :func:`_dlc_bodyparts_to_layer_labels`) and hands it to
        :meth:`_scan_incomplete_frames` as ``target_labels``. Catches
        the post-seeding case where the user has annotated, say, only
        the ``"0"`` label but the project bodyparts demand both
        ``"0"`` and ``"1"`` -- the legacy "active labels = labels
        with any annotation" rule wrongly reported those frames as
        complete.
        """
        target_labels: list[str] | None = None
        if self._dlcproject is not None:
            bodyparts = self._dlcproject.config.get("bodyparts") or []
            if bodyparts:
                target_labels = _dlc_bodyparts_to_layer_labels(bodyparts)

        issues: dict = {}
        for ann in self.annotations:
            if not self._is_manual_layer_name(ann.name):
                continue
            incomplete = self._scan_incomplete_frames(
                ann.data, target_labels=target_labels,
            )
            diff = {"added": [], "removed": [], "modified": []}
            if self._is_manual_annotation_layer(
                self.fname, ann.fname, ann.name
            ):
                mem_data = self._normalize_layer_data(ann.data)
                disk_data = self._load_layer_disk_data(ann.fname)
                diff = self._diff_ann_vs_disk(mem_data, disk_data)
            if any(diff.values()) or incomplete:
                issues[ann.name] = {"diff": diff, "incomplete": incomplete}
        return issues

    @staticmethod
    def _format_pre_flight_summary(
        issues: dict, max_incomplete_examples: int = 3
    ) -> str:
        """Per-layer breakdown for the unified pre-flight modal's
        detailed-text panel.
        """
        blocks = []
        for layer_name, info in issues.items():
            lines = [f"Layer {layer_name!r}:"]
            diff = info.get("diff", {})
            a = len(diff.get("added", []))
            r = len(diff.get("removed", []))
            m = len(diff.get("modified", []))
            if a or r or m:
                pieces = []
                if a:
                    pieces.append(f"+{a} added")
                if r:
                    pieces.append(f"-{r} removed")
                if m:
                    pieces.append(f"~{m} modified")
                lines.append("  Unsaved changes: " + ", ".join(pieces))
            else:
                lines.append("  (no unsaved changes)")
            incomplete = info.get("incomplete", {})
            if incomplete:
                n = len(incomplete)
                lines.append(f"  Incomplete frames: {n}")
                for i, (frame, missing) in enumerate(sorted(incomplete.items())):
                    if i >= max_incomplete_examples:
                        lines.append(
                            f"    ... ({n - max_incomplete_examples} more)"
                        )
                        break
                    lines.append(
                        f"    frame {frame}: missing {', '.join(missing)}"
                    )
            else:
                lines.append("  (no incomplete frames)")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def _prompt_training_options(self, qt_window):
        """Show the Training options modal and return kwargs ready to
        splat into :meth:`DLCProject.train_iteration`.

        Builds the initial state via
        :func:`_default_training_options` from the live ``DLCProject``,
        runs :class:`TrainingOptionsDialog` synchronously, and
        translates the user's choices via
        :func:`_training_options_to_train_iteration_kwargs`.

        Returns:
            dict | None: kwargs for ``train_iteration``, or ``None``
            if the user clicked Cancel (caller returns without
            kicking off training).
        """
        TrainingOptionsDialog = _make_training_options_class()
        initial_state = _default_training_options(self._dlcproject)
        options = TrainingOptionsDialog(
            qt_window, initial_state=initial_state,
        ).exec_()
        if options is None:
            return None
        return _training_options_to_train_iteration_kwargs(options)

    def _prompt_seed_bundle(self, qt_window) -> Optional[str]:
        """Multi-step modal sequence that fires when ``Create DLC Project``
        is clicked with an empty active manual layer.

        Two entry points depending on whether a seed-bundles root has
        been remembered:

        - **Root set + non-empty**: opens a list-picker
          (:class:`SeedBundlePickerDialog`) showing every valid bundle
          under the root with its name + bodyparts + description.
          Quick-select; no file-dialog navigation required.
        - **No root set / empty root**: opens the legacy
          :class:`ConfirmOverlay` -> ``QFileDialog`` -> confirm path.
          After a successful pick, offers to remember the picked
          bundle's parent as the bundles root so next session uses
          the picker.

        Returns the validated bundle folder path on Accept, or ``None``
        on any Cancel / invalid bundle path. Caller (``create_dlc_project``)
        treats ``None`` as "user bailed -- leave the UI alone".
        """
        # Loop so the picker's "Change bundles root" action can
        # re-open the dialog against the new root, and "Browse
        # elsewhere" can fall through to the file-dialog branch.
        while True:
            root = get_seed_bundles_root()
            if root is not None and root.is_dir():
                bundles = list_seed_bundles(root)
            else:
                bundles = []

            if bundles:
                action = self._pick_from_seed_bundles(qt_window, root, bundles)
                if action is None:
                    return None
                kind = action[0]
                if kind == "use":
                    info = action[1]
                    bundle_path = str(info["path"])
                    if self._confirm_seed_bundle(qt_window, bundle_path, info):
                        return bundle_path
                    return None
                if kind == "set_root":
                    set_seed_bundles_root(action[1])
                    continue  # re-list against the new root
                if kind == "browse":
                    # Fall through to legacy Browse flow below, but
                    # don't loop -- a Browse pick should either
                    # accept and return, or cancel out.
                    pass

            # Legacy flow: explain + Browse + validate + confirm.
            picked = self._browse_for_seed_bundle(qt_window)
            if picked is None:
                return None
            # First-time-Browse polite ask: remember the parent as
            # the root so the picker takes over next session. Skip
            # if a root is already configured (user already declined
            # or has a different setup in mind).
            if get_seed_bundles_root() is None:
                self._maybe_remember_seed_bundles_root(qt_window, picked)
            return picked

    def _pick_from_seed_bundles(self, qt_window, root, bundles):
        """Drive :class:`SeedBundlePickerDialog`. Returns the
        dialog's raw result tuple (or ``None`` on cancel)."""
        PickerDialog = _make_seed_bundle_picker_class()
        return PickerDialog(qt_window, root=root, bundles=bundles).exec_()

    def _browse_for_seed_bundle(self, qt_window) -> Optional[str]:
        """Legacy seed-bundle flow used when no bundles root is set
        (or the picker user clicked Browse elsewhere): intent
        overlay -> ``QFileDialog`` -> validate -> confirm. Returns
        the validated bundle path on accept, ``None`` on any cancel
        or invalid bundle."""
        from qtpy.QtWidgets import QFileDialog

        ConfirmOverlay = _make_confirm_overlay_class()

        # Only show the intent overlay when there's no remembered
        # root -- if the user got here via "Browse elsewhere" from
        # the picker, they already understand the situation.
        if get_seed_bundles_root() is None:
            result = ConfirmOverlay(
                qt_window,
                title="No annotations in active layer",
                message=(
                    f"Active layer {self.ann.name!r} has no labels. "
                    "To create a DLC project from this session, "
                    "either annotate frames manually first, or seed "
                    "iteration-0 from a pre-trained snapshot bundle "
                    "(a folder containing snapshot-*.pt + "
                    "pytorch_config.yaml + pose_cfg.yaml).\n\n"
                    "Inference from the bundled snapshot will run on "
                    "the current video and load as a dense reference "
                    "overlay; your manual refinements then become "
                    "iteration-1."
                ),
                buttons=[
                    ("Browse for seed bundle…", "primary"),
                    ("Cancel", "neutral"),
                ],
                default="Cancel",
                severity="warning",
            ).exec_()
            if result != "Browse for seed bundle…":
                return None

        bundle_dir = QFileDialog.getExistingDirectory(
            qt_window,
            "Choose seed bundle folder",
            "",
            QFileDialog.ShowDirsOnly,
        )
        if not bundle_dir:
            return None

        try:
            info = inspect_seed_bundle(bundle_dir)
        except (FileNotFoundError, ValueError) as exc:
            ConfirmOverlay(
                qt_window,
                title="Invalid seed bundle",
                message=(
                    f"The selected folder is not a usable seed bundle:\n\n"
                    f"{exc}\n\n"
                    "Re-click 'Create DLC Project' to try again."
                ),
                buttons=[("OK", "neutral")],
                default="OK",
                severity="error",
            ).exec_()
            return None

        if self._confirm_seed_bundle(qt_window, bundle_dir, info):
            return bundle_dir
        return None

    def _confirm_seed_bundle(self, qt_window, bundle_path, info) -> bool:
        """Final confirm-with-detected-info overlay shared by the
        picker and Browse paths. Returns True iff the user clicked
        ``Create and seed``.
        """
        ConfirmOverlay = _make_confirm_overlay_class()
        description = info.get("description") or "(no description)"
        result = ConfirmOverlay(
            qt_window,
            title="Confirm seed bundle",
            message=(
                f"Bundle: {bundle_path}\n"
                f"Snapshot: {info['snapshot'].name}\n"
                f"Bodyparts ({len(info['bodyparts'])}): {info['bodyparts']}\n"
                f"Net type: {info.get('net_type') or '(unset)'}\n"
                f"Description: {description}\n\n"
                "Create the project, install this snapshot as iteration-0, "
                "and run inference on the current video?"
            ),
            buttons=[
                ("Create and seed", "primary"),
                ("Cancel", "neutral"),
            ],
            default="Cancel",
            severity="info",
        ).exec_()
        return result == "Create and seed"

    def _maybe_remember_seed_bundles_root(self, qt_window, bundle_path) -> None:
        """After the first successful Browse pick, ask the user if
        they want to remember the bundle's parent folder as the
        seed-bundles root so the next session opens the list-picker
        directly. No-op if they say no (the next Browse will ask
        again)."""
        ConfirmOverlay = _make_confirm_overlay_class()
        parent = Path(bundle_path).parent
        result = ConfirmOverlay(
            qt_window,
            title="Remember bundles location?",
            message=(
                f"Use this folder as your seed-bundles root?\n\n"
                f"{parent}\n\n"
                "Next time you click Create DLC Project on an empty "
                "layer, DUSTrack will list every bundle in this "
                "folder so you don't have to browse."
            ),
            buttons=[
                ("Remember it", "primary"),
                ("Not now", "neutral"),
            ],
            default="Remember it",
            severity="info",
        ).exec_()
        if result == "Remember it":
            set_seed_bundles_root(parent)

    def _has_trainable_labels(self) -> bool:
        """True if the project has *any* source of labels training
        could consume: at least one non-empty manual annotation layer
        in the session, or at least one ``.h5`` under the project's
        ``labeled-data/`` folder (already-extracted labels from
        prior iterations).

        Pure predicate -- no side effects. Used by
        :meth:`process_dlc_project` to decide between hard-blocking
        the Train DLC click and falling through to the
        "Continue training without new data?" confirm.
        """
        for ann in self.annotations:
            if not self._is_manual_layer_name(ann.name):
                continue
            if any(ann.data.values()):
                return True
        if self._dlcproject is not None:
            labels_dir = Path(self._dlcproject.paths["labels"])
            if labels_dir.is_dir():
                for _ in labels_dir.rglob("*.h5"):
                    return True
        return False

    def _prompt_no_trainable_labels(self, qt_window) -> None:
        """Hard-block overlay for the Train DLC path when the active
        manual layer is empty AND no other source of labels exists in
        the project (no other non-empty manual layer, no
        ``labeled-data/*.h5``). Distinct from
        :meth:`_prompt_empty_layer_train_confirm` -- there's nothing
        to confirm, the user has to add labels before training can do
        anything.

        Typical trigger: freshly-seeded project, user clicks Train
        before annotating any iteration-1 frames.
        """
        ConfirmOverlay = _make_confirm_overlay_class()
        ConfirmOverlay(
            qt_window,
            title="No labels to train on",
            message=(
                f"Active layer {self.ann.name!r} has no labels, and no "
                "other annotation layer or 'labeled-data/' file in this "
                "project has any either. Training would have nothing "
                "to consume.\n\n"
                "Annotate some frames in the active layer first, or "
                "use 'Apply manual corrections' to convert a DLC "
                "prediction trace into a manual annotation layer."
            ),
            buttons=[("OK", "neutral")],
            default="OK",
            severity="error",
        ).exec_()

    def _prompt_empty_layer_train_confirm(self, qt_window) -> bool:
        """Modal that fires when Train DLC is clicked with an empty
        active manual layer. Returns True iff the user confirmed
        ``Continue training``.

        "Empty active layer" means no label has any frames -- the
        check at the call site is ``not any(self.ann.data.values())``,
        the same predicate used by ``_rewire_to_in_project_paths``
        to decide whether a layer needs an on-disk save. The empty
        layer itself is *not* saved by this path -- the pre-flight
        scan downstream only acts on layers with diffs or
        incompleteness, and an empty layer has neither.

        User intent in this state: "train for more iterations without
        new labels." Training will reuse whatever labels already
        exist in ``labeled-data/``; if none do (e.g. a freshly-seeded
        iteration-1), DLC will fail downstream with its own error.
        """
        ConfirmOverlay = _make_confirm_overlay_class()
        body = (
            f"Active layer {self.ann.name!r} has no labels.\n\n"
            "Training will reuse the labels already in "
            "'labeled-data/' from previous iterations. The empty "
            "active layer will not be saved.\n\n"
            "Continue without adding new labels?"
        )
        result = ConfirmOverlay(
            qt_window,
            title="No annotations in active layer",
            message=body,
            buttons=[
                ("Continue training", "primary"),
                ("Cancel", "neutral"),
            ],
            default="Cancel",
            severity="warning",
        ).exec_()
        return result == "Continue training"

    def _prompt_unified_pre_flight(self, qt_window, issues: dict) -> bool:
        """Single modal for the combined save-state + incompleteness
        pre-flight. Returns True iff the user picked
        *Save and clean*.

        Routes through :class:`ConfirmOverlay` (rc2) so the modal
        shares visual vocabulary with the new ``Discard unsaved`` /
        ``Remove layer`` confirms; pre-rc2 used ``QMessageBox``. The
        per-layer breakdown is shown inline rather than behind a
        collapsed "Show Details..." toggle -- the breakdown is the
        substance the user needs to decide on, not optional extra.
        """
        ConfirmOverlay = _make_confirm_overlay_class()
        n = len(issues)
        header = (
            f"{n} manual annotation layer{'s' if n != 1 else ''} "
            f"{'have' if n != 1 else 'has'} unsaved changes and/or "
            "incomplete frames."
        )
        breakdown = self._format_pre_flight_summary(issues)
        body = (
            f"{header}\n\n"
            f"{breakdown}\n\n"
            "Save and clean will:\n"
            " - save in-memory edits to disk for the listed layer(s),\n"
            " - drop frames missing one or more bodyparts (per-layer "
            "recovery sidecars written next to each annotation file),\n"
            " - then start training.\n\n"
            "Cancel returns to the UI without changes."
        )
        result = ConfirmOverlay(
            qt_window,
            title="Pre-flight issues",
            message=body,
            buttons=[
                ("Save and clean", "primary"),
                ("Cancel", "neutral"),
            ],
            default="Cancel",
            severity="warning",
        ).exec_()
        return result == "Save and clean"

    def _apply_pre_flight_remediations(self, issues: dict) -> None:
        """For each layer with issues, drop incomplete frames (with
        recovery sidecar) and save the (possibly trimmed) layer.

        Layers whose ``ann.fname`` is ``None`` (in-session unsaved
        layers, the first-time-training case) get a canonical fname
        derived from the video stem + layer name before save:
        ``<video_stem>_annotations_<layer_name>.json``. The recovery
        sidecar needs the same path resolved.
        """
        for layer_name, info in issues.items():
            ann = self.annotations[layer_name]
            if ann.fname is None:
                ann.fname = str(make_annotation_file_name(
                    Path(self.fname), annotation_suffix=ann.name
                ))
            incomplete = info.get("incomplete") or {}
            if incomplete:
                self._save_dropped_incomplete_sidecar(ann, incomplete)
                # Drop the incomplete frames directly using the scan's
                # output. The pre-fix pair (remove_empty_labels +
                # keep_overlapping_frames) silently failed in the
                # project-aware case: a required-but-empty label (the
                # user touched only "0" in a ["point0", "point1"]
                # project) got dropped by remove_empty_labels, and
                # keep_overlapping_frames then preserved every
                # incomplete frame because the now-single-label
                # schema trivially "overlapped" with itself. Routing
                # mutations through ``ann.remove(label, frame)`` keeps
                # the revision counter consistent (see
                # ``feedback_revision_counter_invalidation_pattern``)
                # and works in both project-aware and legacy modes.
                for frame in incomplete:
                    for label in list(ann.data.keys()):
                        if frame in ann.data[label]:
                            ann.remove(label, frame)
            ann.save()
        self.update()

    def _prompt_save_on_close(self, qt_window, unsaved: dict) -> str:
        """Modal triggered by the save-on-close guard. Returns the user's
        choice as one of ``"save"`` / ``"discard"`` / ``"cancel"``.

        *Save* writes every layer with diffs and lets the window close;
        *Discard* lets the window close without writing; *Cancel* keeps
        the window open. ``Cancel`` is the default button so that
        accidental Enter / Esc do not silently lose data. Routes
        through :class:`ConfirmOverlay` (rc2); pre-rc2 used
        ``QMessageBox``.
        """
        ConfirmOverlay = _make_confirm_overlay_class()
        n = len(unsaved)
        header = (
            f"{n} annotation layer{'s' if n != 1 else ''} "
            f"{'have' if n != 1 else 'has'} unsaved changes."
        )
        breakdown = self._format_unsaved_summary(unsaved)
        body = (
            f"{header}\n\n"
            f"{breakdown}\n\n"
            "Save all writes the in-memory edits to disk before closing.\n"
            "Discard closes without writing -- changes are lost.\n"
            "Cancel keeps the window open."
        )
        result = ConfirmOverlay(
            qt_window,
            title="Unsaved annotations",
            message=body,
            buttons=[
                ("Save all", "primary"),
                ("Discard", "destructive"),
                ("Cancel", "neutral"),
            ],
            default="Cancel",
            severity="destructive",
        ).exec_()
        if result == "Save all":
            return "save"
        if result == "Discard":
            return "discard"
        return "cancel"

    def _save_unsaved_layers(self, unsaved: dict) -> None:
        """Persist every layer with diffs. Called from the save-on-close
        guard when the user picks *Save all*.
        """
        for layer_name in unsaved:
            ann = self.annotations[layer_name]
            ann.save()

    def _install_close_guard(self) -> None:
        """Patch the QMainWindow ``closeEvent`` so window close triggers
        the unsaved-diffs scan + modal.

        Monkey-patch rather than subclass because the QMainWindow is
        constructed inside matplotlib's Qt backend; intercepting it
        without owning the type means patching the instance. The
        original ``closeEvent`` is chained at the end so any
        backend-internal cleanup still runs.

        No-op on the mpl fallback path (no Qt window to hook).
        """
        qt_window = self._find_qt_window()
        if qt_window is None:
            return
        if getattr(qt_window, "_dustrack_close_guard_installed", False):
            return  # idempotent: a second __init__ pass (e.g. subclass) must not stack

        original_close_event = qt_window.closeEvent
        dustrack_self = self

        def closeEvent(event):
            try:
                unsaved = dustrack_self._scan_unsaved_layers()
            except Exception:
                # If the scan itself fails (e.g. annotation list mutated
                # mid-shutdown), don't block the close -- the guard is a
                # safety net, not a hard gate.
                unsaved = {}
            if unsaved:
                choice = dustrack_self._prompt_save_on_close(qt_window, unsaved)
                if choice == "cancel":
                    event.ignore()
                    return
                if choice == "save":
                    dustrack_self._save_unsaved_layers(unsaved)
                # "discard" falls through to the original closeEvent
            original_close_event(event)

        qt_window.closeEvent = closeEvent
        qt_window._dustrack_close_guard_installed = True

    def _find_qt_window(self):
        """Return the QMainWindow hosting ``self.figure``, or ``None``
        if we can't locate it (non-Qt backend, headless run, etc.).
        Wraps :func:`datanavigator._qt.find_qt_window` with the
        try/except shape that callers want.
        """
        try:
            from datanavigator._qt import find_qt_window
            return find_qt_window(self.figure)
        except Exception:
            return None

    def _run_with_overlay(
        self,
        qt_window,
        *,
        work_fn,
        on_success=None,
        title: str = "Working",
        initial_phase: str = "Starting up",
        hint: str = "",
        show_progress_bar: bool = True,
        phase_patterns=None,
        success_summary: str = "Done.",
    ):
        """Drive ``work_fn`` on a worker thread under a progress
        overlay, tee its stdout/stderr to both the launching terminal
        and the overlay log, and on completion transition the overlay
        to a "Done" state. The user clicks Done to dismiss the overlay
        and (on success) fire ``on_success(result)`` on the GUI
        thread; ``result`` is the return value of ``work_fn``.

        Failure paths fold into the overlay (title flips to "Failed",
        progress bar reads "Failed", phase becomes the exception
        message); the Done button still dismisses cleanly. If
        ``on_success`` itself raises after a successful ``work_fn``,
        the overlay shows that as a separate failure mode -- the work
        succeeded, but the follow-up didn't.

        Pure GUI plumbing: the heavy thread/queue/QTimer machinery is
        all in here so the button handlers stay small.
        """
        from qtpy.QtCore import QTimer

        ProgressOverlay = _make_progress_overlay_class()
        overlay = ProgressOverlay(
            qt_window,
            title=title,
            initial_phase=initial_phase,
            hint=hint,
            show_progress_bar=show_progress_bar,
        )

        log_queue: "queue.Queue[str]" = queue.Queue()
        result_state: dict = {"exc": None, "tb": "", "done": False, "value": None}
        phase_patterns = phase_patterns or []

        def _worker():
            sink = _Tee(sys.__stdout__, _QueueWriter(log_queue))
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = sink
            sys.stderr = sink
            try:
                result_state["value"] = work_fn()
            except BaseException as e:  # noqa: BLE001 (re-raised on GUI thread)
                result_state["exc"] = e
                # Capture the traceback now (on the worker thread) so
                # the GUI thread doesn't have to walk a stale stack.
                result_state["tb"] = traceback.format_exc()
                # Push the traceback through the same teed sink so it
                # lands in the overlay log + the launching terminal
                # without us having to wire a second channel.
                try:
                    sys.stdout.write(result_state["tb"])
                    sys.stdout.flush()
                except Exception:
                    pass
            finally:
                sys.stdout, sys.stderr = old_out, old_err
                result_state["done"] = True

        thread = threading.Thread(
            target=_worker, name="dustrack-overlay-worker", daemon=True,
        )

        timer = QTimer(qt_window)
        timer.setInterval(200)

        # Per-tick state lives in a dict so the inner closure can mutate
        # without nonlocal gymnastics. ``line_buf`` is the unterminated
        # tail (no \n yet); ``last_cr_match`` lets us short-circuit
        # repeated tqdm redraws.
        tick_state = {"line_buf": ""}

        def _scan_segment(seg: str):
            """Feed a single fragment (may be \\r-terminated tqdm
            redraw or a full line) through phase + progress matchers.
            Does not append to the visible log.
            """
            if not seg.strip():
                return
            for pat, label in phase_patterns:
                if pat.search(seg):
                    overlay.set_phase(label)
                    break
            for pat in _PROGRESS_PATTERNS:
                m = pat.search(seg)
                if m:
                    cur, tot = int(m.group(1)), int(m.group(2))
                    if tot > 0 and cur <= tot:
                        overlay.set_progress(cur, tot)
                    break

        def _drain_and_update():
            chunks = []
            try:
                while True:
                    chunks.append(log_queue.get_nowait())
            except queue.Empty:
                pass
            if chunks:
                tick_state["line_buf"] += "".join(chunks)
                # Split into \n-terminated lines + unterminated tail.
                *full_lines, tick_state["line_buf"] = tick_state["line_buf"].split("\n")
                for line in full_lines:
                    # tqdm redraws use \r to overwrite the previous
                    # frame within a single logical line. Run each
                    # \r-segment through the matchers (so the
                    # progress bar updates), but only show the final
                    # segment in the log (the latest frame).
                    segments = line.split("\r") if "\r" in line else [line]
                    for seg in segments:
                        _scan_segment(seg)
                    visible = segments[-1] if segments else line
                    overlay.append_log(visible)
                # Drive progress detection off the still-unterminated
                # tail too, so tqdm's "in flight" redraws update the
                # bar between newlines.
                if "\r" in tick_state["line_buf"]:
                    for seg in tick_state["line_buf"].split("\r"):
                        _scan_segment(seg)

            if not result_state["done"]:
                return

            timer.stop()
            # Flush any trailing partial line into the log so the user
            # can read it.
            if tick_state["line_buf"].strip():
                overlay.append_log(tick_state["line_buf"].split("\r")[-1])

            exc = result_state["exc"]
            if exc is not None:
                # Stringified KeyError / IndexError / etc. is often just
                # the offending key ("0"), which is useless on its own.
                # Pair it with the type name + repr fallback so the
                # summary is always self-describing; the traceback was
                # already streamed into the overlay log on the worker
                # side.
                exc_str = str(exc) or repr(exc)
                sys.__stderr__.write(f"{title} failed: {type(exc).__name__}: {exc_str}\n")
                overlay.mark_done(
                    success=False,
                    summary=f"{type(exc).__name__}: {exc_str}",
                )
                return

            # work_fn succeeded -- run on_success (e.g. layer refresh)
            # on the GUI thread BEFORE showing Done, so a follow-up
            # error is surfaced in the overlay instead of after it
            # dismisses.
            value = result_state["value"]
            if on_success is not None:
                try:
                    on_success(value)
                except Exception as e:  # noqa: BLE001
                    # Mirror the worker-side failure path: stream the
                    # traceback into the overlay log so the user can
                    # diagnose without depending on the launching
                    # terminal, and fall back to repr() for exceptions
                    # whose str() is empty (bare ``assert X`` produces
                    # ``AssertionError`` with str() == "").
                    tb = traceback.format_exc()
                    sys.__stderr__.write(
                        f"{title} succeeded but follow-up failed:\n{tb}"
                    )
                    for line in tb.splitlines():
                        overlay.append_log(line)
                    exc_str = str(e) or repr(e)
                    overlay.mark_done(
                        success=False,
                        summary=(
                            f"Work succeeded, but follow-up step raised "
                            f"{type(e).__name__}: {exc_str}"
                        ),
                    )
                    return

            overlay.mark_done(success=True, summary=success_summary)

        timer.timeout.connect(_drain_and_update)
        thread.start()
        timer.start()

    def _refresh_dlc_layers(self, video_index: int = 0):
        """Load any annotation files produced by DLC training into the
        live DUSTrack session, plus a fresh empty ``iteration-{N+1}``
        layer to capture next-round manual refinements, set newly-added
        dense layers (DLC inference + any LK-RSTC output, see
        :func:`_is_dense_layer_name`) to a line plot, point the overlay
        statevar at the freshest ``dlc_*`` trace, and activate the new
        iteration layer so the user can immediately start annotating.

        Mirrors the loading + new-layer logic in
        :meth:`DLCProject.annotate` but operates on the existing
        ``self`` rather than spawning a new DUSTrack window. Idempotent:
        layers already present in ``self.annotations`` are skipped, and
        the new-iteration layer is only requested if the iteration
        suffix isn't already in the session.
        """
        # Mirror annotate()'s suffix logic: after a successful training,
        # latest_iteration_is_trained() is True so we want N+1; if
        # training is partial / not yet done, fall back to the current
        # iteration number (matches annotate() exactly).
        if self._dlcproject.latest_iteration_is_trained():
            new_iter = self._dlcproject.latest_iteration + 1
        else:
            new_iter = self._dlcproject.latest_iteration
        new_suffix = f"iteration-{new_iter}"

        fm = VideoFileManager(self._dlcproject, video_index)
        # get_new_json (called inside get_all_annotation_layers when a
        # suffix is passed) raises ValueError if the file already exists
        # on disk -- e.g. user trained, refined + saved, then re-trained
        # without restarting. Fall back to the no-suffix call in that
        # case; the layer is already known to the session via the
        # original annotate() / earlier _refresh_dlc_layers run.
        try:
            all_layers = fm.get_all_annotation_layers(new_suffix)
            requested_new_iter = True
        except ValueError:
            all_layers = fm.get_all_annotation_layers()
            requested_new_iter = False

        existing = set(self.annotations.names)
        new_layers = {
            name: path for name, path in all_layers.items() if name not in existing
        }
        if not new_layers:
            # Even with nothing to load, point the active layer at the
            # next-iteration layer if it's already present -- keeps the
            # post-training UX consistent across first-run vs re-run.
            if requested_new_iter and new_suffix in self.annotations.names:
                self.statevariables["annotation_layer"].set_state(new_suffix)
                self.update()
            return

        self.add_annotation_layers(new_layers)
        # Limit scope to newly-added layers so an in-place refresh that
        # adds zero new dlc_ layers preserves whatever overlay the user
        # had selected. On a fresh DUSTrack (DLCProject.annotate path),
        # the same helper is called with scope=None so it operates on
        # all current dlc_ layers.
        self._normalize_dlc_layer_display(scope=new_layers.keys())
        # Reset empty manual layers' labels to match the project's
        # bodyparts. Two issues this fixes (both visible after
        # seeding from an external snapshot): (a) ``add_annotation_layers``'
        # union pass adds missing labels but never removes the
        # session-bootstrap default ``"0"``, leaving spurious labels
        # when bodyparts don't start at 0; (b) for the simple
        # ``["point0", "point1"]`` case the data ends up correct but
        # the ``annotation_label`` dropdown was set at __init__ from
        # a one-label ``self.ann.labels`` snapshot and would otherwise
        # stay stale. See :func:`_dlc_bodyparts_to_layer_labels`.
        self._normalize_empty_manual_layer_labels()
        if new_suffix in self.annotations.names:
            self.statevariables["annotation_layer"].set_state(new_suffix)
        self._restructure_annotation_order()
        # Re-bootstrap the label-related statevariables from the
        # (possibly just-normalised) active layer so the dropdown
        # reflects the project's bodyparts instead of the stale
        # ``__init__``-time snapshot.
        self._rebootstrap_label_states()
        self.update()

    def _normalize_empty_manual_layer_labels(self) -> None:
        """When the project's bodyparts are known, reset every *empty*
        manual annotation layer's labels to match the bodyparts exactly.

        Empty here means ``not any(ann.data.values())`` -- no label
        has any frames. Layers with real annotations are untouched.

        Without this, two failure modes show up after seeding from
        an external snapshot bundle:

        - The session-bootstrap ``"manual"`` layer is created with
          ``{"0": {}}`` (the ``n_labels=1`` default in
          :class:`VideoAnnotation`). The union pass in
          :meth:`add_annotation_layers` adds new labels but never
          removes the bootstrap ``"0"``, so a bundle with bodyparts
          ``["point1", "point3"]`` produces a layer with labels
          ``["0", "1", "3"]`` instead of ``["1", "3"]``.
        - Even when bodyparts are ``["point0", "point1"]`` and the
          union ends up correct, the active layer's labels were
          captured at ``__init__`` time into the
          ``annotation_label`` statevariable; without re-bootstrap
          (see :meth:`_rebootstrap_label_states`) the dropdown shows
          only the original ``"0"``.

        Saves the layer to disk if its JSON path is already known
        (set by :meth:`_rewire_to_in_project_paths`) so the on-disk
        annotation file reflects the new label set on the next
        cold-open.
        """
        if self._dlcproject is None:
            return
        bodyparts = self._dlcproject.config.get("bodyparts") or []
        if not bodyparts:
            return
        target_labels = _dlc_bodyparts_to_layer_labels(bodyparts)
        for ann in self.annotations._list:
            if ann.name == "buffer":
                continue
            if not self._is_manual_layer_name(ann.name):
                continue
            if any(ann.data.values()):
                continue  # real data present -- preserve as-is
            if list(ann.data.keys()) == target_labels:
                continue  # already canonical
            ann.data = {label: {} for label in target_labels}
            ann.sort_labels()
            ann.re_setup_display()
            if ann.fname is not None and Path(ann.fname).suffix == ".json":
                ann.save()

    def _rebootstrap_label_states(self) -> None:
        """Refresh the ``label_range`` + ``annotation_label``
        statevariables from the *current* active layer's labels.

        At construction time, :class:`_DUSTrackBase.__init__` reads
        ``self.ann.labels`` once to seed these statevariables. Any
        downstream code that changes the active layer's labels
        (e.g. :meth:`_refresh_dlc_layers` after train / seed, which
        adds DLC trace layers and may rewrite ``iteration-N``
        labels via :meth:`_normalize_empty_manual_layer_labels`)
        leaves the dropdown stale unless this is called. The
        contents mirror the ``__init__`` bootstrap.
        """
        if not self.ann.labels:
            return
        first_label = self.ann.labels[0]
        try:
            label_int = int(first_label)
        except (TypeError, ValueError):
            return  # non-numeric labels skip the range bootstrap
        self.statevariables["label_range"].set_state(label_int // 10)
        self.update_annotation_label_states()
        states = self.statevariables["annotation_label"].states
        if states:
            self.statevariables["annotation_label"].set_state(states[0])

    def _normalize_dlc_layer_display(self, scope=None):
        """Apply the post-load display convention for DLC-pipeline layers:
        every *dense* layer renders as a line plot (DLC inference plus
        any LK-RSTC jitter-reduced output, regardless of source layer),
        and the latest ``dlc_*`` inference layer is set as the
        ``annotation_overlay``.

        Two predicates, on purpose:

        - *Dense* (see :func:`_is_dense_layer_name`) drives the
          plot-type pass. Broader than ``dlc_*`` so a Reduce-jitter
          output named ``dlccorr_lkmovavg_0.500`` (LK over the
          manual-corrections layer) renders continuously like every
          other DLC-pipeline trace.
        - ``dlc_*`` (narrow) drives the overlay pin. Overlay semantics
          are "show me the model prediction next to manual", which a
          smoothing artifact doesn't satisfy -- if the latest dense
          layer were used, clicking Reduce jitter on a manual layer
          would silently swap the overlay onto the smoothed layer.

        ``scope`` selects which layer names participate (the plot-type
        pass is idempotent so re-running on an already-line layer is a
        no-op):

        - ``None`` -- fresh-construction path: scope = all current
          layers in the session. Always (re-)points the overlay if at
          least one dlc_* layer exists.
        - iterable of names -- in-place refresh path: scope = the
          freshly-added layers only. If none of them are ``dlc_*``,
          the overlay isn't touched (preserves prior selection).

        Single source of truth shared by :meth:`DLCProject.annotate`,
        :meth:`_refresh_dlc_layers`, and :meth:`_adopt_layer` so the
        on-screen state is identical regardless of how the user
        entered the session.
        """
        if scope is None:
            names = [a.name for a in self.annotations]
        else:
            names = list(scope)
        dense_names = [n for n in names if _is_dense_layer_name(n)]
        for name in dense_names:
            self.annotations[name].set_plot_type("line")
        dlc_names = [n for n in names if n.startswith("dlc_")]
        if dlc_names:
            self.statevariables["annotation_overlay"].set_state(dlc_names[-1])

    def _restructure_annotation_order(self) -> None:
        """Regroup ``self.annotations`` into the canonical layer order
        produced by :meth:`DLCProject.annotate` at fresh load:
        ``manuals -> manual_corrections -> labeled_data -> dlc_* ->
        dlccorr* -> buffer``.

        In-session adds (post-train :meth:`_refresh_dlc_layers`,
        Reduce-jitter / :meth:`apply_manual_corrections` via
        :meth:`_adopt_layer`) append to the end of ``self.annotations``,
        which interleaves manuals with prior DLC layers and breaks the
        grouping a returning user would see on close+reopen. This helper
        re-runs after those paths so the dropdown rotation matches the
        fresh-load order regardless of how the session got here.

        Six groups, classified by layer name:

        - ``buffer`` -- the scratch layer, always last.
        - ``labeled_data`` -- DLC training-input HDF5.
        - ``dlc_*`` (prefix) -- DLC inference traces and any LK-RSTC
          smoothed version of one. Names emitted only by
          :meth:`VideoFileManager.canonical_layer_name` for files under
          ``videos/iteration-*/``.
        - ``dlccorr*`` (prefix) -- the manual-corrections splice
          produced by :meth:`apply_manual_corrections` and any
          downstream LK output of it (``dlccorr_lkmovavg_*``). Its own
          group at the tail of the DLC chain rather than folded into
          manuals: it isn't a hand-edited layer (the manual entries it
          incorporates live in a separate active layer), and isn't
          folded into the ``dlc_*`` group either since it isn't a model
          inference.
        - ``*_manual_corrections`` (suffix) -- the source-of-corrections
          layer (typically ``iteration-N_manual_corrections``) that
          :meth:`apply_manual_corrections` renames the patch layer to
          on a successful splice. Distinct from the regular manuals
          block: it's the canonical "this was the live correction
          source for the dlccorr that exists" marker, and grouped
          right after manuals so the relationship to its
          ``iteration-N`` prefix peer stays visually adjacent.
        - manuals -- everything else, i.e. ``*_annotations_*.json``-
          backed layers that aren't ``buffer`` / ``dlccorr*`` /
          ``*_manual_corrections``.

        Intra-group order is preserved (so the user's current ordering
        within each group survives a refresh). Active layer and overlay
        are preserved by name across the reorder.
        """
        manuals: list[str] = []
        manual_corr: list[str] = []
        labeled: list[str] = []
        dlc: list[str] = []
        dlccorr: list[str] = []
        buf: list[str] = []
        for ann in self.annotations._list:
            name = ann.name
            if name == "buffer":
                buf.append(name)
            elif name == "labeled_data":
                labeled.append(name)
            elif name.startswith("dlccorr"):
                dlccorr.append(name)
            elif name.startswith("dlc_"):
                dlc.append(name)
            elif name.endswith("_manual_corrections"):
                manual_corr.append(name)
            else:
                manuals.append(name)

        target = manuals + manual_corr + labeled + dlc + dlccorr + buf
        if target == self.annotations.names:
            return

        active = None
        overlay = None
        if "annotation_layer" in self.statevariables.names:
            active = self.statevariables["annotation_layer"].current_state
        if "annotation_overlay" in self.statevariables.names:
            overlay = self.statevariables["annotation_overlay"].current_state

        self.annotations.reorder(target)
        self._refresh_annotation_state_lists()

        if active is not None and active in self.annotations.names:
            self.statevariables["annotation_layer"].set_state(active)
        if "annotation_overlay" in self.statevariables.names:
            if overlay is None or overlay in self.annotations.names:
                self.statevariables["annotation_overlay"].set_state(overlay)

    def _adopt_layer(
        self,
        ann_or_fname,
        *,
        set_active: bool = False,
        set_overlay=None,
    ) -> str | None:
        """Adopt an in-session-produced annotation file into the live layer list
        under the same conventions used by the cold-open (:meth:`DLCProject.annotate`)
        and post-train (:meth:`_refresh_dlc_layers`) paths.

        Single entry point for layer additions that bypass
        :class:`VideoFileManager` (e.g. :meth:`process_with_lk`):

        - Re-derives the layer name from the filepath via
          :meth:`VideoFileManager.canonical_layer_name`. Any caller-provided
          ``.name`` on a :class:`VideoAnnotation` is ignored -- the path is
          authoritative, so a freshly-built ``VideoAnnotation`` whose
          ``.name`` would otherwise fall back to ``"noname"`` gets the
          same name it would receive on reload.
        - Adds via :meth:`add_annotation_layers` ``{name: fname}`` so the
          dnav plotting / buffer / label-union plumbing fires identically
          to the bulk-load path.
        - If the layer name is a *dense* DLC-pipeline output (see
          :func:`_is_dense_layer_name`), runs
          :meth:`_normalize_dlc_layer_display` over just this one layer
          so plot-type (and, for ``dlc_*`` names, ``annotation_overlay``)
          end up where they would after a close + reopen.
        - Skips (returns ``None``) if a layer with that name is already
          loaded -- mirroring :meth:`_refresh_dlc_layers`'s idempotency.

        Args:
            ann_or_fname: A :class:`VideoAnnotation`, ``Path``, or path
                string pointing at the annotation file on disk.
            set_active: If True, set ``annotation_layer`` to the new layer.
            set_overlay: If not None, set ``annotation_overlay`` to this
                layer name. Use this to pin the original source layer as
                overlay when the new layer is a derived (e.g. smoothed)
                version, matching the pre-harmonization behaviour of
                :meth:`process_with_lk`.

        Returns:
            The canonical layer name on success, or ``None`` if the layer
            was already loaded.
        """
        # Before dustrack 1.2.0a1 dnav.VideoAnnotation was the parent
        # and the dustrack subclass added the ``postprocess`` hook --
        # ``lk_moving_average_filter`` would return the parent type so
        # ``isinstance(..., dustrack.VideoAnnotation)`` silently fell
        # through to ``str(obj)``. Now there's a single VideoAnnotation
        # class (relocated to dustrack with postprocess attached at
        # import time in dustrack/__init__.py); the check is unambiguous.
        if isinstance(ann_or_fname, VideoAnnotation):
            fname = ann_or_fname.fname
        else:
            fname = str(ann_or_fname)
        name = VideoFileManager.canonical_layer_name(fname)
        already_loaded = name in self.annotations.names
        if not already_loaded:
            self.add_annotation_layers({name: fname})
            # Promote the freshly-added layer to the dustrack subclass so
            # ``ann.postprocess`` is available in-session (matches the
            # __init__-time promotion done after the bulk add).
            new_layer = self.annotations[name]
            if not isinstance(new_layer, VideoAnnotation):
                new_layer.__class__ = VideoAnnotation
            if _is_dense_layer_name(name):
                self._normalize_dlc_layer_display(scope=[name])
        # Apply the requested overlay / active state even if the layer
        # was already present -- e.g. Reduce jitter on a layer whose
        # cached output is already loaded should still swap the UI to
        # the smoothed layer with the source pinned as overlay.
        if set_overlay is not None:
            self.statevariables["annotation_overlay"].set_state(set_overlay)
        if set_active:
            self.statevariables["annotation_layer"].set_state(name)
        if not already_loaded:
            self._restructure_annotation_order()
        return None if already_loaded else name

    def process_with_lk(self, event=None, *args, **kwargs):
        """
        Apply Lucas-Kanade optical flow post-processing to reduce tracking jitter.

        rc2 (1.1.0rc2): on a Qt backend, the LK-RSTC pass runs off the
        GUI thread under a progress overlay that mirrors the Train DLC
        flow (tqdm bars drive the progress widget; phase label
        reflects "Submitting" vs "Processing"; Done button lets the
        user confirm before the smoothed layer swaps in). On non-Qt
        backends the call runs synchronously and returns the new
        :class:`VideoAnnotation`.

        Uses the Lucas-Kanade RSTC (Reverse Sigmoid Tracking
        Correction) algorithm to smooth trajectories. The processed
        annotation is saved and added as a new layer, with the
        original set as overlay for comparison.

        rc2 layer-naming harmonisation: the new layer is adopted via
        :meth:`_adopt_layer`, which derives its name from the output
        filepath via :meth:`VideoFileManager.canonical_layer_name` --
        identical to what a close + reopen would show. Replaces the
        earlier behaviour where the in-session layer briefly carried
        the ``"noname"`` fallback until reload. When the source layer
        is a DLC trace, the smoothed output is also plot-type-normalised
        to ``"line"`` so it looks identical to other ``dlc_*`` layers.

        The source layer is `save()`-ed to disk right before LK
        kicks off (mirroring the pre-train save in
        :meth:`process_dlc_project`), so the on-disk state matches
        what LK sees. In the typical workflow the source is the
        ``dlccorr`` layer (active after
        :meth:`apply_manual_corrections`) and the save persists any
        in-memory manual edits. Sources without a ``.json`` filename
        (e.g. raw DLC traces loaded from ``.h5``) are read-only
        inputs; the save is skipped for them with a one-line note.

        Args:
            event: Mouse/keyboard event (unused, for button compatibility).
            *args: Additional arguments passed to lk_moving_average_filter.
            **kwargs: Additional keyword arguments (e.g., window_size) passed to filter.

        Returns:
            VideoAnnotation: The smoothed annotation layer on the sync
            path. ``None`` on the Qt async path -- the new layer is
            added to ``self.annotations`` when the Done button is
            clicked.

        See Also:
            dustrack.postprocess.lk_moving_average_filter: The filtering algorithm.
        """
        source_ann = self.ann
        source_layer_name = source_ann.name

        # Persist the source layer to disk before smoothing kicks off
        # so the on-disk state matches what LK sees. Skip gracefully
        # for non-JSON-backed sources (raw DLC traces / h5) since
        # VideoAnnotation.save() raises on non-json filenames.
        source_fname = getattr(source_ann, "fname", None)
        if source_fname is not None and Path(source_fname).suffix == ".json":
            source_ann.save()
        else:
            print(
                f"[reduce_jitter] skipping pre-save of source layer "
                f"{source_layer_name!r} (no .json filename to anchor to)."
            )

        # GUI default: skip the per-window .pkl sidecar. The .pkl is a
        # contract for ``pn-projects/wobble`` and ``gaitmusic`` callers
        # (their ``.rawlk`` property feeds ``lk_gradients`` velocity
        # estimation) -- but those go through the direct API, not the
        # button. GUI users almost never inspect per-window data; the
        # sidecar is ~10-12x larger than the averaged .json and the
        # write costs a couple of seconds on real videos. Callers who
        # want it back can pass ``save_raw=True`` explicitly.
        kwargs.setdefault("save_raw", False)

        def _smooth():
            ann_processed = lk_moving_average_filter(source_ann, *args, **kwargs)
            ann_processed.save()
            return ann_processed

        qt_window = self._find_qt_window()
        if qt_window is None:
            ann_processed = _smooth()
            self._adopt_layer(
                ann_processed,
                set_active=True,
                set_overlay=source_layer_name,
            )
            self.update()
            return ann_processed

        def _on_success(ann_processed):
            self._adopt_layer(
                ann_processed,
                set_active=True,
                set_overlay=source_layer_name,
            )
            self.update()

        self._run_with_overlay(
            qt_window,
            work_fn=_smooth,
            on_success=_on_success,
            title=f"Reducing jitter ({source_layer_name})",
            initial_phase=f"Preparing LK-RSTC pass on layer {source_layer_name!r}",
            hint=(
                "Output is also streamed to the launching terminal. "
                "The smoothed layer will load when you click Done."
            ),
            show_progress_bar=True,
            phase_patterns=_JITTER_PHASES,
            success_summary=(
                f"Jitter reduction complete on layer {source_layer_name!r}. "
                f"Smoothed layer loaded."
            ),
        )
        return None
    
    def remove_current_layer(self, event=None):
        """Remove the active annotation layer from the DUSTrack session.

        Session-only: the underlying JSON / HDF5 file on disk is *not*
        touched -- so on next launch the layer reappears if its file
        is still next to the video. Pair with a manual file delete
        (or ``Save annotation as...`` to a different name + delete the
        original) when the intent is "undo manual corrections".

        Refuses with a notice if only one removable layer remains in
        the session (excluding the implicit ``"buffer"`` layer, which
        is never user-visible but always present). Otherwise confirms
        via :class:`ConfirmOverlay`; the confirm body is severity-
        aware via :func:`_is_dense_layer_name`:

        - Dense / derived (``dlc_*``, ``dlccorr``, ``*lkmovavg*``):
          regenerable, default button = Remove.
        - Sparse / authored (manual labels): irreversible, default
          button = Cancel.
        """
        qt_window = self._find_qt_window()
        layer_name = self._current_layer
        if layer_name == "buffer":
            # Defensive: ``buffer`` should never be the user-selected
            # primary, but guard anyway.
            return

        removable = [
            n for n in self.annotations.names if n != "buffer"
        ]
        if len(removable) <= 1:
            if qt_window is None:
                return
            ConfirmOverlay = _make_confirm_overlay_class()
            ConfirmOverlay(
                qt_window,
                title="Cannot remove only remaining layer",
                message=(
                    f"Layer {layer_name!r} is the only annotation "
                    "layer in this session (excluding the internal "
                    "buffer). Removing it would leave the session "
                    "with no editable layer.\n\n"
                    "Use Discard unsaved annotations to reset its "
                    "contents instead."
                ),
                buttons=[("OK", "neutral")],
                default="OK",
                severity="info",
            ).exec_()
            return

        if qt_window is None:
            # mpl fallback: drop the layer silently.
            self.remove_annotation_layer(layer_name)
            self.update()
            self._refresh_workflow_button_state()
            return

        ConfirmOverlay = _make_confirm_overlay_class()
        if _is_dense_layer_name(layer_name):
            if layer_name == "dlccorr":
                regen_hint = "re-running Apply manual corrections"
            elif "lkmovavg" in layer_name:
                regen_hint = "re-running Reduce jitter"
            else:
                regen_hint = "re-running the DLC inference / training pipeline"
            body = (
                f"Remove regenerable layer {layer_name!r}?\n\n"
                f"This layer can be reproduced by {regen_hint}. "
                "The layer is dropped from the current session only; "
                "the backing file on disk is not deleted, so the "
                "layer will reappear on next launch unless you "
                "remove the file manually."
            )
            default = "Remove layer"
            severity = "warning"
        else:
            n_frames = len(self.annotations[layer_name].frames)
            body = (
                f"Remove layer {layer_name!r}? "
                f"{n_frames} manually annotated frame(s) will be "
                "dropped from this session.\n\n"
                "The backing file on disk is not deleted, so the "
                "layer will reappear on next launch unless you "
                "remove the file manually. Once removed from the "
                "session, any in-memory edits since the last save "
                "are lost."
            )
            default = "Cancel"
            severity = "destructive"

        result = ConfirmOverlay(
            qt_window,
            title="Remove layer",
            message=body,
            buttons=[
                ("Remove layer", "destructive"),
                ("Cancel", "neutral"),
            ],
            default=default,
            severity=severity,
        ).exec_()
        if result == "Remove layer":
            self.remove_annotation_layer(layer_name)
            self.update()
            # Removing the active or overlay layer may change which
            # buttons are valid; the statevar on_change covers the
            # active/overlay re-pick but not the more general "an
            # overlay layer disappeared from the dropdown" case.
            self._refresh_workflow_button_state()

    def copy_existing_annotations_from_overlay(self, event=None):
        """
        Copy overlay annotation points to the current annotation layer for selected frames.
        
        Useful when DLC predictions are more accurate than manual labels.
        Typically used with manual annotations in the primary annotation layer,
        and the model predictions in the overlay layer. Data is copied only for
        frames that exist in the primary annotation layer. Perform this action
        within a specified frame range by selecting an interval.
        
        Args:
            event: Mouse/keyboard event (unused, for button compatibility).
        
        Raises:
            ValueError: If no annotation overlay is currently selected.
        
        Note:
            Only affects the current label. Other labels remain unchanged.
        """
        overlay_name = self._current_overlay
        if overlay_name is None:
            raise ValueError('No annotation overlay selected.')
        overlay_ann = self.annotations[overlay_name]
        current_label = self._current_label
        if (self._current_layer, current_label) in self.events[0].to_dict():
            event_start, event_end = self.events[0].to_dict()[(self._current_layer, current_label)][0]
        else:
            event_start, event_end = 0, self.ann.n_frames - 1
        # if an event is specified, nudge data only in the selected interval
        for frame_num in self.ann.frames:
            if event_start <= frame_num <= event_end:
                self.ann.add(overlay_ann.data[current_label][frame_num], current_label, frame_num)
        self.update()

    @staticmethod
    def _merge_overlay_with_patch(source_data, patch_data):
        """Merge two annotation ``data`` dicts -- patch overrides source.

        Both inputs are nested dicts in :class:`VideoAnnotation` shape:
        ``{label: {frame_num: [x, y]}}``. The result starts from
        ``source_data`` (the baseline, typically dense DLC predictions),
        then layers ``patch_data`` on top: at any (label, frame) present
        in patch, the patch value wins. Labels that appear in only one
        of the two inputs are carried through with their full contents.

        Pure function so the splicing logic can be exercised from
        synthetic inputs without instantiating the GUI; consumed by
        :meth:`apply_manual_corrections`.
        """
        all_labels = sorted(set(source_data) | set(patch_data))
        merged = {}
        for label in all_labels:
            merged[label] = {}
            if label in source_data:
                merged[label].update(source_data[label])
            if label in patch_data:
                merged[label].update(patch_data[label])
        return merged

    # Suffix appended to the source-of-corrections layer's name on a
    # successful :meth:`apply_manual_corrections` run. The full new name
    # is ``{old_name}_manual_corrections`` -- typically
    # ``iteration-N_manual_corrections`` -- so the rename is visually
    # adjacent to its ``iteration-N`` peer in the layer dropdown and
    # picked up by the ``_manual_corrections`` suffix branch in
    # :meth:`_restructure_annotation_order`.
    MANUAL_CORRECTIONS_SUFFIX = "_manual_corrections"

    def apply_manual_corrections(self, event=None):
        """Splice the active layer's manual entries into the overlay to produce/refresh the ``dlccorr`` layer.

        **Workflow context** (step 4 of the DUSTrack pipeline):
        after iterating DLC training (steps 2-3), you flip into a
        manual-correction mode. The active annotation layer holds
        sparse hand-edits (a few frames where DLC was wrong); the
        annotation overlay points at the DLC trace you want to
        correct. Clicking this button produces a layer named
        :attr:`CORRECTIONS_LAYER_NAME` (``"dlccorr"``) that is the
        overlay's data with your manual entries spliced in wherever
        they exist. The file lives next to the video as
        ``<video>_annotations_dlccorr.json`` and is automatically
        excluded from DLC training input (the ``_dlccorr`` filter in
        :func:`_extract_frames`) -- this is a terminal output.

        **Preflight save.** If the active (patch) layer has any
        unsaved in-memory edits, they are written to disk before the
        splice runs. The user has explicitly clicked Apply, so
        "intent to commit" is signalled; saving the source as a side
        effect keeps the on-disk state coherent with ``dlccorr``
        (which is always saved by this method).

        **Source-layer rename.** On a successful splice the patch
        layer is renamed in place from ``<old_name>`` to
        ``<old_name>_manual_corrections`` (the file is moved on disk
        too, old file deleted). The rename marks "this is the layer
        whose manual entries the on-disk ``dlccorr`` was spliced from"
        so the relationship survives reload. No-op if the patch is
        already named ``*_manual_corrections`` (idempotent re-apply).

        **Post-apply state.** The corrections layer becomes the
        active annotation layer and the (now-renamed) manual layer
        becomes the overlay so you can see where your hand was. To
        iterate, switch the active layer back to your manual layer,
        set the overlay back to the DLC trace, add more points, click
        again. Each click regenerates the corrections layer from the
        current ``(overlay, active)`` pair, so adding annotations
        directly to the corrections layer is not recommended --
        they'll be discarded on the next apply.

        **Idempotency.** If a ``dlccorr`` layer is already present in
        the session, its in-memory data is replaced wholesale (with
        a ``_revision`` bump so the trace cache invalidates) and the
        file overwritten. Otherwise a fresh :class:`VideoAnnotation`
        is built, saved, and adopted under the canonical machinery.

        Args:
            event: Mouse/keyboard event (unused, for button compat).

        Raises:
            ValueError: If no annotation overlay is currently set, or
                if the active annotation layer is already the
                corrections layer (would create a circular splice).
        """
        overlay_name = self._current_overlay
        if overlay_name is None:
            raise ValueError(
                "No annotation overlay selected. Set the layer you want to "
                "correct (typically a dlc_* trace) as the overlay before "
                "applying."
            )
        patch = self.ann
        patch_name = patch.name
        if patch_name == self.CORRECTIONS_LAYER_NAME:
            raise ValueError(
                f"The active annotation layer is already {self.CORRECTIONS_LAYER_NAME!r}. "
                "Switch the active layer to your manual annotations layer first; "
                "the corrections layer is a derived output and shouldn't be the "
                "splice input."
            )

        # Preflight: save the patch layer if it has unsaved edits, so
        # the on-disk file matches the in-memory data the splice is
        # about to consume. Scoped to the patch layer only -- the
        # overlay is typically a dlc_* h5 trace and isn't editable
        # interactively. See the apply_manual_corrections docstring
        # ("Preflight save") for the rationale.
        if patch.fname is not None:
            mem_data = self._normalize_layer_data(patch.data)
            disk_data = self._load_layer_disk_data(patch.fname)
            diff = self._diff_ann_vs_disk(mem_data, disk_data)
            if any(diff.values()):
                patch.save()
                print(
                    f"Apply manual corrections: auto-saved {patch_name!r} "
                    "(had unsaved edits) before splicing."
                )

        source = self.annotations[overlay_name]

        merged_data = self._merge_overlay_with_patch(source.data, patch.data)

        if self.CORRECTIONS_LAYER_NAME in self.annotations.names:
            # Refresh in place: replace the layer's data wholesale so the
            # file on disk and the in-memory layer agree. Bumping
            # _revision invalidates the per-label trace cache (the
            # caches keyed on (label, plot_type, _revision) -- see the
            # revision-counter pattern note).
            layer = self.annotations[self.CORRECTIONS_LAYER_NAME]
            layer.data = merged_data
            layer._revision += 1
            layer.save()
        else:
            # First-time creation: build a fresh VideoAnnotation pointing
            # at the canonical filename, populate its data, save, and
            # adopt under the same machinery used for any other layer.
            fname = make_annotation_file_name(self.fname, self.CORRECTIONS_LAYER_NAME)
            new_ann = VideoAnnotation(fname=str(fname), vname=self.fname)
            new_ann.data = merged_data
            new_ann._revision += 1
            new_ann.save()
            self._adopt_layer(new_ann)

        # Rename the patch layer to <old>_manual_corrections so the
        # source-of-corrections is identifiable on reload. Skip if
        # already suffixed (idempotent re-apply -- e.g. user re-clicked
        # Apply without changing the active layer back to the renamed
        # one in the overlay flip).
        if not patch_name.endswith(self.MANUAL_CORRECTIONS_SUFFIX):
            new_patch_name = patch_name + self.MANUAL_CORRECTIONS_SUFFIX
            self._rename_annotation_layer(patch_name, new_patch_name)
            patch_name = new_patch_name

        self.statevariables["annotation_layer"].set_state(self.CORRECTIONS_LAYER_NAME)
        self.statevariables["annotation_overlay"].set_state(patch_name)
        self._restructure_annotation_order()
        self.update()

    def _rename_annotation_layer(self, old_name: str, new_name: str) -> None:
        """Rename an annotation layer in-place: ``.name`` + ``.fname`` +
        on-disk file. The old on-disk file is deleted after the new file
        is written so a crash mid-rename leaves at least one copy on
        disk. Statevariable rotations and the active/overlay selections
        are resynced; selections that pointed at the old name are
        re-pinned to the new name.

        Used by :meth:`apply_manual_corrections` to tag the
        source-of-corrections layer with the ``_manual_corrections``
        suffix on a successful splice; written as a general helper
        because the rename mechanics (file move + statevar resync +
        selection preservation) are independent of that workflow.
        """
        if old_name not in self.annotations.names:
            raise KeyError(f"No annotation layer named {old_name!r}.")
        if new_name in self.annotations.names:
            raise ValueError(
                f"Cannot rename {old_name!r} -> {new_name!r}: a layer with "
                f"the target name already exists."
            )

        ann = self.annotations[old_name]
        old_fname = ann.fname
        # ``make_annotation_file_name`` produces
        # ``<video>_annotations_<new_name>.json`` so the on-disk filename
        # stem-after-``_annotations_`` equals the layer name -- the same
        # invariant :meth:`VideoFileManager.canonical_layer_name` relies
        # on when round-tripping a name from disk.
        new_fname = make_annotation_file_name(self.fname, new_name)

        # Write the new file before unlinking the old, so an
        # interrupted rename leaves the data on disk.
        ann.save(new_fname)
        ann.fname = new_fname
        ann.name = new_name
        if old_fname is not None and Path(old_fname).exists() and str(old_fname) != str(new_fname):
            try:
                Path(old_fname).unlink()
            except OSError as exc:
                # Don't strand the user with a broken rename: log and
                # continue. The new file is in place; the stale old
                # file at worst gets picked up by the next training
                # pass alongside the new one (extract_frames will
                # merge them), which is recoverable.
                print(
                    f"Warning: could not delete old annotation file "
                    f"{old_fname!r} after rename to {new_fname!r}: {exc}"
                )

        # Resync statevariable rotations + preserve selections that
        # pointed at the old name.
        self._refresh_annotation_state_lists()
        if "annotation_layer" in self.statevariables.names:
            sv = self.statevariables["annotation_layer"]
            if sv.current_state == old_name:
                sv.set_state(new_name)
        if "annotation_overlay" in self.statevariables.names:
            sv = self.statevariables["annotation_overlay"]
            if sv.current_state == old_name:
                sv.set_state(new_name)

    def save_annotation_as(self, event=None):
        """Save the active annotation layer to a user-chosen path.

        Opens a Qt save-file dialog seeded with the video's folder and a
        suggested filename of ``<video_stem>_annotations_<layer>.json``.
        Falls back to ``self.ann.save()`` (which writes to the layer's
        existing ``.fname``) when no Qt window is available -- e.g.
        headless / Agg backend.
        """
        layer = self.ann
        suggested_dir = str(Path(self.fname).parent)
        layer_name = self._current_layer or ""
        suggested_name = Path(
            make_annotation_file_name(self.fname, layer_name)
        ).name
        suggested_path = str(Path(suggested_dir) / suggested_name)

        qt_window = self._find_qt_window()
        if qt_window is None:
            layer.save()
            return

        from qtpy.QtWidgets import QFileDialog

        fname, _ = QFileDialog.getSaveFileName(
            qt_window,
            "Save annotation layer as...",
            suggested_path,
            "Annotation JSON (*.json);;All files (*)",
        )
        if not fname:
            return
        if not fname.lower().endswith(".json"):
            fname += ".json"
        layer.save(fname)

    def swap_active_and_overlay(self, event=None):
        """Swap the active annotation layer with the overlay layer.

        No-op if no overlay is currently selected.
        """
        active = self._current_layer
        overlay = self._current_overlay
        if overlay is None:
            return
        self.statevariables["annotation_layer"].set_state(overlay)
        self.statevariables["annotation_overlay"].set_state(active)
        self.update()

    def update(self):
        """
        Update the display with current frame and maintain frozen axis limits if set.
        
        Returns:
            Result from parent class update() method.
        """
        ret = super().update()
        if self._ax_lims['state']:
            if self._ax_lims['x'][0] is not None:
                self._ax_trace_x.set_xlim(self._ax_lims['x'])
                self._ax_trace_y.set_xlim(self._ax_lims['x'])
            if self._ax_lims['y_trace_x'][0] is not None:
                self._ax_trace_x.set_ylim(self._ax_lims['y_trace_x'])
            if self._ax_lims['y_trace_y'][0] is not None:
                self._ax_trace_y.set_ylim(self._ax_lims['y_trace_y'])
            plt.draw()
        return ret


class DLCData(pysampled.Data):
    """
    Data container for DeepLabCut tracking results.
    
    Provides convenient loading and manipulation of DLC output files (HDF5 format),
    with automatic extraction of metadata like body part names and coordinate labels.
    
    Attributes:
        signal_names (list): Names of tracked body parts (e.g., ['nose', 'left_ear']).
        signal_coords (list): Coordinate names (typically ['x', 'y', 'likelihood']).
    
    Example:
        >>> # Load from DLC output file
        >>> data = DLCData.from_hdf('video_dlc_resnet50_model_name.h5')
        >>> 
        >>> # Load from video (finds associated HDF5 file)
        >>> data = DLCData.from_video('video.mp4', iter_num=250000)
    """
    def __setstate__(self, state):
        """
        Restore object state with backwards compatibility.
        
        Handles legacy attribute names ('coords', 'label_names') by converting
        them to current naming convention ('signal_coords', 'signal_names').
        """
        super().__setstate__(state)
        if "coords" in self.meta:
            self.signal_coords = self.meta.pop("coords")
        if "label_names" in self.meta:
            self.signal_names = self.meta.pop("label_names")
    
    @classmethod
    def from_hdf(cls, file_path):
        """
        Load DLC data from an HDF5 file.
        
        Args:
            file_path (str): Path to the DLC output HDF5 file.
        
        Returns:
            DLCData: Loaded data with extracted metadata.
        
        Raises:
            AssertionError: If file doesn't exist.
            FileNotFoundError: If corresponding labeled video cannot be found.
        """
        assert os.path.exists(file_path)
        df_h5 = pd.read_hdf(file_path)
        label_names = list(df_h5.columns.unique(level='bodyparts'))
        coords = list(df_h5.columns.unique(level='coords'))
        vid_paths = pyfilemanager.FileManager(Path(file_path).parent).add()[f'*{Path(file_path).stem}*_labeled.mp4']
        if len(vid_paths) == 0:
            raise FileNotFoundError('Could not find the video file')
        sr = int(cv.VideoCapture(vid_paths[0]).get(cv.CAP_PROP_FPS))
        return DLCData(df_h5.values, sr, meta=dict(label_names=label_names, coords=coords))
    
    @classmethod
    def from_video(cls, vid_path, iter_num=None):
        """
        Load DLC data associated with a video file.
        
        Automatically searches for HDF5 files matching the video name and
        loads the specified training iteration (or the highest if not specified).
        
        Args:
            vid_path (str): Path to the video file.
            iter_num (int, optional): Training iteration number. If None, uses highest.
        
        Returns:
            DLCData: Loaded tracking data.
        
        Raises:
            AssertionError: If video file doesn't exist or requested iteration not found.
        """
        assert os.path.exists(vid_path)
        # find the hdf file
        vid_path = Path(vid_path)
        h5_list = pyfilemanager.FileManager(vid_path.parent).add()[f'{vid_path.stem}*.h5']
        iter_num_to_fname = {int(Path(x).stem.split('_')[-1]):x for x in h5_list}
        if iter_num is None:
            # pick the highest iteration number
            iter_num = max(iter_num_to_fname)
        assert iter_num in iter_num_to_fname
        h5_file = iter_num_to_fname[iter_num]
        return cls.from_hdf(h5_file)


class DLCProject:
    """Interface to deeplabcut training and inference
    Current workflow:
        1. Create a project with some videos. Videos will be copied.
            d = DLCProject(r'C:/data_opr02/004_02/ml_models/dlc', name='opr02_s004_muscles', experimenter='praneeth', videos=[<video_list>])
        2. Launch the initial annnotator for video 0, repeat if there are more videos
            d.annotate(0) 
        3. Extract frames, train network, evaluate network, analyze videos, and create labeled video
            d.process()
        4. Refine the labels
            d.annotate(0, 'praneeth_2') # the second argument determines the suffix for the annotations file.
            **CAUTION**: Make sure that the files are read by extract_frames in the correct order! 
            Pay attention to the output of this method.
        5. Re-train network with refined labels
            d.process()

        Repeat steps 4 and 5 until satisfied with the results.
    """
    def __init__(self, path, videos=[], name='test_01', experimenter=_config.EXPERIMENTER, annotation_suffix='', internal_to_dlc_labels: dict=None):
        """
        Initialize or load a DeepLabCut project.
        
        If a config.yaml exists at the path, loads the existing project.
        Otherwise, creates a new project with the provided videos.
        
        Args:
            path (str): Directory containing or for the project.
            videos (list): List of video file paths to include.
            name (str): Project name (must contain underscore for proper config handling).
            experimenter (str): Experimenter identifier.
            annotation_suffix (str): Suffix for annotation files (e.g., 'manual', 'refined').
            internal_to_dlc_labels (dict, optional): Custom label name mapping.
        
        Note:
            Videos are copied into the project folder by default.
            Project names without underscores may cause config issues with network paths.
        """
        if not HAS_DLC:
            raise RuntimeError('Install deeplabcut to use DLCProject functionality.')
        
        config_path = None
        if os.path.isfile(path):
            assert Path(path).stem == 'config' and Path(path).suffix == '.yaml'
            config_path = path
        if os.path.isdir(path):
            if os.path.exists(Path(path) / 'config.yaml'):
                config_path = Path(path) / 'config.yaml'
        self.path = path

        assert isinstance(annotation_suffix, str)
        self.annotation_suffix = annotation_suffix

        if isinstance(videos, str):
            videos = [videos]

        new_project = False
        if config_path is None:
            assert len(videos) > 0
            config_path = deeplabcut.create_new_project(name, experimenter, videos, working_directory=path, copy_videos=True)
            new_project = True
        
        self.config_path = config_path

        self.internal_to_dlc_labels = internal_to_dlc_labels

        if new_project:
            annotation_file_names = self.copy_annotations(videos)
            n_annotations_set = {len(VideoAnnotation(fname, vname).labels) for fname, vname in zip(annotation_file_names, videos)}
            assert len(n_annotations_set) == 1 # number of annotations in all the files should match
            annotation_names = [set(VideoAnnotation(fname, vname).labels) for fname, vname in zip(annotation_file_names, videos)]
            common_labels = functools.reduce(lambda x, y: x.intersection(y), annotation_names)
            all_labels = functools.reduce(lambda x, y: x.union(y), annotation_names)
            assert common_labels == all_labels
            annotation_names = sorted(list(common_labels))
            bodyparts = [f'point{x}' for x in annotation_names]
            self.edit_config(bodyparts=bodyparts, skeleton=None)
            self.edit_config(snapshotindex='all') # evaluate all snapshots
            if not os.path.exists(self.paths['models']):
                os.makedirs(self.paths['models'])

        # Re-anchor each video path so it shares config.yaml's root, regardless
        # of which NIC / drive letter / OS was used when the project was created.
        new_video_sets = {}
        for k, v in self.config["video_sets"].items():
            try:
                new_video_sets[rebase_to_config(self.config_path, k)] = v
            except ValueError as e:
                print(f"rebase_to_config: leaving path unchanged ({e})")
                new_video_sets[k] = v
        self.edit_config(video_sets=new_video_sets)

        try:
            deeplabcut.auxiliaryfunctions.read_config(self.config_path)
        except ScannerError:
            print("Config file is corrupted. Fix it manually.")
            print("If there is no _ in the name, then the config file has issues "
                  "when dealing with folders on the server.")

    @property
    def paths(self) -> Mapping[str, Path]:
        """
        Full paths to project folder and standard DLC subfolders.
        
        Returns:
            dict: Mapping of folder names to Path objects with keys:
                - 'project': Main project directory
                - 'models': Trained model weights (dlc-models or dlc-models-pytorch)
                - 'results': Evaluation results
                - 'labels': Labeled frame data
                - 'training_data': Training datasets
                - 'videos': Video files
        """
        project_path = Path(self.config_path).parent
        model_folder_name = 'dlc-models-pytorch' if DLC3 else 'dlc-models'
        evaluation_folder_name = 'evaluation-results-pytorch' if DLC3 else 'evaluation-results'
        return dict(
            project       = project_path,
            models        = project_path / model_folder_name,
            results       = project_path / evaluation_folder_name,
            labels        = project_path / 'labeled-data',
            training_data = project_path / 'training-datasets',
            videos        = project_path / 'videos',
        )
    
    @property
    def config(self) -> dict:
        """
        Current project configuration dictionary.
        
        Returns:
            dict: Parsed contents of config.yaml.
        """
        return deeplabcut.auxiliaryfunctions.read_config(self.config_path)
    
    @property
    def name(self) -> str:
        """Project name from configuration."""
        return self.config['Task']

    @property
    def trackers(self) -> list:
        """
        Names of tracked body parts as used internally by DLC.
        
        Returns:
            list: Body part names (e.g., ['point0', 'point1']).
        """
        return self.config['bodyparts']

    @property
    def label_names(self) -> list:
        """
        Human-readable names for tracked points.
        
        Returns meaningful names from dlc_trackermap.txt if available,
        otherwise returns the internal tracker names.
        
        Returns:
            list: Display names for body parts.
        """
        trackermap = self.trackermap
        return [trackermap[tracker] if tracker in trackermap else tracker for tracker in self.trackers]

    @property
    def trackermap(self):
        """
        Load meaningful label names from dlc_trackermap.txt.
        
        This file maps internal names (point0, point1) to biological names
        (nose, left_ear, etc.) for better interpretability.
        
        Returns:
            dict: Mapping from internal names to display names.
        
        Example dlc_trackermap.txt content:
            point0 - muscle_boundary
            point1 - fascia
            point2 - bone
        """
        map_file = Path(self.paths['project']) / 'dlc_trackermap.txt'
        # Path.read_text rather than builtin open() because the module
        # defines a top-level `open` (the workflow entry point) that
        # shadows builtins.open inside this module.
        if map_file.is_file():
            text = map_file.read_text(encoding='utf-8-sig')
            trackermap = [x.split(' - ') for x in text.splitlines() if x]
            return {x[0]: x[1] for x in trackermap}
        else:
            return {}
    
    def edit_config(self, config_file=None, **kwargs):
        """
        Modify project configuration parameters.
        
        Args:
            config_file (str, optional): Path to config file. Defaults to main config.
            **kwargs: Configuration parameters to update (e.g., iteration=2, snapshotindex=5).
        
        Returns:
            Result of deeplabcut.auxiliaryfunctions.edit_config().
        """
        if config_file is None:
            config_file = self.config_path
        assert os.path.exists(config_file)
        return deeplabcut.auxiliaryfunctions.edit_config(config_file, kwargs)

    @property
    def video_list(self) -> list[Path]:
        """Full paths to videos in the project."""
        return list(self.config['video_sets'].keys())
    
    @property
    def video_names(self) -> list[str]:
        """Video filenames without extensions."""
        return [Path(vname).stem for vname in self.video_list]
    
    @property
    def current_iteration(self) -> int:
        """Model iteration number currently set in config.yaml."""
        return self.config['iteration']
    
    @current_iteration.setter
    def current_iteration(self, iteration_num: int):
        """
        Set the active model iteration in config.yaml.
        
        Args:
            iteration_num: Iteration number, or 'latest' for most recent,
                or 'next' for latest+1 (if latest is trained).
        """
        if isinstance(iteration_num, str):
            assert iteration_num in ('latest', 'next')
            if iteration_num == 'latest':
                iteration_num = self.latest_iteration
            elif iteration_num == 'next':
                if self.latest_iteration_is_trained():
                    iteration_num = self.latest_iteration + 1
                else:
                    iteration_num = self.latest_iteration
        assert isinstance(iteration_num, int)
        self.edit_config(iteration=iteration_num)
    
    @property
    def latest_iteration(self) -> int:
        """Highest iteration number in dlc-models folder."""
        all_iterations = self.all_iterations
        if not all_iterations:
            return 0
        return self.all_iterations[-1]
    
    @property
    def latest_trained_iteration(self) -> int:
        """Most recent iteration that has saved model snapshots."""
        return max([iteration for iteration,snapshot in self.all_snapshots.items() if len(snapshot)], default=-1)
    
    @property
    def all_iterations(self) -> list:
        """All iteration numbers found in dlc-models, sorted ascending."""
        ret = [int(x.split('-')[-1]) for x in os.listdir(self.paths['models']) if x.startswith('iteration-') and os.path.isdir(self.paths['models'] / x)]
        ret.sort()
        return ret

    @property
    def all_snapshots(self) -> Mapping[int, list[int]]:
        """
        Training snapshots for each model iteration.
        
        Returns:
            dict: Maps iteration number to list of training iteration numbers.
                For DLC3, identifies .pt files; for DLC2, identifies .index files.
        """
        if DLC3:
            ext = ".pt"
        else:
            ext = ".index"
    
        ret = {}
        for iteration_num in self.all_iterations:
            source_path = self.paths['models'] / f'iteration-{iteration_num}'
            snapshot_filenames = pyfilemanager.FileManager(source_path).add()[f'*train/snapshot*{ext}']
            snapshot_numbers = [int(Path(x).stem.split('-')[-1]) for x in snapshot_filenames if "best" not in Path(x).stem]
            snapshot_numbers.sort()
            snapshot_numbers += [int(Path(x).stem.split('-')[-1]) for x in snapshot_filenames if "best" in Path(x).stem]
            ret[iteration_num] = snapshot_numbers
        return ret
    
    def current_iteration_is_trained(self) -> bool:
        """Check if current iteration has any saved snapshots."""
        return self.iteration_is_trained(self.current_iteration)
    
    def latest_iteration_is_trained(self) -> bool:
        """Check if latest iteration has any saved snapshots."""
        return self.iteration_is_trained(self.latest_iteration)

    def iteration_is_trained(self, iteration_num: int) -> bool:
        """
        Check if a specific iteration has been trained.
        
        Args:
            iteration_num (int): Model iteration to check.
        
        Returns:
            bool: True if snapshots exist for this iteration.
        """
        if iteration_num not in self.all_snapshots:
            return False
        return len(self.all_snapshots[iteration_num]) > 0
    
    def increment_iteration(self):
        """
        Advance to next iteration if current one is trained.
        
        Returns:
            self: For method chaining.
        """
        self.current_iteration = 'next'
        return self
        
    def add_videos(self, videos: list[Path]):
        """
        Add new videos to existing project and copy their annotations.
        
        Args:
            videos: List of video file paths to add.
        
        Returns:
            self: For method chaining.
        """
        if isinstance(videos, (str, Path)):
            videos = [videos]
        deeplabcut.add_new_videos(self.config_path, videos, copy_videos=True)
        self.copy_annotations(videos)
        return self
    
    def copy_annotations(self, video_name: Union[Path, list]):
        """
        Copy DUSTrack/_DUSTrackBase JSON files into project's video folder.
        
        Args:
            video_name: Single video path or list of video paths.
        
        Returns:
            str or list: Path(s) to copied annotation file(s), or None if not found.
        
        Note:
            Looks for files matching {video_stem}_annotations_{suffix}.json
        """
        if isinstance(video_name, list):
            copied_files = []
            for this_video_name in video_name:
                copied_file = self.copy_annotations(this_video_name)
                if copied_file is not None:
                    copied_files.append(copied_file)
            return copied_files
        v = Path(video_name)
        a_name = f'{v.stem}_annotations{"_" if self.annotation_suffix else ""}{self.annotation_suffix}.json'
        annotation_file_src = v.parent / a_name
        annotation_file_dest = Path(self.config_path).parent / 'videos' / a_name
        if os.path.exists(annotation_file_src):
            shutil.copyfile(annotation_file_src, annotation_file_dest)
            return annotation_file_dest
        return None

    def extract_frames(self, annotation_file_names=None, suffix_merged='merged', save_merged_json=False, check=False):
        """
        Extract labeled frames from videos and convert annotations to DLC format.
        
        This method:
        1. Finds all annotation JSON files for each video
        2. Merges multiple annotation files if present
        3. Extracts the annotated frames from videos
        4. Converts annotations to DLC's CSV/HDF5 format in labeled-data folder
        
        Args:
            annotation_file_names (list, optional): Specific annotation files to use.
                If None, automatically finds all matching files.
            suffix_merged (str): Suffix for merged annotation file. Defaults to 'merged'.
            save_merged_json (bool): Whether to save the merged JSON. Defaults to False.
            check (bool): Whether to run deeplabcut.check_labels(). Defaults to False.
        
        Returns:
            self: For method chaining.
        
        Note:
            Automatically excludes files with '_dlccorr' suffix (correction files).
        """
        annotation_file_names_input = annotation_file_names
        for video_file_name in self.video_list:
            coords = self.config["video_sets"][video_file_name]["crop"].split(",")
            video_stem = Path(video_file_name).stem
            output_path = self.paths['labels'] / video_stem

            if annotation_file_names_input is None:
                pattern = f'{video_stem}*_annotations*.json'
                fm = pyfilemanager.FileManager(self.paths['videos']).add()
                file_names = fnmatch.filter([Path(x).name for x in fm.all_files], pattern)
                annotation_file_names = sorted([fm[file_name][0] for file_name in file_names])
                # annotation_file_names = sorted(pyfilemanager.FileManager(self.paths['videos']).add()[f'{video_stem}*_annotations*.json'])
                # ignore the *correction* files. In theory, no training is to be done after the dlccorr files are created, but just being careful.
                annotation_file_names = [x for x in annotation_file_names if "_dlccorr" not in x]
                print(f'Loading annotations from {len(annotation_file_names)} file(s): ')
                print([Path(x).stem for x in annotation_file_names])
                print()
            
            if len(annotation_file_names) == 0:
                # there are multiple videos, but one of them does not have any labels
                continue
            
            ann = VideoAnnotation.from_multiple_files(
                fname_list = annotation_file_names,
                vname = video_file_name,
                name = suffix_merged,
                fname_merged = make_annotation_file_name(video_file_name, suffix_merged)
            )

            if save_merged_json:
                ann.save()
            _extract_frames_decord(video_file_name, ann.frames, output_path, coords)
            ann.to_dlc(
                scorer       = self.config['scorer'],
                output_path  = output_path,
                file_prefix  = f"CollectedData_{self.config['scorer']}",
                img_prefix   = 'img',
                img_suffix   = '.png',
                label_prefix = 'point',
                save         = True,
                internal_to_dlc_labels=self.internal_to_dlc_labels
                )
            
            if check:
                deeplabcut.check_labels(self.config_path) # this creates an _labeled folder, which doesn't seem necessary in this case
        
        return self
    
    def get_pose_cfg_file(self, iteration_num: int=None, type_: str='train') -> Path:
        """
        Get path to pose configuration file for an iteration.
        
        Args:
            iteration_num (int, optional): Iteration number. Defaults to current.
            type_ (str): 'train' or 'test' subfolder. Defaults to 'train'.
        
        Returns:
            Path: Full path to pose_cfg.yaml (DLC2) or pytorch_config (DLC3).
        """
        if iteration_num is None:
            iteration_num = self.current_iteration
        assert type_ in ('train', 'test')
        if DLC3:
            cfg_name = "pytorch_config"
        else:
            cfg_name = "pose_cfg"
        cfg_files = pyfilemanager.FileManager(self.paths['models'] / f'iteration-{iteration_num}').add()[f'*{type_}/{cfg_name}*']
        assert len(cfg_files) == 1
        return cfg_files[0]
    
    def get_best_snapshot(self, iteration_num: int=None) -> int:
        """
        Find training iteration with lowest test error.
        
        For DLC3, uses the snapshot marked as 'best' unless DLC3_USE_LAST_SNAPSHOT
        is True in config, in which case returns the last snapshot.
        For DLC2, parses CombinedEvaluation-results.csv.
        
        Args:
            iteration_num (int, optional): Model iteration. Defaults to current.
        
        Returns:
            int: Training iteration number of best snapshot.
        """
        if iteration_num is None:
            iteration_num = self.current_iteration
        
        if DLC3:
            source_path = self.paths['models'] / f'iteration-{iteration_num}'
            snapshot_filenames = pyfilemanager.FileManager(source_path).add()[f'*train/snapshot*.pt']
            snapshot_numbers = [int(Path(x).stem.split('-')[-1]) for x in snapshot_filenames]
            best_snapshot_number = [int(Path(x).stem.split('-')[-1]) for x in snapshot_filenames if "best" in Path(x).stem]
            if not _config.DLC3_USE_LAST_SNAPSHOT:
                if best_snapshot_number:
                    return best_snapshot_number[0]
            return sorted(snapshot_numbers)[-1]

        eval_file_name = self.paths['results'] / f'iteration-{iteration_num}' / 'CombinedEvaluation-results.csv'
        if os.path.exists(eval_file_name):
            # pick the snapshot with the lowest training error
            df_eval = pd.read_csv(eval_file_name)
            df_eval = df_eval.rename(columns=lambda x: x.strip())
            best_snapshot = df_eval[df_eval['Test error(px)'] == min(df_eval['Test error(px)'])]['Training iterations:'].iloc[0]
        else:
            # pick the latest snapshot
            print('Could not evaluate best snapshot, setting it to latest')
            best_snapshot = self.all_snapshots[iteration_num][-1]
        return best_snapshot
    
    def get_best_snapshot_test_error(self, iteration_num: int=None) -> float:
        """
        Get test error (RMSE in pixels) at best snapshot.
        
        Args:
            iteration_num (int, optional): Model iteration. Defaults to latest trained.
        
        Returns:
            float: Test error in pixels, or -1.0 if evaluation file doesn't exist.
        """
        if iteration_num is None:
            iteration_num = self.latest_trained_iteration
        eval_file_name = self.paths['results'] / f'iteration-{iteration_num}' / 'CombinedEvaluation-results.csv'
        column_name = 'test rmse_pcutoff' if DLC3 else 'Test error(px)'
        if os.path.exists(eval_file_name):
            # pick the snapshot with the lowest training error
            df_eval = pd.read_csv(eval_file_name)
            df_eval = df_eval.rename(columns=lambda x: x.strip())
            return float(min(df_eval[column_name]))
        return -1.
    
    def get_best_snapshot_idx(self, iteration_num: int=None) -> int:
        """
        Get snapshot index (not training iteration number) of best snapshot.
        
        Args:
            iteration_num (int, optional): Model iteration. Defaults to current.
        
        Returns:
            int: Index in the all_snapshots list for this iteration.
        """
        if iteration_num is None:
            iteration_num = self.current_iteration
        best_snapshot = self.get_best_snapshot(iteration_num)
        return self.all_snapshots[iteration_num].index(best_snapshot)

    def initialize_weights(self, source_iteration: int=None, source_snapshot: int=None, dest_iteration: int=None):
        """
        Initialize model weights from a previous iteration (transfer learning).
        
        Used when refining a model with additional labels. Edits the pose_cfg
        file to set init_weights parameter.
        
        Args:
            source_iteration (int, optional): Iteration to copy from. 
                Defaults to second-to-last iteration.
            source_snapshot (int, optional): Training iteration within source_iteration.
                Defaults to best snapshot.
            dest_iteration (int, optional): Iteration to initialize. 
                Defaults to latest iteration.
        
        Returns:
            self: For method chaining.
        
        Note:
            Does nothing if there's only one iteration (no source to copy from).
        """        
        all_iterations = self.all_iterations
        if source_iteration is None: # pick the last iteration
            if len(all_iterations) <= 1:
                return self
            source_iteration = all_iterations[-2]

        if source_snapshot is None:
            source_snapshot = self.get_best_snapshot(source_iteration)

        if dest_iteration is None:
            dest_iteration = all_iterations[-1]
        
        # find the correct pose_cfg file
        cfg_file = self.get_pose_cfg_file(dest_iteration)
        source_path = self.paths['models'] / f'iteration-{source_iteration}'
        ext = '.pt' if DLC3 else '.index'
        init_weights_files = pyfilemanager.FileManager(source_path).add()[f'*train/snapshot-*{source_snapshot}{ext}']
        assert len(init_weights_files) == 1

        if DLC3:
            self.edit_config(cfg_file, resume_training_from=init_weights_files[0].removesuffix('.index'))
        else:
            self.edit_config(cfg_file, init_weights=init_weights_files[0].removesuffix('.index'))
        return self

    def _initialize_weights_from_external_path(self, external_path):
        """Edit the latest iteration's pose_cfg to initialise weights from
        an external snapshot file.

        Sibling of :meth:`initialize_weights` for the
        ``train_iteration(refine_mode="external")`` UI path. Unlike
        :meth:`initialize_weights` (which resolves an in-project
        ``(source_iteration, source_snapshot)`` pair via ``FileManager``),
        this helper takes the path directly and writes it verbatim into
        the destination iteration's pose_cfg.

        The DLC3 ``train_iteration`` path normally bypasses this helper
        and passes ``snapshot_path=`` straight to ``train_network`` so
        pose_cfg stays clean. This helper is the DLC2 fallback (DLC2's
        ``train_network`` has no runtime override) and is also available
        on DLC3 if a caller wants the pose_cfg path explicitly.

        Args:
            external_path (str or Path): Path to an external snapshot
                file (``.pt`` on DLC3, ``.index`` on DLC2). A trailing
                ``.pt`` / ``.index`` extension is stripped to match the
                pose_cfg convention :meth:`initialize_weights` uses.

        Returns:
            self: For method chaining.
        """
        external_path = str(external_path)
        # Match initialize_weights' convention: pose_cfg expects the
        # extensionless prefix (DLC2: init_weights=.../snapshot-200;
        # DLC3: resume_training_from=.../snapshot-200).
        for ext in (".pt", ".index"):
            if external_path.endswith(ext):
                external_path = external_path[: -len(ext)]
                break

        dest_iteration = self.all_iterations[-1]
        cfg_file = self.get_pose_cfg_file(dest_iteration)
        if DLC3:
            self.edit_config(cfg_file, resume_training_from=external_path)
        else:
            self.edit_config(cfg_file, init_weights=external_path)
        return self

    def create_training_dataset(self, **kwargs):
        """Call deeplabcut.create_training_dataset."""
        net_type = kwargs.pop('net_type', 'resnet_50')
        deeplabcut.create_training_dataset(self.config_path, net_type=net_type, **kwargs)
        return self

    def train(self, **kwargs):
        """
        Train the neural network model.
        
        Sets custom learning rate schedule and trains with more iterations than
        DLC defaults for better convergence.
        
        Args:
            **kwargs: Passed to deeplabcut.train_network().
                - maxiters (int): Total training iterations. Default: 500000 (DLC2) or 1000 (DLC3 epochs).
                - max_snapshots_to_keep (int): Max saved checkpoints. Default: 20.
        
        Returns:
            self: For method chaining.
        
        Note:
            Custom learning rate schedule: [0.005@10k, 0.02@350k, 0.002@425k, 0.001@1M]
        """
        maxiters = kwargs.pop('maxiters', 500000)
        max_snapshots_to_keep = kwargs.pop('max_snapshots_to_keep', 20)
        cfg_file = self.get_pose_cfg_file()
        self.edit_config(cfg_file, multi_step = [[0.005, 10000], [0.02, 350000], [0.002, 425000], [0.001, 1000000]])
        deeplabcut.train_network(self.config_path, maxiters=maxiters, max_snapshots_to_keep=max_snapshots_to_keep, pytorch_cfg_updates={"runner.eval_interval": 25},**kwargs)
        return self
    
    def evaluate(self, **kwargs):
        """
        Evaluate all training snapshots on test set.
        
        Temporarily sets snapshotindex to 'all' to evaluate every checkpoint,
        then restores original value.
        
        Args:
            **kwargs: Passed to deeplabcut.evaluate_network().
        
        Returns:
            self: For method chaining.
        """
        current_snapshotindex_value = self.config['snapshotindex']
        self.edit_config(snapshotindex='all')
        deeplabcut.evaluate_network(self.config_path, **kwargs)
        self.edit_config(snapshotindex=current_snapshotindex_value)
        return self

    def analyze_videos(self, iteration_num=None, snapshotindex=None, create_video=True, **kwargs):
        """
        Run inference on videos and optionally create labeled output videos.
        
        Args:
            iteration_num (int, optional): Model iteration to use. Defaults to current.
            snapshotindex (int, optional): Snapshot index to use. 
                Defaults to best snapshot. Negative indices supported.
            create_video (bool): Whether to create labeled video. Defaults to True.
            **kwargs: Additional arguments for deeplabcut.analyze_videos().
                - videos: List of video paths or indices. If not provided, analyzes all videos.
        
        Returns:
            self: For method chaining.
        
        Note:
            Results saved to videos/iteration-{N}/ subfolder.
            If videos kwarg contains integers, they're treated as indices into self.video_list.
        """
        if iteration_num is None:
            iteration_num = self.current_iteration
        
        if snapshotindex is None:
            snapshotindex = self.get_best_snapshot_idx(iteration_num)
        else:
            n_snapshots = len(self.all_snapshots[iteration_num])
            if snapshotindex < 0:
                snapshotindex = snapshotindex % n_snapshots
            assert 0 <= snapshotindex < n_snapshots
        
        save_as_csv = kwargs.pop('save_as_csv', True)

        # DeepLabCut's PyTorch backend defaults to batch_size=1 when neither
        # the kwarg nor the project config sets one, which leaves an RTX-class
        # GPU heavily under-utilised. The throughput knee for ResNet-50 BU on
        # a 706x558 video on DLC 3.0.0rc14 + RTX 4090 is batchsize~4 (median
        # 154 fps; see S:/_corpus/dustrack/dlc_inference_bench_2026-05-20/).
        # Respect the project config if it sets ``batch_size`` explicitly.
        if 'batchsize' not in kwargs and self.config.get('batch_size') is None:
            kwargs['batchsize'] = 4

        if "videos" in kwargs:
            assert isinstance(kwargs["videos"], list)
            # if kwargs["videos"] is a list of integers, convert to list of video paths using self.video_list
            if all(isinstance(v, int) for v in kwargs["videos"]):
                video_indices = kwargs["videos"]
                video_list = []
                for idx in video_indices:
                    if idx < 0:
                        idx = len(self.video_list) + idx
                    assert 0 <= idx < len(self.video_list), f"Video index {idx} is out of range."
                    video_name = self.video_list[idx]
                    assert os.path.exists(video_name), f"Video {video_name} does not exist."
                    video_list.append(video_name)
                kwargs["videos"] = video_list
        else:
            kwargs["videos"] = self.video_list
        
        current_snapshotindex_value = self.config['snapshotindex']
        self.edit_config(snapshotindex=snapshotindex)

        common_params = dict(
            config     = self.config_path, 
            videos     = kwargs.pop('videos'), 
            destfolder = self.paths['videos'] / f'iteration-{iteration_num}'
            )

        deeplabcut.analyze_videos(**common_params, save_as_csv=save_as_csv, **kwargs)
        if create_video:
            deeplabcut.create_labeled_video(**common_params)
        
        self.edit_config(snapshotindex=current_snapshotindex_value)
        return self
    # refine can be both bool or string, if string, it is the path of the model to initialize weights from
    def process(self, iteration_num=None, maxiters=None, refine: Union[bool, str]=True, create_video=True, source_snapshot=None, **kwargs):
        """
        Automated workflow: extract frames, train, evaluate, and analyze.
        
        This is the main method for handling the full DLC pipeline. It intelligently
        decides what steps to run based on the current project state:
        - If iteration already evaluated: just analyze videos
        - If frames need extraction: extract them
        - If not trained: train the model
        - If refining: initialize weights from previous iteration
        
        Args:
            iteration_num (int or str, optional): Iteration to process. 
                Can be integer or 'latest'. Defaults to 'latest'.
            maxiters (int, optional): Training iterations. 
                Defaults: 500000 (DLC2) or 1000 epochs (DLC3).
            refine (bool): Use transfer learning from previous iteration. Defaults to True.
            create_video (bool): Create labeled output video. Defaults to True.
            source_snapshot (int, optional): Specific snapshot for weight initialization.
            **kwargs: Additional arguments.
                - videos: List of videos to analyze (can be indices or paths).
        
        Returns:
            self: For method chaining.
        
        Example:
            >>> proj = DLCProject('path/to/project')
            >>> proj.process()  # Full automated workflow
        """
        if iteration_num is None:
            iteration_num = 'latest'
        else:
            assert isinstance(iteration_num, int)

        if maxiters is None:
            if DLC3:
                # TEMPORARY: dropped from 1000 → 50 to speed up the
                # datanavigator/DUSTrack test-bed iteration loop
                # (S:\_corpus\dustrack\). REVERT to 1000 before 1.1.0rc2
                # ships.
                maxiters = 50 # epochs
            else:
                maxiters = 500000

        self.current_iteration = iteration_num

        current_iteration = self.current_iteration
        latest_iteration = self.latest_iteration
        if current_iteration < latest_iteration:
            return self.evaluate().analyze_videos(create_video=create_video)

        self.extract_frames() # do this every time in case there are any updates to the manual annotations.
        
        if self.latest_iteration_is_trained():
            self.increment_iteration() # increment iteration in the config.yaml file
        
        if not os.path.exists(self.paths['training_data'] / f'iteration-{self.current_iteration}'):
            self.create_training_dataset()
        
        if isinstance(refine, bool) and refine:
            if not self.latest_iteration_is_trained() and self.current_iteration == self.latest_iteration:
                if source_snapshot is not None:
                    source_iteration = self.latest_iteration - int(not self.latest_iteration_is_trained())
                else:
                    source_iteration = None
                self.initialize_weights(source_iteration=source_iteration, source_snapshot=source_snapshot)

        if not self.current_iteration_is_trained():
            try:
                if DLC3:
                    if isinstance(refine, str):
                        self.train(epochs=maxiters, snapshot_path=refine)
                    else:
                        self.train(epochs=maxiters)
                else:
                    self.train(maxiters=maxiters)
            except KeyboardInterrupt:
                pass

        analyze_videos_kwargs = {}
        if "videos" in kwargs:
            analyze_videos_kwargs["videos"] = kwargs.pop("videos")
        if "analyze_batchsize" in kwargs:
            analyze_videos_kwargs["batchsize"] = kwargs.pop("analyze_batchsize")

        return self.evaluate().analyze_videos(create_video=create_video, **analyze_videos_kwargs)

    def train_iteration(
        self,
        *,
        refine_mode: Literal["scratch", "in_project", "external"] = "scratch",
        source_iteration: int = None,
        source_snapshot: int = None,
        external_snapshot_path: str = None,
        maxiters: int = None,
        create_video: bool = False,
        videos: list = None,
        analyze_batchsize: int = None,
    ):
        """Explicit-args training driver for UI-triggered flows.

        Distinct from :meth:`process` (auto-infer for CLI ergonomics).
        Caller decides everything: refine source, training duration,
        output options. Strict validation per ``refine_mode``; no
        inference and no silent fallbacks.

        The mechanics of advancing iterations (extract_frames →
        increment_iteration if latest is trained → create_training_dataset
        if needed) mirror :meth:`process`; the only difference is *how*
        weights are initialised once the destination iteration is in
        place.

        Args:
            refine_mode: How to initialise weights for the next training
                round. ``"scratch"`` starts from random init (no pose_cfg
                edit); ``"in_project"`` copies weights from a snapshot
                in this project (requires ``source_iteration``, optional
                ``source_snapshot``); ``"external"`` initialises from an
                arbitrary snapshot file (requires
                ``external_snapshot_path``; supported on both DLC2 and
                DLC3 -- DLC3 passes the path through
                ``train_network(snapshot_path=...)``, DLC2 edits
                pose_cfg's ``init_weights`` via
                :meth:`_initialize_weights_from_external_path`).
            source_iteration: in-project iteration to copy weights from.
                Only valid with ``refine_mode="in_project"``; must point
                at a trained iteration.
            source_snapshot: specific snapshot within
                ``source_iteration``. Only valid with
                ``refine_mode="in_project"``; defaults to the best
                snapshot when ``None``.
            external_snapshot_path: path to an external snapshot file
                (``.pt`` on DLC3, ``.index`` on DLC2). Only valid with
                ``refine_mode="external"``; the file must exist at
                call time.
            maxiters: training epochs (DLC3) or iterations (DLC2).
                Defaults to the same values :meth:`process` uses (50 /
                500000) so the two methods stay consistent until the
                UI exposes the field.
            create_video: write a labeled output video after analyze.
                Defaults to ``False`` (the UI ergonomics default;
                :meth:`process` defaults to ``True`` for CLI parity).
            videos: list of videos (indices or paths) to analyze.
                Forwarded to ``analyze_videos``. ``None`` analyses every
                video in the project.
            analyze_batchsize: batchsize for ``analyze_videos``.
                Forwarded on if set; ``None`` lets ``analyze_videos``
                pick its own default (post-2026-05-20: rc14 knee at 4).

        Returns:
            self: For method chaining.

        Raises:
            ValueError: on refine_mode / argument mismatch (see
                :meth:`_validate_train_iteration_args`).
            TypeError: if ``source_iteration`` / ``source_snapshot``
                aren't ``int`` when given.
            FileNotFoundError: if ``external_snapshot_path`` is set but
                the file doesn't exist.
        """
        self._validate_train_iteration_args(
            refine_mode=refine_mode,
            source_iteration=source_iteration,
            source_snapshot=source_snapshot,
            external_snapshot_path=external_snapshot_path,
        )

        if maxiters is None:
            maxiters = 50 if DLC3 else 500000  # same defaults as process()

        # Iteration advancement mechanics (mirror process()).
        self.extract_frames()  # capture any new manual annotations
        if self.latest_iteration_is_trained():
            self.increment_iteration()
        if not os.path.exists(self.paths['training_data'] / f'iteration-{self.current_iteration}'):
            self.create_training_dataset()

        # Apply refine mode.
        if refine_mode == "in_project":
            self.initialize_weights(
                source_iteration=source_iteration,
                source_snapshot=source_snapshot,
            )
        elif refine_mode == "external" and not DLC3:
            # DLC2: no runtime override -- edit pose_cfg's init_weights
            # to point at the external snapshot. DLC3 handles this
            # inline at the train call below via snapshot_path=.
            self._initialize_weights_from_external_path(external_snapshot_path)
        # refine_mode == "scratch": no pose_cfg edit; pose_cfg is fresh
        # from create_training_dataset and has no init weights set.

        # Train.
        if not self.current_iteration_is_trained():
            train_kwargs = {}
            if DLC3:
                train_kwargs["epochs"] = maxiters
                if refine_mode == "external":
                    train_kwargs["snapshot_path"] = external_snapshot_path
            else:
                train_kwargs["maxiters"] = maxiters
            try:
                self.train(**train_kwargs)
            except KeyboardInterrupt:
                pass

        # Evaluate + analyze.
        analyze_kwargs = {}
        if videos is not None:
            analyze_kwargs["videos"] = videos
        if analyze_batchsize is not None:
            analyze_kwargs["batchsize"] = analyze_batchsize
        return self.evaluate().analyze_videos(create_video=create_video, **analyze_kwargs)

    def _validate_train_iteration_args(
        self,
        *,
        refine_mode,
        source_iteration,
        source_snapshot,
        external_snapshot_path,
    ):
        """Strict validation for :meth:`train_iteration`. Raises on
        mismatch.

        The discriminator is ``refine_mode``; the helper enforces a
        canonical valid-combo table:

        - ``"scratch"``: every source / external arg must be ``None``.
        - ``"in_project"``: ``source_iteration`` required (``int``,
          trained); ``source_snapshot`` optional (``int`` or ``None``
          -- ``None`` lets :meth:`initialize_weights` pick the best
          snapshot); ``external_snapshot_path`` must be ``None``.
        - ``"external"``: ``external_snapshot_path`` required (string,
          file must exist); ``source_iteration`` /
          ``source_snapshot`` must be ``None``.
        """
        valid_modes = {"scratch", "in_project", "external"}
        if refine_mode not in valid_modes:
            raise ValueError(
                f"refine_mode must be one of {sorted(valid_modes)}, "
                f"got {refine_mode!r}"
            )

        if refine_mode == "scratch":
            for name, value in (
                ("source_iteration", source_iteration),
                ("source_snapshot", source_snapshot),
                ("external_snapshot_path", external_snapshot_path),
            ):
                if value is not None:
                    raise ValueError(
                        f"refine_mode='scratch' is incompatible with "
                        f"{name}={value!r}"
                    )
            return

        if refine_mode == "in_project":
            if external_snapshot_path is not None:
                raise ValueError(
                    "refine_mode='in_project' is incompatible with "
                    f"external_snapshot_path={external_snapshot_path!r}"
                )
            if source_iteration is None:
                raise ValueError(
                    "refine_mode='in_project' requires source_iteration"
                )
            if not isinstance(source_iteration, int):
                raise TypeError(
                    f"source_iteration must be int, got "
                    f"{type(source_iteration).__name__}"
                )
            if not self.iteration_is_trained(source_iteration):
                trained = [i for i, snaps in self.all_snapshots.items() if snaps]
                raise ValueError(
                    f"source_iteration={source_iteration} is not a "
                    f"trained iteration. Trained iterations: {trained}"
                )
            if source_snapshot is not None and not isinstance(source_snapshot, int):
                raise TypeError(
                    f"source_snapshot must be int or None, got "
                    f"{type(source_snapshot).__name__}"
                )
            return

        if refine_mode == "external":
            for name, value in (
                ("source_iteration", source_iteration),
                ("source_snapshot", source_snapshot),
            ):
                if value is not None:
                    raise ValueError(
                        f"refine_mode='external' is incompatible with "
                        f"{name}={value!r}"
                    )
            if external_snapshot_path is None:
                raise ValueError(
                    "refine_mode='external' requires external_snapshot_path"
                )
            if not os.path.exists(external_snapshot_path):
                raise FileNotFoundError(
                    f"External snapshot not found: {external_snapshot_path}"
                )
            return

    def annotate(self, video_index: int=0, new_annotation_suffix=None, **dustrack_kwargs):
        """
        Launch interactive annotation GUI for a video.

        Opens DUSTrack interface with existing annotation layers loaded,
        including any DLC predictions as line plot overlays.

        Args:
            video_index (int): Index of video in video_list. Defaults to 0.
                Negative indices supported.
            new_annotation_suffix (str, optional): Suffix for new annotation layer.
                Defaults to 'iteration-{N}' where N is the next iteration number.
            **dustrack_kwargs: Forwarded to the DUSTrack constructor. Notable
                pass-through options: ``fast_render=True`` (datanavigator
                1.5.0+ Tier 2 Qt-native video pane, ~3x speedup on the
                interosseous_pn24-x benchmark), ``dark_mode=True``,
                ``clahe_clip``, ``clahe_grid``, ``gamma``, ``brightness``.

        Returns:
            DUSTrack: Interactive annotation interface.

        Note:
            Creates a 'buffer' layer for temporary annotations.
            Latest DLC predictions are automatically set as overlay.
        """
        if video_index < 0:
            video_index = len(self.video_list) + video_index
        assert 0 <= video_index < len(self.video_list)

        if new_annotation_suffix is None:
            if self.latest_iteration_is_trained():
                new_iteration_num = self.latest_iteration + 1
            else:
                new_iteration_num = self.latest_iteration
            new_annotation_suffix = f'iteration-{new_iteration_num}'

        fm_annotations = VideoFileManager(self, video_index)
        annotation_names = fm_annotations.get_all_annotation_layers(new_annotation_suffix)
        annotation_names['buffer'] = fm_annotations.get_new_json('buffer')
        # fast_render default is set by DUSTrack.__init__; no need to
        # duplicate here. Callers can pass ``fast_render=False`` via
        # ``dustrack_kwargs`` to opt out.
        ret = DUSTrack(self.video_list[video_index], annotation_names, height_ratios=(3,1,1), **dustrack_kwargs)
        # Wire the DUSTrack back to this project so the Train / Reduce
        # jitter buttons (and `_refresh_dlc_layers`) work on a
        # re-entered session — without this the GUI's `_dlcproject`
        # stays at its `__init__` default of None and "Train DLC model"
        # raises "DLCProject not created."
        ret._dlcproject = self
        # Single helper drives the post-load display state for both
        # the fresh-construction path (this method) and the in-place
        # refresh path (DUSTrack._refresh_dlc_layers).
        ret._normalize_dlc_layer_display()
        # Fold dlccorr (saved as ``*_annotations_dlccorr.json``, so it
        # rides the manuals block out of get_all_annotation_layers) into
        # its own group at the tail of the DLC chain. Without this, a
        # fresh open of a re-entered project would show dlccorr mixed
        # with manuals while a post-train refresh would show it grouped
        # with the DLC chain -- _restructure_annotation_order keeps the
        # two paths in lockstep.
        ret._restructure_annotation_order()
        # ``_dlcproject`` was None when DUSTrack.__init__ ran its initial
        # gate evaluation; re-run now that it's set so "Train DLC model"
        # enables and "Create DLC Project" disables.
        ret._refresh_workflow_button_state()
        ret.update()

        return ret

    def get_trajectories(self, videos=None, iteration=None):
        """
        Load tracking results as DLCData objects.
        
        Args:
            videos (list or str, optional): Videos to load. Defaults to all videos.
            iteration (int, optional): Model iteration. Defaults to current.
        
        Returns:
            dict: Maps video stem to DLCData object.
        
        Raises:
            ValueError: If a requested video is not in the project.
        """
        if iteration is None:
            iteration = self.current_iteration
        if videos is None:
            videos = self.video_list
        elif isinstance(videos, str):
            videos = [videos]

        data = {}
        for video in videos:
            if video not in self.video_list:
                raise ValueError(f"{video} does not exist in this project. It cannot be loaded.")
            data[Path(video).stem] = DLCData.from_video(video) ### Need to find a way to relate training iterations (gradient descent) with training iterations (number of times retrained)
        return data
    
    def open(self):
        """Open project folder in Windows Explorer."""
        os.system(f'explorer.exe "{str(Path(self.config_path).parent)}"')


def _is_dlc_project_root(folder) -> bool:
    """Cheap structural check for a DLC project folder.

    DLC's ``create_new_project`` always lays down ``config.yaml`` next to
    ``videos/`` and ``labeled-data/``; requiring all three avoids matching
    a stray ``config.yaml`` that belongs to something else. No YAML
    parsing -- pure filesystem.
    """
    f = Path(folder)
    return (
        (f / 'config.yaml').is_file()
        and (f / 'videos').is_dir()
        and (f / 'labeled-data').is_dir()
    )


def _find_dlc_config(path):
    """Resolve ``path`` to the DLC ``config.yaml`` that contains it, or None.

    Resolves four input shapes:

    - ``config.yaml`` file -> that path (only if the sibling project structure exists)
    - DLC project folder -> ``folder / 'config.yaml'``
    - Any file inside a project (notably a video under ``videos/``) -> walks up
      ancestors until a DLC-root is found
    - Anything else (a bare video outside any project, a non-existent path) -> None

    Returning None signals Phase 1 to :func:`open`. Note the walk-up stops
    at the filesystem root; in practice DLC's layout means it terminates
    after one step.
    """
    p = Path(path)
    if not p.exists():
        return None

    if p.is_file() and p.name.lower() == 'config.yaml':
        return p if _is_dlc_project_root(p.parent) else None

    if p.is_dir() and _is_dlc_project_root(p):
        return p / 'config.yaml'

    if p.is_file():
        for ancestor in p.parents:
            if _is_dlc_project_root(ancestor):
                return ancestor / 'config.yaml'

    return None


def _find_video_index(project, video_path):
    """Look up a video's index in ``project.video_list`` by filename stem.

    Stem matching (rather than full-path equality) is robust to the
    drive-letter / UNC / posix shuffling that :func:`rebase_to_config`
    already handles inside ``DLCProject``. Returns None if the video
    isn't part of the project.
    """
    target_stem = Path(video_path).stem
    for i, name in enumerate(project.video_names):
        if name == target_stem:
            return i
    return None


def _session_inside_dlc_project(dustrack) -> Optional[Path]:
    """Return the DLC project root the session sits inside, or None.

    Reuses :func:`_find_dlc_config` for the filesystem walk-up so the
    structural check (``config.yaml + videos/ + labeled-data/``) stays
    in one place. ``self._dlcproject`` is checked first as the cheap
    short-circuit: a session that was opened via ``dustrack.open(<project>)``
    or that survived a successful ``create_dlc_project`` already knows
    its project; we only fall back to walking up ``self.fname``'s
    ancestors when the attribute is unset (e.g. a video opened bare
    that happens to live inside an existing project tree).
    """
    proj = getattr(dustrack, "_dlcproject", None)
    if proj is not None:
        config_path = getattr(proj, "config_path", None)
        if config_path is not None:
            return Path(config_path).parent
    fname = getattr(dustrack, "fname", None)
    if fname is None:
        return None
    config = _find_dlc_config(fname)
    return config.parent if config is not None else None


# Extensions surfaced as the "Videos" filter group in the no-arg
# :func:`open` picker. Anything Qt-renderable beyond this list is still
# reachable via the "All files" fallback. Kept conservative so users
# don't accidentally pick an audio/image asset and crash on construction.
_VIDEO_PICKER_EXTENSIONS = ("mp4", "avi", "mov", "mkv", "mts", "m4v", "wmv", "webm")


def _prompt_for_videos(parent=None):
    """Pop a Qt file-picker for one-or-more video files.

    Bootstraps a ``QApplication`` if one isn't already running (same
    pattern as :func:`_pin_qt_palette`), so this works as the very first
    Qt call in a fresh process. Returns:

    - ``list[Path]`` -- user picked one or more files (order preserved
      as the user clicked them).
    - ``None`` -- user cancelled, OR qtpy isn't importable in the env
      (mpl-only install path); caller falls back to a no-op.

    Files-only by design for 1.2.0a3. Folder-picker support (pick a
    directory, recurse for videos) is on the 1.2.0 roadmap and arrives
    alongside the multi-video swap-state contract.
    """
    try:
        from qtpy.QtWidgets import QApplication, QFileDialog
    except ImportError:
        return None
    _ = QApplication.instance() or QApplication([])
    exts = " ".join(f"*.{e}" for e in _VIDEO_PICKER_EXTENSIONS)
    file_filter = f"Videos ({exts});;All files (*.*)"
    paths, _selected_filter = QFileDialog.getOpenFileNames(
        parent,
        "Open video(s) for DUSTrack",
        "",
        file_filter,
    )
    if not paths:
        return None
    return [Path(p) for p in paths]


def open(path=None, layer_name=None, **dustrack_kwargs):
    """Open a DUSTrack annotation session; auto-resolves Phase 1 vs Phase 2 from ``path``.

    The unified entry point for the DUSTrack workflow. Users hand it a
    path and DUSTrack figures out whether they're starting fresh on a
    standalone video or resuming inside a DLC project.

    **Phase 1 -- bare video, no DLC project context.**
        Equivalent to ``DUSTrack(path, layer_name, **kwargs)``. Works
        without ``deeplabcut`` installed -- the GUI plus the LK-RSTC
        post-processing run standalone, which is the "Option 1"
        install path from the paper.

    **Phase 2 -- DLC project context.**
        Accepts a video inside a project's ``videos/`` folder, a
        ``config.yaml``, or a project folder. Resolves the
        :class:`DLCProject` and dispatches to :meth:`DLCProject.annotate`
        so a fresh DUSTrack opens with all existing annotation layers,
        DLC trace overlays, and a new iteration layer wired up.

    The two-phase split mirrors DUSTrack's deliberate copy-on-project-
    creation design: once a DLC project exists, the project folder is
    the workspace and the original video becomes a frozen "rewind point"
    (delete the folder to start over). ``open()`` honors that boundary
    -- pointing at the original video gives you Phase 1, pointing at
    the in-project copy gives you Phase 2.

    Args:
        path: Video file, ``config.yaml``, DLC project folder, a
            sequence of any of these (first dispatches, the rest land
            on ``tracker._video_queue`` for future multi-video
            navigation), or ``None`` -- ``None`` pops a Qt file picker
            and lets the user pick one or more videos.
        layer_name: Annotation layer name. Optional in both phases:
            Phase 1 defaults to ``'iteration-0'`` (the canonical seed
            name for the rest of the DLC pipeline -- the next DLC
            training iteration lands as ``iteration-1``); Phase 2
            defaults to ``iteration-{N+1}`` (the next-iteration suffix
            derived from the project's training history). Callers can
            still pass an explicit name to override.
        **dustrack_kwargs: Forwarded to the underlying :class:`DUSTrack`
            constructor (``dark_mode``, ``fast_render``, ``clahe_clip``,
            ``gamma``, ``brightness``, etc.).

    Returns:
        DUSTrack: Live annotation UI, ready to use. ``None`` if the
        no-arg form's file picker was cancelled.

    Raises:
        FileNotFoundError: If ``path`` doesn't exist.
        ValueError: Path is a directory that isn't a DLC project, or
            an empty sequence was supplied.
        ImportError: Phase 2 entry on a system without ``deeplabcut``
            installed.

    Examples:
        Zero-argument launch (pops a video picker)::

            import dustrack
            tracker = dustrack.open()

        Fresh annotation (default layer name ``'iteration-0'``)::

            tracker = dustrack.open('video.mp4')

        Multi-video launch (first opens; rest stash on
        ``tracker._video_queue`` until the navigation UI lands)::

            tracker = dustrack.open(['v0.mp4', 'v1.mp4', 'v2.mp4'])

        Resume after closing the UI mid-workflow (any of these work)::

            tracker = dustrack.open('S:/path/to/project/videos/video.mp4')
            tracker = dustrack.open('S:/path/to/project/config.yaml')
            tracker = dustrack.open('S:/path/to/project/')

        With UI options::

            tracker = dustrack.open('video.mp4', 'manual', dark_mode=True)
    """
    if path is None:
        picked = _prompt_for_videos()
        if picked is None:
            return None
        path = picked

    queued: list[Path] = []
    if isinstance(path, (list, tuple)):
        if len(path) == 0:
            raise ValueError("dustrack.open: empty path sequence")
        # First entry dispatches; the rest ride along as a queue for
        # the future multi-video swap-state contract (Roadmap *Next
        # 1.2.0* item 3). No nav UI consumes ``_video_queue`` yet.
        queued = [Path(q) for q in path[1:]]
        path = path[0]

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"dustrack.open: path does not exist: {path}")

    config_path = _find_dlc_config(p)

    if config_path is None:
        # Phase 1: no DLC project context. ``layer_name`` defaults to
        # ``iteration-0`` so a bare-video session seeds the canonical
        # DLC iteration-N naming -- the next DLC training round lands
        # as ``iteration-1`` rather than colliding with whatever ad-hoc
        # name the user picked.
        if not p.is_file():
            raise ValueError(
                f"dustrack.open: {path!s} is a directory but doesn't look like "
                "a DLC project (no config.yaml + videos/ + labeled-data/). "
                "Pass a video file or a DLC project folder."
            )
        if layer_name is None:
            layer_name = "iteration-0"
        tracker = DUSTrack(str(p), layer_name, **dustrack_kwargs)
    else:
        # Phase 2: project found.
        if not HAS_DLC:
            raise ImportError(
                f"dustrack.open: detected a DLC project at {config_path.parent}, "
                "but deeplabcut is not installed. Install deeplabcut to resume "
                "the project, or point at a video outside the project to use "
                "DUSTrack standalone."
            )
        project = DLCProject(str(config_path))

        # If the caller pointed at a specific video inside the project,
        # respect that; otherwise default to the first video (matches
        # DLCProject.annotate's own default).
        video_index = 0
        if p.is_file():
            match = _find_video_index(project, p)
            if match is not None:
                video_index = match

        tracker = project.annotate(
            video_index=video_index,
            new_annotation_suffix=layer_name,
            **dustrack_kwargs,
        )

    # Stash any list-form leftovers for the future multi-video nav
    # (Roadmap *Next 1.2.0* item 3). Always set the attribute (even on
    # the empty case) so consumers don't need a ``getattr`` dance.
    tracker._video_queue = queued
    return tracker


def _extract_frames(video_file_name: str, frame_idx: list, output_path: str, coords: list):
    """
    Legacy frame extraction using DLC's VideoWriter (OpenCV-based).
    
    Note:
        This function is kept for backwards compatibility but
        _extract_frames_decord is now used by default for better
        performance, and because of discrepancy in extracted frames (seeking
        issues) when using OpenCV vs decord.
    
    Args:
        video_file_name (str): Path to video file.
        frame_idx (list): Frame numbers to extract (0-indexed).
        output_path (str): Directory to save extracted frames.
        coords (list): Crop coordinates [x, y, width, height].
    
    Returns:
        list: Paths to saved image files.
    """
    cap = VideoWriter(video_file_name)
    cap.set_bbox(*map(int, coords))
    indexlength = int(np.ceil(np.log10(len(cap))))
    output_path.mkdir(parents=True, exist_ok=True)
    img_names = []
    for index in frame_idx:
        cap.set_to_frame(index)  # extract a particular frame
        frame = cap.read_frame(crop=True)
        if frame is not None:
            img_name = output_path / f'img{str(index).zfill(indexlength)}.png'
            if not os.path.exists(img_name):
                image = img_as_ubyte(frame)
                io.imsave(img_name, image)
                print(f'{img_name.parent.stem}/{img_name.stem} saved!')
            else:
                print(f'{img_name.parent.stem}/{img_name.stem} already exists. Skipping extraction.')
            img_names.append(img_name)
        else:
            print("Frame", index, " not found!")
    cap.close()
    return img_names

def _extract_frames_decord(video_file_name: str, frame_idx: list, output_path: str, coords: list):
    """
    Extract video frames using Decord library for better performance.
    
    This is the default frame extraction method. It uses batch reading for
    better I/O efficiency compared to OpenCV sequential reading.
    
    Args:
        video_file_name (str): Path to video file.
        frame_idx (list): Frame numbers to extract (0-indexed).
        output_path (str): Directory to save extracted frames.
        coords (list): Crop coordinates. Interpreted as:
            - [x1, y1, x2, y2] if values look like absolute corners
            - [x, y, width, height] otherwise
    
    Returns:
        list: Paths to saved image files.
    
    Note:
        Skips extraction if image file already exists.
        Handles invalid frame indices gracefully.
    """
    # No need to set a bridge; default 'native' is fine and we use .asnumpy().
    # Force pix_fmt='rgb24' so DLC's labeled-data folder gets 3-channel PNGs
    # even when the source is monochrome-encoded (dnav 1.5.0a2 would
    # otherwise auto-detect gray and write 1-channel PNGs that DLC's
    # ResNet-50 backbone can't ingest).
    vr = VideoReader(video_file_name, ctx=cpu(0), num_threads=1, pix_fmt='rgb24')  # HWC RGB uint8
    n_frames = len(vr)
    indexlength = max(1, int(np.ceil(np.log10(max(1, n_frames)))))

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    def _crop(img, coords):
        if not coords:
            return img
        x0, y0, c2, c3 = map(int, coords)
        h, w = img.shape[:2]

        # Interpret coords as [x1,y1,x2,y2] if c2/c3 look like absolute corners; else [x,y,w,h]
        if (c2 > x0) and (c3 > y0) and (c2 <= w) and (c3 <= h):
            x1, y1, x2, y2 = x0, y0, c2, c3
        else:
            x1, y1 = x0, y0
            x2, y2 = x0 + c2, y0 + c3

        # Clamp to bounds
        x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return img
        return img[y1:y2, x1:x2]

    # Keep only valid indices (and preserve order)
    valid_indices = [int(i) for i in frame_idx if isinstance(i, (int, np.integer)) and 0 <= int(i) < n_frames]

    img_names = []
    if not valid_indices:
        for idx in frame_idx:
            print(f"Frame {idx} not found!")
        return img_names

    # Batch fetch for consistent seeking & speed
    batch = vr.get_batch(valid_indices).asnumpy()  # (N, H, W, 3) RGB uint8
    for k, idx in enumerate(valid_indices):
        frame = batch[k]
        if coords:
            frame = _crop(frame, coords)

        image = frame if frame.dtype == np.uint8 else img_as_ubyte(frame)
        img_name = output_path / f'img{str(idx).zfill(indexlength)}.png'
        if not os.path.exists(img_name):
            io.imsave(str(img_name), image)
            print(f'{img_name.parent.stem}/{img_name.stem} saved!')
        else:
            print(f'{img_name.parent.stem}/{img_name.stem} already exists. Skipping extraction.')
        img_names.append(img_name)

    return img_names



def get_annotation_file_name(video_file_name: Path, annotation_suffix: str='') -> Union[str, None]:
    """
    Get full path to annotation file if it exists.
    
    Args:
        video_file_name (Path): Video file path.
        annotation_suffix (str): Annotation suffix (e.g., 'manual', 'refined').
    
    Returns:
        str or None: Full path if file exists, None otherwise.
    """
    annotation_file_name = make_annotation_file_name(video_file_name, annotation_suffix)
    if os.path.exists(annotation_file_name):
        return annotation_file_name
    return None

def make_annotation_file_name(video_file_name: Path, annotation_suffix: str='') -> str:
    """
    Construct annotation filename from video filename and suffix.
    
    Args:
        video_file_name (Path): Video file path.
        annotation_suffix (str): Annotation suffix. Empty string means no suffix.
    
    Returns:
        str: Full path to annotation file (may not exist yet).
    
    Example:
        >>> make_annotation_file_name('video.mp4', 'manual')
        'video_annotations_manual.json'
        >>> make_annotation_file_name('video.mp4', '')
        'video_annotations.json'
    """
    v = Path(video_file_name)
    annotation_file_name = v.parent / f'{v.stem}_annotations{"_" if annotation_suffix else ""}{annotation_suffix}.json'
    return annotation_file_name


class VideoFileManager(pyfilemanager.FileManager):
    """
    File manager for organizing annotation and result files for one video.
    
    Provides convenient access to all files associated with a video in a DLC project:
    - Manual annotation JSON files
    - DLC prediction HDF5 files
    - Labeled data files for training
    
    Attributes:
        project_name (str): Name of the DLC project.
        video_stem (str): Video filename without extension.
        video_fname (str): Full path to video file.
    """
    def __init__(self, d: DLCProject, video_index: int):
        """
        Initialize file manager for a specific video.
        
        Args:
            d (DLCProject): Parent DLC project.
            video_index (int): Index of video in project's video list.
        """
        if not HAS_DLC:
            raise ImportError("Install deeplabcut to use VideoFileManager.")
        
        base_dir = d.paths['project']
        super().__init__(base_dir, exclude_hidden=True)
        self.add()
        self.project_name = d.name
        self.video_stem = d.video_names[video_index]
        self.video_fname = d.video_list[video_index]
    
    @property
    def annotations(self) -> dict:
        """
        Map annotation names to file paths.

        Returns:
            dict: {annotation_name: file_path} for all JSON annotation files.
        """
        pattern = f'*{self.video_stem}*_annotations*.json'
        file_names = fnmatch.filter([Path(x).name for x in self.all_files], pattern)
        files = [self[file_name][0] for file_name in file_names]
        return {self.canonical_layer_name(fname): fname for fname in files}

    @property
    def annotation_files(self) -> list:
        """List of full paths to annotation JSON files."""
        return list(self.annotations.values())

    @property
    def annotation_names(self) -> list:
        """List of annotation layer names (without paths or extensions)."""
        return list(self.annotations.keys())

    @staticmethod
    def canonical_layer_name(fname) -> str:
        """Single source of truth for DUSTrack layer names derived from a filepath.

        The DUSTrack workflow produces three categories of layer file:

        - Manual / hand-edited annotations: ``<video>_annotations[_<name>].json``.
          Returns the suffix after ``_annotations`` (or empty string if absent),
          which is what users picked when they saved.
        - DLC prediction traces: live under ``videos/iteration-{N}/`` and have
          ``DLC`` in the stem. Returns ``'dlc_iteration-{N}_<last underscore-token of stem>'``.
          This pattern also catches LK-RSTC post-processed outputs, which
          inherit the DLC source stem -- so a jitter-reduced layer gets a
          deterministic ``dlc_iteration-{N}_<window>`` name rather than the
          ``"noname"`` fallback that ``VideoAnnotation.__init__`` produces for
          paths without ``_annotations``.
        - Anything else: the file stem.

        Called by :attr:`annotations` / :attr:`dlc_traces` at fresh-load
        time AND by :meth:`DUSTrack._adopt_layer` for in-session adds, so
        the name a user sees in the layer panel is identical regardless
        of whether the layer was discovered on disk or produced live.
        """
        p = Path(fname)
        stem = p.stem
        if '_annotations' in stem:
            return stem.split('_annotations')[-1].removesuffix('.json').strip('_')
        if 'DLC' in stem:
            return 'dlc_' + p.parts[-2] + '_' + stem.split('_')[-1]
        return stem

    @staticmethod
    def _get_video_name(fname):
        """Return the 'name' of the video file <video_name>_annotations_<name>.json.
        For example, C:\\video01_annotations_brachialis_praneeth.json will return video01
        """
        return Path(fname).stem.split('_annotations')[0]

    @property
    def dlc_traces(self) -> dict:
        """
        Map DLC trace names to HDF5 file paths.

        Returns:
            dict: {trace_name: file_path} for all DLC prediction files.
                Trace names format: 'dlc_iteration-{N}_{training_iter}'
        """
        fm_temp = pyfilemanager.FileManager(str(Path(self.base_dir) / "videos")).add()
        fnames = fm_temp[f'{self.video_stem}DLC*{self.project_name}*.h5'] + fm_temp[f'{self.video_stem}DLC*{self.project_name}*.json']
        return {self.canonical_layer_name(fname): fname for fname in fnames}

    @property
    def dlc_trace_files(self):
        """List of full paths to DLC prediction HDF5 files."""
        return list(self.dlc_traces.values())

    @property
    def dlc_trace_names(self):
        """List of DLC trace identifiers."""
        return list(self.dlc_traces.keys())

    @property
    def labeled_data(self):
        """
        Path to HDF5 file containing training labels in DLC format.
        
        Returns:
            str: Full path to CollectedData HDF5 file.
        
        Raises:
            AssertionError: If file doesn't exist or multiple files found.
        """
        fm_temp = pyfilemanager.FileManager(str(Path(self.base_dir) / "labeled-data")).add()
        ret = fm_temp[f'{self.video_stem}*CollectedData*.h5']
        assert len(ret) == 1
        return ret[0]

    def get_new_json(self, new_suffix) -> Path:
        """
        Create path for a new annotation file with given suffix.
        Used to generate the the filename for the next refinement iteration.
        
        Args:
            new_suffix (str): Suffix for new annotation layer.
        
        Returns:
            Path: Full path to new JSON file.
        
        Raises:
            ValueError: If file with this suffix already exists.
        """
        annotations_json_new = (
            Path(self.video_fname).parent / 
            f'{self.video_stem}_annotations_{new_suffix}.json'
            )
        if os.path.exists(annotations_json_new):
            raise ValueError(f'File with {new_suffix} suffix already exists!')
        return annotations_json_new
    
    def get_all_annotation_layers(self, new_annotation_suffix: str=None):
        """
        Collect all annotation sources for loading into DUSTrack.
        
        Args:
            new_annotation_suffix (str, optional): Suffix for a new layer to create.
        
        Returns:
            dict: Maps layer names to file paths, including:
                - Existing JSON annotation files
                - New empty layer (if suffix provided)
                - Labeled training data
                - DLC prediction HDF5 files
        """
        if new_annotation_suffix is None:
            new_json = {}
        else:
            new_json = {new_annotation_suffix : self.get_new_json(new_annotation_suffix)}
        
        try:
            labeled_data = dict(labeled_data=self.labeled_data)
        except AssertionError:
            labeled_data = {}
        
        return dict(
            **self.annotations, 
            **new_json,
            **labeled_data, 
            **self.dlc_traces
            )


def merge_annotations_in_folder(path, annotation_suffix='merged'):
    """
    Merge multiple annotation files for each video in a folder.
    
    Useful for combining annotations from multiple annotators or sessions.
    Creates a single merged JSON file for each video.
    
    Args:
        path (str): Directory containing videos and annotation JSON files.
        annotation_suffix (str): Suffix for merged output files. Defaults to 'merged'.
    """
    fm = pyfilemanager.FileManager(path).add_by_depth(0)
    all_names = [Path(x).name for x in fm.all_files]
    all_video_names = fnmatch.filter(all_names, '*.mp4')
    video_files = [fm[name][0] for name in all_video_names]
    for video_file in video_files:
        video_stem = Path(video_file).stem.split('_annotations')[0]
        pattern = f'{video_stem}*_annotations*.json'
        file_names = fnmatch.filter(all_names, pattern)
        annotation_file_names = sorted([fm[file_name][0] for file_name in file_names])
        if len(annotation_file_names) == 0:
            continue
        print(f'Merging {len(annotation_file_names)} files for {video_stem}:')
        print(annotation_file_names)
        print(make_annotation_file_name(video_file, annotation_suffix))
        ann = VideoAnnotation.from_multiple_files(
            fname_list = annotation_file_names,
            vname = str(Path(path) / video_file),
            name = annotation_suffix,
            fname_merged = make_annotation_file_name(video_file, annotation_suffix)
        )
        ann.save()



def rebase_to_config(config_path: str, old_path: str) -> str:
    """
    Rebase 'old_path' (some file inside the project) onto the project root
    implied by 'config_path' (points to config.yaml or the project dir).

    Keeps the correct root/anchor:
      - Posix: leading "/"
      - Windows: drive letters (e.g., "C:\\") and UNC shares ("\\\\server\\share")
    Returns separators inferred from 'config_path'.
    """
    # Choose path flavor by the config path
    is_windows_like = ("\\" in config_path) or config_path.startswith("\\\\") or bool(re.match(r"^[A-Za-z]:", config_path))
    PathCls = PureWindowsPath if is_windows_like else PurePosixPath

    # Parse the config path *as-is* to keep its anchor
    cfg = PathCls(config_path)
    # Project root is the directory that contains config.yaml; if a directory is passed, use it
    new_root = cfg.parent if cfg.name.lower() == "config.yaml" else cfg
    if not new_root.name:
        raise ValueError(f"Cannot infer project folder name from: {config_path!r}")
    project_name = new_root.name

    # Split helper that handles both slash types
    split = lambda p: [x for x in re.split(r"[\\/]+", p.strip()) if x]

    old_parts = split(old_path)

    # Find the LAST occurrence of the project folder name (exact, then case-insensitive)
    def find_idx(parts, name):
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == name:
                return i
        name_cf = name.casefold()
        for i in range(len(parts) - 1, -1, -1):
            if parts[i].casefold() == name_cf:
                return i
        return None

    idx = find_idx(old_parts, project_name)
    if idx is None:
        raise ValueError(f"Project folder {project_name!r} not found in old_path: {old_path!r}")

    tail = old_parts[idx + 1:]
    rebased = new_root / PathCls(*tail) if tail else new_root
    return str(rebased)