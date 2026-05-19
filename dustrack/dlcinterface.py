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
from typing import Mapping, Union

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


class VideoAnnotation(dnav.VideoAnnotation):
    """
    Enhanced VideoAnnotation with integrated post-processing.

    This subclass extends dnav.VideoAnnotation by adding the Lucas-Kanade
    moving average filter as a default post-processing method for smoothing trajectories.

    Attributes:
        postprocess: Function reference to lk_moving_average_filter for jitter reduction.
    """
    postprocess = lk_moving_average_filter


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


class DUSTrack(dnav.VideoPointAnnotator):
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
        >>> # Basic usage
        >>> tracker = DUSTrack('video.mp4', "pn") # pn is the name of the annotation layer, that can be saved as {video_name}_annotations_pn.json
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

    def __init__(self, *args,
                 clahe_clip=2.0, clahe_grid=8, gamma=1.2, brightness=10,
                 dark_mode=False, enhance_enabled=True, **kwargs):
        # Store enhancement settings
        self._clahe_clip = clahe_clip
        self._clahe_grid = clahe_grid
        self._gamma = gamma
        self._brightness = brightness
        self._enhance_enabled = enhance_enabled
        self._dark_mode = dark_mode

        # Create image processor function
        def image_processor(im):
            if self._enhance_enabled:
                return enhance_ultrasound_image(
                    im, self._clahe_clip, self._clahe_grid,
                    self._gamma, self._brightness
                )
            return im

        kwargs['image_process_func'] = image_processor
        # DUSTrack defaults to datanavigator 1.5.0+ Tier 2 (Qt-native
        # video pane, ~3x speedup on real videos). Override with
        # ``DUSTrack(..., fast_render=False)`` only if a subclass needs
        # matplotlib Axes on the image region (no in-tree subclass
        # does today; this is forward-looking).
        kwargs.setdefault('fast_render', True)
        super().__init__(*args, **kwargs)

        for ann in self.annotations:
            ann.__class__ = VideoAnnotation

        self._dlcproject = None
        self._ax_lims = {'state': False, 'x': [None, None], 'y_trace_x': [None, None], 'y_trace_y': [None, None]}

        # Apply dark theme if enabled
        if dark_mode:
            self._apply_dark_theme()

        self.buttons.add(text="Keyboard shortcuts", action_func=(lambda s, ev: s.show_key_bindings(f="new", pos="center left")).__get__(self))
        # Double separators mark the major button-group boundaries in
        # rc2's vertical sidebar: shortcuts | DLC pipeline | trace +
        # display controls | (state variables, separated by dnav's own
        # trailing double separator on the statevars widget).
        self.buttons.add_separator(style="double")
        if HAS_DLC:
            self.buttons.add(text="Create DLC Project", action_func=self.create_dlc_project)
            self.buttons.add(text="Train DLC model", action_func=self.process_dlc_project)
            self.buttons.add(text="Apply manual corrections", action_func=self.apply_manual_corrections)
            self.buttons.add(text="Reduce jitter", action_func=self.process_with_lk)
            self.buttons.add_separator(style="double")
        self.buttons.add(text="Trace: line", action_func=(lambda s, ev: s.ann.set_plot_type("line")).__get__(self))
        self.buttons.add(text="Trace: dot", action_func=(lambda s, ev: s.ann.set_plot_type("dot")).__get__(self))
        self.buttons.add(text="Freeze plot axes", action_func=self.freeze_plot_axes)
        self.buttons.add(text="Unfreeze plot axes", action_func=self.unfreeze_plot_axes)
        self.buttons.add(text="Replace existing from overlay", action_func=self.copy_existing_annotations_from_overlay)
        self.buttons.add(text="Toggle enhance", action_func=self._toggle_enhancement)
        self.buttons.add(text="Save annotation as...", action_func=self.save_annotation_as)
        self.buttons.add_separator(style="double")
        self.buttons.add(text="Swap layers", action_func=self.swap_active_and_overlay)

        self.statevariables._text._pos = dnav.utils._parse_pos("bottom left")
        
        if self.__class__.__name__ == "DUSTrack":
            plt.show(block=False)
            self.update()
            plt.setp(self._ax_trace_x.get_xticklabels(), visible=False)
            plt.draw()

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

    def _toggle_enhancement(self, event=None):
        """Toggle image enhancement on/off."""
        self._enhance_enabled = not self._enhance_enabled
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


    def create_dlc_project(self, event=None, name=None, path=None, experimenter=_config.EXPERIMENTER) -> DLCProject:
        """
        Create a new DeepLabCut project using current annotations as training labels.

        rc2 (1.1.0rc2): on a Qt backend, project creation runs off the
        GUI thread under a modal overlay (no progress bar -- it's a
        fast op, but the overlay surfaces DLC's stdout and a Done
        button so the user can confirm the project location and any
        warnings before continuing). On non-Qt backends the call runs
        synchronously and returns the new :class:`DLCProject`.

        Args:
            event: Mouse/keyboard event (unused, for button compatibility).
            name (str, optional): Project name. Defaults to "{video_name}_{annotation_layer}".
            path (str, optional): Directory for project. Defaults to video's parent directory.
            experimenter (str, optional): Experimenter name. Defaults to config value.

        Returns:
            DLCProject: The newly created project instance on the sync
            path. ``None`` on the Qt async path -- read
            ``self._dlcproject`` after the Done button is clicked.

        Note:
            Project names must contain an underscore for proper DLC configuration handling.
        """
        if not HAS_DLC:
            raise ImportError('deeplabcut is not installed. Cannot create DLC project.')
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

        qt_window = self._find_qt_window()
        if qt_window is None:
            self._dlcproject = _build_project()
            self._rewire_to_in_project_paths()
            return self._dlcproject

        def _on_success(project: DLCProject):
            self._dlcproject = project
            self._rewire_to_in_project_paths()

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
        project_root = Path(self._dlcproject.path)

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

        A **pre-flight check on the active annotation layer** runs
        before the overlay starts: any frame missing one or more
        bodyparts triggers a modal with a per-bodypart breakdown and
        a choice between *Drop and train* (drop the incomplete frames,
        write a ``<fstem>.dustrack-dropped-incomplete-<ts>`` recovery
        sidecar next to the annotation file, save the trimmed layer,
        then train) and *Cancel* (return to the UI to fix manually).
        On the clean path the active layer is still saved right
        before training kicks off, so the on-disk state reflects
        exactly what feeds DLC's training input. The check covers
        the active layer only; sibling annotation files in the
        project folder may have their own incomplete frames that
        still feed :meth:`DLCProject.extract_frames`.

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
            *args: Additional positional arguments forwarded to
                :meth:`DLCProject.process`.
            **kwargs: Additional keyword arguments forwarded to
                :meth:`DLCProject.process`.

        Returns:
            DUSTrack: ``self`` on the Qt path (training is asynchronous;
            the same DUSTrack will refresh in place when the user
            clicks Done). Also ``self`` if the user cancels the
            pre-flight modal -- the UI is left intact for manual
            fixes. On the fallback path, the freshly-launched
            DUSTrack from :meth:`DLCProject.annotate`.

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
            plt.close(self.figure)
            self._dlcproject.process(*args, **kwargs)
            return self._dlcproject.annotate()

        # Pre-flight: scan the active layer for incomplete frames. If
        # any, ask the user whether to drop them (with sidecar) or
        # bail back to the UI.
        incomplete = self._scan_incomplete_frames(self.ann.data)
        if incomplete:
            if not self._prompt_drop_or_cancel(qt_window, incomplete):
                return self  # user cancelled -- leave the UI intact
            self._save_dropped_incomplete_sidecar(incomplete)
            # remove_empty_labels first so the underlying
            # frames_overlapping is well-defined (placeholder labels
            # with zero annotations would otherwise empty the set).
            self.ann.remove_empty_labels()
            self.keep_overlapping_frames()

        # Save the active layer right before training kicks off so the
        # on-disk state matches what DLC will see.
        self.save()

        def _train():
            self._dlcproject.process(*args, **kwargs)

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
    def _scan_incomplete_frames(data: dict) -> dict:
        """Find frames missing one or more bodyparts in an annotation
        ``data`` dict (``{label: {frame: [x, y]}}``).

        Considers only labels with at least one annotation -- empty
        labels are UI placeholders and shouldn't fail every frame.
        Returns ``{frame: [missing_label, ...]}`` for incomplete
        frames, frame-sorted with missing-labels lists in the same
        order as the active label list. Empty dict iff every active
        frame has every active label.

        Pure data-in / data-out; testable from synthetic dicts.
        """
        active_labels = [L for L, frames in data.items() if frames]
        if not active_labels:
            return {}
        all_frames: set = set()
        for L in active_labels:
            all_frames.update(data[L].keys())
        incomplete: dict = {}
        for frame in sorted(all_frames):
            missing = [L for L in active_labels if frame not in data[L]]
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

    def _save_dropped_incomplete_sidecar(self, incomplete_frames: dict):
        """Persist the dropped-frame contents next to the active layer.

        Returns the sidecar path on success, ``None`` if the active
        layer has no on-disk filename (in-memory only).
        """
        import datetime
        import json
        if self.ann.fname is None:
            return None
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        sidecar = self._build_dropped_incomplete_sidecar_name(self.ann.fname, ts)
        payload = self._build_dropped_incomplete_payload(self.ann.data, incomplete_frames)
        with open(sidecar, "w") as f:
            json.dump(payload, f, indent=2)
        return sidecar

    def _prompt_drop_or_cancel(self, qt_window, incomplete_frames: dict) -> bool:
        """Show the pre-flight modal; return True iff user picked
        *Drop and train*.
        """
        from qtpy.QtWidgets import QMessageBox

        n = len(incomplete_frames)
        msg = QMessageBox(qt_window)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Incomplete frames detected")
        msg.setText(
            f"{n} frame{'s' if n != 1 else ''} in the active layer "
            f"({self.ann.name!r}) "
            f"{'are' if n != 1 else 'is'} missing one or more bodyparts."
        )
        msg.setInformativeText(
            "DeepLabCut tolerates per-bodypart NaN in its CSV but partial "
            "frames degrade the trained model in practice.\n\n"
            "Choose Drop and train to drop these frames from the active "
            "layer (a sidecar is written so the data is recoverable) and "
            "kick off training. Choose Cancel to return to the UI and fix "
            "them manually.\n\n"
            "Note: this check covers the active layer only. Other "
            "annotation files in the project folder may have their own "
            "incomplete frames that will still feed DLC's training input."
        )
        msg.setDetailedText(self._format_incomplete_breakdown(incomplete_frames))
        drop_btn = msg.addButton("Drop and train", QMessageBox.AcceptRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)
        msg.setDefaultButton(cancel_btn)
        msg.exec_()
        return msg.clickedButton() is drop_btn

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
                    sys.__stderr__.write(
                        f"{title} succeeded but follow-up failed: {e}\n"
                    )
                    overlay.mark_done(
                        success=False,
                        summary=(
                            f"Work succeeded, but follow-up step raised "
                            f"{type(e).__name__}: {e}"
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
        ``dlc_*`` layers to a line plot, point the overlay statevar at
        the freshest DLC trace, and activate the new iteration layer
        so the user can immediately start annotating.

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
        if new_suffix in self.annotations.names:
            self.statevariables["annotation_layer"].set_state(new_suffix)
        self.update()

    def _normalize_dlc_layer_display(self, scope=None):
        """Apply the post-load display convention for DLC trace layers:
        every ``dlc_*`` layer renders as a line plot, and the latest
        ``dlc_*`` layer is set as the ``annotation_overlay``.

        ``scope`` selects which layer names participate in the overlay
        decision (the line-plot conversion is always applied to that
        same scope -- ``set_plot_type`` is idempotent):

        - ``None`` -- fresh-construction path: scope = all current
          ``dlc_*`` layers in the session. Always (re-)points the
          overlay if at least one exists.
        - iterable of names -- in-place refresh path: scope = the
          freshly-added layers only. If none of them are ``dlc_*``,
          the overlay isn't touched (preserves prior selection).

        Single source of truth shared by :meth:`DLCProject.annotate` and
        :meth:`_refresh_dlc_layers` so the on-screen state is identical
        regardless of how the user entered the session.
        """
        if scope is None:
            dlc_names = [a.name for a in self.annotations if a.name.startswith("dlc_")]
        else:
            dlc_names = [n for n in scope if n.startswith("dlc_")]
        for name in dlc_names:
            self.annotations[name].set_plot_type("line")
        if dlc_names:
            self.statevariables["annotation_overlay"].set_state(dlc_names[-1])

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
        - If the layer name starts with ``dlc_``, runs
          :meth:`_normalize_dlc_layer_display` over just this one layer so
          plot-type and ``annotation_overlay`` end up where they would after
          a close + reopen.
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
        # NOTE: `dnav.VideoAnnotation`, not the dustrack subclass.
        # ``lk_moving_average_filter`` returns the parent class (the
        # subclass adds the ``postprocess`` hook only), so the narrower
        # ``isinstance`` was silently falling through to ``str(obj)`` --
        # producing layer names like ``"<datanavigator.pointtracking"``
        # and an empty data dict (load() of a non-existent path).
        if isinstance(ann_or_fname, dnav.VideoAnnotation):
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
            if name.startswith("dlc_"):
                self._normalize_dlc_layer_display(scope=[name])
        # Apply the requested overlay / active state even if the layer
        # was already present -- e.g. Reduce jitter on a layer whose
        # cached output is already loaded should still swap the UI to
        # the smoothed layer with the source pinned as overlay.
        if set_overlay is not None:
            self.statevariables["annotation_overlay"].set_state(set_overlay)
        if set_active:
            self.statevariables["annotation_layer"].set_state(name)
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

        **Post-apply state.** The corrections layer becomes the
        active annotation layer and the manual layer (previously
        active) becomes the overlay so you can see where your hand
        was. To iterate, switch the active layer back to your manual
        layer, set the overlay back to the DLC trace, add more
        points, click again. Each click regenerates the corrections
        layer from the current ``(overlay, active)`` pair, so adding
        annotations directly to the corrections layer is not
        recommended -- they'll be discarded on the next apply.

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

        self.statevariables["annotation_layer"].set_state(self.CORRECTIONS_LAYER_NAME)
        self.statevariables["annotation_overlay"].set_state(patch_name)
        self.update()

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
        Copy DUSTrack/VideoPointAnnotator JSON files into project's video folder.
        
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


def open(path, layer_name=None, **dustrack_kwargs):
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
        path: Video file, ``config.yaml``, or DLC project folder.
        layer_name: Annotation layer name. **Required** for Phase 1
            (e.g. ``'manual'``). Optional for Phase 2; defaults to
            ``iteration-{N+1}`` (next-iteration suffix) when omitted.
        **dustrack_kwargs: Forwarded to the underlying :class:`DUSTrack`
            constructor (``dark_mode``, ``fast_render``, ``clahe_clip``,
            ``gamma``, ``brightness``, etc.).

    Returns:
        DUSTrack: Live annotation UI, ready to use.

    Raises:
        FileNotFoundError: If ``path`` doesn't exist.
        ValueError: Phase 1 entry without ``layer_name``, or a
            directory that isn't a DLC project.
        ImportError: Phase 2 entry on a system without ``deeplabcut``
            installed.

    Examples:
        Fresh annotation::

            import dustrack
            tracker = dustrack.open('video.mp4', 'manual')

        Resume after closing the UI mid-workflow (any of these work)::

            tracker = dustrack.open('S:/path/to/project/videos/video.mp4')
            tracker = dustrack.open('S:/path/to/project/config.yaml')
            tracker = dustrack.open('S:/path/to/project/')

        With UI options::

            tracker = dustrack.open('video.mp4', 'manual', dark_mode=True)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"dustrack.open: path does not exist: {path}")

    config_path = _find_dlc_config(p)

    if config_path is None:
        # Phase 1: no DLC project context.
        if not p.is_file():
            raise ValueError(
                f"dustrack.open: {path!s} is a directory but doesn't look like "
                "a DLC project (no config.yaml + videos/ + labeled-data/). "
                "Pass a video file or a DLC project folder."
            )
        if layer_name is None:
            raise ValueError(
                "dustrack.open: layer_name is required when opening a video "
                "outside a DLC project. Example: "
                "dustrack.open('video.mp4', 'manual')."
            )
        return DUSTrack(str(p), layer_name, **dustrack_kwargs)

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

    return project.annotate(
        video_index=video_index,
        new_annotation_suffix=layer_name,
        **dustrack_kwargs,
    )


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
    # No need to set a bridge; default 'native' is fine and we use .asnumpy()
    vr = VideoReader(video_file_name, ctx=cpu(0), num_threads=1)  # HWC RGB uint8
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