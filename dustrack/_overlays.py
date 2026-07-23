"""Modal / overlay factory functions: Progress, Confirm, TrainingOptions,
SeedBundlePicker, OpenVideoOverlay + the first-paint-notice helper +
stdout teeing for the progress overlay log.

Five overlay classes, each built lazily by a factory function so qtpy
is only imported when actually instantiated (keeps the mpl-fallback
path importable without Qt). Two pure helpers for the Training-options
modal (:func:`_default_training_options`,
:func:`_training_options_to_train_iteration_kwargs`) plus
:func:`_render_recent_session_label` for the OpenVideoOverlay's
recent-sessions list.

Extracted from ``dlcinterface.py`` in dustrack 1.2.0rc1.
"""

from __future__ import annotations

import os
import queue
import re
import sys
from pathlib import Path

from . import _config
from . import dlcloader as _dlcloader


# Phase / progress detection on DLC's stdout. We don't depend on any
# single DLC version's exact format -- if nothing matches, the overlay
# stays in indeterminate-busy mode and the status label shows the last
# recognised phase. Patterns ordered most-specific first.
_TRAINING_PHASES = [
    (
        re.compile(r"extract_frames|extracting frame", re.IGNORECASE),
        "Extracting frames",
    ),
    (
        re.compile(r"create_training_dataset|creating training", re.IGNORECASE),
        "Creating training dataset",
    ),
    (
        re.compile(r"initialize.*weights|loading.*snapshot", re.IGNORECASE),
        "Initializing weights",
    ),
    (
        re.compile(r"started training|train_network|begin training", re.IGNORECASE),
        "Training network",
    ),
    (re.compile(r"evaluate_network|evaluating", re.IGNORECASE), "Evaluating snapshots"),
    (re.compile(r"analyze_videos|analyzing video", re.IGNORECASE), "Analyzing videos"),
    (
        re.compile(r"create_labeled_video|labeled video", re.IGNORECASE),
        "Creating labeled video",
    ),
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
    (
        re.compile(r"Processing tracking jobs", re.IGNORECASE),
        "Processing tracking jobs",
    ),
    (re.compile(r"Submitting jobs", re.IGNORECASE), "Submitting tracking jobs"),
    (re.compile(r"Processing results", re.IGNORECASE), "Processing tracking results"),
    (re.compile(r"Processing sequentially", re.IGNORECASE), "Processing sequentially"),
]
# DLC project creation chatters about copying videos and writing the
# config; useful as phase labels even when the operation completes in
# under a second.
_CREATE_PROJECT_PHASES = [
    (
        re.compile(r"Created.*\bproject\b|new project", re.IGNORECASE),
        "Project skeleton created",
    ),
    (re.compile(r"adding.*video|copying.*video", re.IGNORECASE), "Copying video"),
    (re.compile(r"config.*yaml|writing.*config", re.IGNORECASE), "Writing config"),
    (
        re.compile(r"labeled-data|extract", re.IGNORECASE),
        "Preparing labeled-data folders",
    ),
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
                hint or "Output is also streamed to the launching terminal."
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
            self._hint.setText("Review the output above, then click Done to continue.")
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
                    sys.__stderr__.write(f"Post-overlay callback raised: {exc}\n")

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
        QCheckBox,
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
            checkboxes=None,  # list[dict(key, label, default_checked)]
        ):
            super().__init__(main_window)
            self._mw = main_window
            self._result: str | None = None
            self._loop = QEventLoop()
            # checkbox_states is populated for every exec_() (empty
            # dict when no checkboxes were supplied) so callers can
            # always read it unconditionally.
            self._checkbox_specs = list(checkboxes or [])
            self._checkbox_widgets: dict = {}
            self.checkbox_states: dict = {
                spec["key"]: bool(spec.get("default_checked", False))
                for spec in self._checkbox_specs
            }

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

            # Optional checkboxes -- rendered as a centered column
            # between the message and the button row. Each checkbox's
            # state is tracked on ``self.checkbox_states[key]`` and
            # is the canonical place callers read post-``exec_()``.
            # QSS pins the indicator to a visible size + the label
            # to white so it reads on the dark backdrop (Windows-
            # native disabled is too subtle here, per
            # feedback_dark_overlay_native_disabled_subtle).
            for spec in self._checkbox_specs:
                key = spec["key"]
                cb = QCheckBox(spec["label"])
                cb.setChecked(self.checkbox_states[key])
                cb.setStyleSheet(
                    "QCheckBox { color: white; font-size: 11pt; }"
                    "QCheckBox::indicator { width: 16px; height: 16px; }"
                )
                cb.stateChanged.connect(
                    lambda state, k=key: self._on_checkbox_changed(k, state)
                )
                self._checkbox_widgets[key] = cb
                layout.addWidget(cb, alignment=Qt.AlignCenter)

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
                btn.clicked.connect(
                    lambda _checked=False, lbl=label: self._on_clicked(lbl)
                )
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

        def _on_checkbox_changed(self, key: str, state: int) -> None:
            self.checkbox_states[key] = bool(state)

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
    trained = sorted(i for i, snaps in dlcproject.all_snapshots.items() if snaps)
    snapshots_by_iteration = {i: list(dlcproject.all_snapshots[i]) for i in trained}
    has_trained = bool(trained)
    return {
        "refine_mode": "in_project" if has_trained else "scratch",
        "source_iteration": trained[-1] if has_trained else None,
        "source_snapshot": None,  # None == "best" (initialize_weights default)
        "external_snapshot_path": "",
        "maxiters": 50 if _dlcloader.DLC3 else 500000,
        "create_video": False,
        # Combo population helpers; not forwarded to train_iteration.
        "trained_iterations": trained,
        "snapshots_by_iteration": snapshots_by_iteration,
        "is_dlc3": bool(_dlcloader.DLC3),
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


def _default_create_project_options(*, video_fname, layer_name, experimenter,
                                    last_project_root=None):
    """Initial state dict for the Create DLC Project modal.

    Pure-Python, no Qt deps -- testable in isolation. Computes the
    three editable fields' defaults:

    * ``name``: ``"{video_stem}_{layer_name}"`` -- DLC suffixes
      ``-<experimenter>-<date>`` to form the actual project folder.
    * ``path``: ``last_project_root`` if supplied (the directory the
      user last created a project into, persisted across sessions),
      else the active video's parent folder.
    * ``experimenter``: passed through from the caller (``_config``
      default unless overridden).

    Args:
        video_fname: Active video path (its stem drives the name and
            its parent is the fallback project folder).
        layer_name: Active annotation layer name (the name suffix).
        experimenter: Experimenter string.
        last_project_root: Remembered project root (``None`` on first
            use -> falls back to the video's parent).

    Returns:
        dict: ``{"name", "path", "experimenter"}`` -- the modal seeds
        its three line edits from this.
    """
    from pathlib import Path as _Path

    video = _Path(video_fname)
    default_root = _Path(last_project_root) if last_project_root else video.parent
    return {
        "name": f"{video.stem}_{layer_name}",
        "path": str(default_root),
        "experimenter": str(experimenter),
    }


def _validate_create_project_options(options):
    """Validate Create DLC Project modal state. Pure-Python.

    Returns ``(ok: bool, error_message: str)``. The modal's Create
    button stays disabled / shows the message until all checks pass:

    * ``name`` non-empty and contains an underscore (DLC config
      handling breaks on underscore-free names with network paths --
      see :class:`DLCProject` docstring).
    * ``path`` non-empty (existence is not required -- DLC creates it).
    * ``experimenter`` non-empty.
    """
    name = (options.get("name") or "").strip()
    path = (options.get("path") or "").strip()
    experimenter = (options.get("experimenter") or "").strip()
    if not name:
        return False, "Project name is required."
    if "_" in experimenter:
        # DLC's folder name is <name>-<experimenter>-<date>; an
        # underscore in experimenter is fine, but a dash would split
        # the parse. Guard the dash instead.
        pass
    if "-" in name:
        return False, "Project name cannot contain '-' (reserved by DLC's folder naming)."
    if not path:
        return False, "Project folder is required."
    if not experimenter:
        return False, "Experimenter is required."
    if "-" in experimenter:
        return False, "Experimenter cannot contain '-' (reserved by DLC's folder naming)."
    return True, ""


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
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFrame,
        QGraphicsOpacityEffect,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QRadioButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
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
                # Disabled state -- mute text + indicator so a disabled
                # radio (e.g. "Refine from in-project iteration" with no
                # trained iterations yet) is unmistakably non-actionable
                # on the dark backdrop. The same colors are reused on
                # QCheckBox:disabled for consistency.
                "QRadioButton:disabled { color: #777777; }"
                "QCheckBox:disabled { color: #777777; }"
                "QRadioButton::indicator:disabled { "
                "  background-color: transparent; "
                "  border: 2px solid #666666; "
                "  border-radius: 8px; "
                "}"
                "QRadioButton::indicator:checked:disabled { "
                "  background-color: #4a5a7a; "
                "  border: 2px solid #666666; "
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
            self._in_project_radio = QRadioButton("Refine from in-project iteration")
            self._external_radio = QRadioButton("Refine from external snapshot")
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
                    "Training epochs:"
                    if initial_state.get("is_dlc3", True)
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
            self._create_video_chk = QCheckBox("Create labeled video on completion")
            self._create_video_chk.setChecked(bool(initial_state["create_video"]))
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
            # QGraphicsOpacityEffect dims the whole row (including the
            # native QComboBox / QLineEdit / QSpinBox / QPushButton
            # children, whose Windows-native disabled rendering is too
            # subtle on this dark backdrop). The QLabel children stay
            # white when enabled and inherit the opacity fade when
            # disabled -- no per-label restyling needed.
            if enabled:
                row.setGraphicsEffect(None)
            else:
                effect = QGraphicsOpacityEffect(row)
                effect.setOpacity(0.40)
                row.setGraphicsEffect(effect)
            for child in row.findChildren(QLabel):
                child.setStyleSheet("color: white;")

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
                    self._iter_combo.currentData() if mode == "in_project" else None
                ),
                "source_snapshot": (
                    self._snap_combo.currentData() if mode == "in_project" else None
                ),
                "external_snapshot_path": (
                    self._external_path_edit.text() if mode == "external" else ""
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


def _make_create_project_options_class():
    """Build :class:`CreateProjectOptionsDialog` lazily, mirroring
    :func:`_make_training_options_class`'s qtpy-import-on-demand pattern.

    Modal shown when the user clicks Create DLC Project (1.3.0a2).
    Three editable line edits -- project **name**, project **folder**
    (with a Browse... button), and **experimenter** -- pre-populated
    from :func:`_default_create_project_options`. The user can OK
    straight through or edit any field. A live validation label
    surfaces why Create is disabled (empty name, dash in name, etc.).

    On Create, ``exec_()`` returns ``{"name", "path", "experimenter"}``;
    on Cancel, ``None``. There is deliberately no "link videos" toggle
    -- hard-linking with copy fall-back is the always-right default
    (see ``DLCProject.__init__``'s ``link_videos`` kwarg for the
    scripted override).
    """
    from qtpy.QtCore import QEvent, QEventLoop, QObject, Qt
    from qtpy.QtWidgets import (
        QFileDialog,
        QFrame,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    _ROLE_QSS = {
        "primary": (
            "QPushButton { background-color: #3a86ff; color: white; "
            "  border: 1px solid #2a76ef; padding: 6px 24px; "
            "  font-size: 11pt; font-weight: bold; }"
            "QPushButton:hover { background-color: #4a96ff; }"
            "QPushButton:pressed { background-color: #2a76ef; }"
            "QPushButton:disabled { background-color: #2a4a7a; color: #99aacc; }"
        ),
        "neutral": (
            "QPushButton { background-color: #555555; color: white; "
            "  border: 1px solid #444444; padding: 6px 24px; "
            "  font-size: 11pt; }"
            "QPushButton:hover { background-color: #666666; }"
            "QPushButton:pressed { background-color: #444444; }"
        ),
    }

    class CreateProjectOptionsDialog(QObject):
        """Synchronous modal for the Create DLC Project pre-flight.

        ``initial_state`` is the dict from
        :func:`_default_create_project_options`. On Create, ``exec_()``
        returns a dict with the user-edited values; on Cancel, ``None``.
        """

        def __init__(self, main_window, *, initial_state: dict):
            super().__init__(main_window)
            self._mw = main_window
            self._result = None
            self._loop = QEventLoop()
            self._state = dict(initial_state)

            self._frame = QFrame(main_window)
            self._frame.setObjectName("dustrack_create_project_overlay")
            self._frame.setStyleSheet(
                "#dustrack_create_project_overlay { "
                "  background-color: rgba(0, 0, 0, 200); "
                "}"
                "QLabel { color: white; }"
                "#dustrack_create_project_title { color: white; "
                "  font-size: 22pt; font-weight: bold; }"
                "#dustrack_create_project_error { color: #ff9a9a; "
                "  font-size: 10pt; }"
            )
            self._frame.setFocusPolicy(Qt.StrongFocus)

            outer = QVBoxLayout(self._frame)
            outer.setAlignment(Qt.AlignCenter)
            outer.addStretch(1)

            title_lbl = QLabel("Create DLC project")
            title_lbl.setObjectName("dustrack_create_project_title")
            title_lbl.setAlignment(Qt.AlignCenter)
            outer.addWidget(title_lbl)

            # Inner content card: no QSS on QWidget {} so the native
            # QLineEdit rendering survives (feedback_qt_qss_vs_palette).
            content = QWidget()
            content.setMinimumWidth(560)
            content.setMaximumWidth(720)
            content_layout = QVBoxLayout(content)
            content_layout.setSpacing(10)

            # --- Project name ---
            content_layout.addWidget(QLabel("Project name"))
            self._name_edit = QLineEdit(self._state.get("name", ""))
            self._name_edit.textChanged.connect(self._revalidate)
            content_layout.addWidget(self._name_edit)

            # --- Project folder + Browse ---
            content_layout.addWidget(QLabel("Project folder"))
            folder_row = QHBoxLayout()
            self._path_edit = QLineEdit(self._state.get("path", ""))
            self._path_edit.textChanged.connect(self._revalidate)
            folder_row.addWidget(self._path_edit, stretch=1)
            self._browse_btn = QPushButton("Browse...")
            self._browse_btn.setStyleSheet(_ROLE_QSS["neutral"])
            self._browse_btn.clicked.connect(self._on_browse_clicked)
            folder_row.addWidget(self._browse_btn)
            content_layout.addLayout(folder_row)

            # --- Experimenter ---
            content_layout.addWidget(QLabel("Experimenter"))
            self._experimenter_edit = QLineEdit(self._state.get("experimenter", ""))
            self._experimenter_edit.textChanged.connect(self._revalidate)
            content_layout.addWidget(self._experimenter_edit)

            # --- Live validation message ---
            self._error_lbl = QLabel("")
            self._error_lbl.setObjectName("dustrack_create_project_error")
            self._error_lbl.setWordWrap(True)
            content_layout.addWidget(self._error_lbl)

            # --- Create / Cancel buttons ---
            button_row = QHBoxLayout()
            button_row.setAlignment(Qt.AlignCenter)
            self._create_btn = QPushButton("Create")
            self._create_btn.setMinimumWidth(160)
            self._create_btn.setStyleSheet(_ROLE_QSS["primary"])
            self._create_btn.clicked.connect(self._on_create_clicked)
            self._cancel_btn = QPushButton("Cancel")
            self._cancel_btn.setMinimumWidth(160)
            self._cancel_btn.setStyleSheet(_ROLE_QSS["neutral"])
            self._cancel_btn.clicked.connect(self._on_cancel_clicked)
            button_row.addWidget(self._create_btn)
            button_row.addWidget(self._cancel_btn)
            content_layout.addLayout(button_row)

            outer.addWidget(content, alignment=Qt.AlignCenter)
            outer.addStretch(1)

            main_window.installEventFilter(self)
            self._revalidate()

            self._frame.show()
            self._reposition()
            self._frame.raise_()
            self._name_edit.setFocus()

        # -- Slots -----------------------------------------------------

        def _current_options(self):
            return {
                "name": self._name_edit.text().strip(),
                "path": self._path_edit.text().strip(),
                "experimenter": self._experimenter_edit.text().strip(),
            }

        def _revalidate(self, *_):
            ok, msg = _validate_create_project_options(self._current_options())
            self._create_btn.setEnabled(ok)
            self._error_lbl.setText("" if ok else msg)

        def _on_browse_clicked(self):
            start_dir = self._path_edit.text().strip() or ""
            chosen = QFileDialog.getExistingDirectory(
                self._mw,
                "Choose the folder to create the DLC project in",
                start_dir,
            )
            if chosen:
                self._path_edit.setText(chosen)

        def _on_create_clicked(self):
            options = self._current_options()
            ok, _msg = _validate_create_project_options(options)
            if not ok:
                return  # button shouldn't be enabled, but guard anyway
            self._result = options
            self._dismiss()
            self._loop.quit()

        def _on_cancel_clicked(self):
            self._result = None
            self._dismiss()
            self._loop.quit()

        # -- Lifecycle (mirror TrainingOptionsDialog) ------------------

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
            """Block until the user clicks Create or Cancel.

            Returns ``{"name", "path", "experimenter"}`` on Create;
            ``None`` on Cancel.
            """
            self._loop.exec_()
            return self._result

    return CreateProjectOptionsDialog


# ---------------------------------------------------------------------------
# Blip-detection options modal (Detect blip outliers click flow, 1.3.0)
# ---------------------------------------------------------------------------


def _format_blip_results_text(report) -> str:
    """Render a :class:`dustrack.blip.BlipReport` as the multi-line
    string the modal shows in its results section. Pure-data helper
    so the rendering is unit-testable without Qt.
    """
    if report is None:
        return "Press Detect to scan the active layer."

    lines = []
    total = len(report)
    if total == 0:
        lines.append("0 blips found.")
        # Still surface per-label stats so users can see what the
        # threshold was — informative when tuning knobs down.
        for label, stats in report.per_label_stats.items():
            med = stats.get("median_d", float("nan"))
            thr = stats.get("threshold", float("nan"))
            lines.append(
                f"  Label {label!r}:  med={med:.3f}  threshold={thr:.3f}"
            )
        return "\n".join(lines)

    for label, stats in report.per_label_stats.items():
        n = int(stats.get("n_blips", 0))
        med = stats.get("median_d", float("nan"))
        thr = stats.get("threshold", float("nan"))
        lines.append(
            f"Label {label!r}:  {n} blips  med={med:.3f}  threshold={thr:.3f}"
        )
    lines.append(f"Total: {total} blips")
    hist = report.length_histogram()
    if hist:
        hist_str = "  ".join(f"{length}:{count}" for length, count in hist.items())
        lines.append(f"Length histogram: {hist_str}")
    # Skipped summary -- aggregate across labels.
    n_edge = sum(int(s.get("n_skipped_edge", 0)) for s in report.per_label_stats.values())
    n_long = sum(int(s.get("n_skipped_long", 0)) for s in report.per_label_stats.values())
    n_noreturn = sum(
        int(s.get("n_skipped_noreturn", 0)) for s in report.per_label_stats.values()
    )
    if n_edge or n_long or n_noreturn:
        lines.append(
            f"(skipped: edge={n_edge}  long={n_long}  noreturn={n_noreturn})"
        )
    return "\n".join(lines)


def _make_blip_options_class():
    """Build :class:`BlipOptionsDialog` lazily, mirroring
    :func:`_make_training_options_class`'s qtpy-import-on-demand pattern.

    Two-stage modal for the **Detect blip outliers** workflow button.
    Stage 1: user tunes the three detection knobs (threshold factor,
    max blip length, return position factor) and clicks **Detect** to
    run :func:`dustrack.blip.detect_blips` synchronously against the
    active annotation. Detection is fast (~0.14 s on a 36715-frame
    pia02 trace), so an in-modal blocking call is fine and avoids the
    overhead of an extra async overlay for the cheap stage.

    Stage 2: results populate in-modal (per-label blip counts +
    thresholds + length histogram); the **Interpolate** button enables.
    Clicking Interpolate closes the modal and returns
    ``(report, knobs)``; the caller (``DUSTrack.detect_blips_workflow``)
    kicks off the slow LK interpolation pass under a separate
    :class:`ProgressOverlay`. **Cancel** returns ``None`` -- caller
    aborts.

    Knobs persist across re-Detect clicks within the same modal session
    so users can iterate ("tune down threshold_factor, look at the
    histogram, tune again").
    """
    from qtpy.QtCore import QEvent, QEventLoop, QObject, Qt
    from qtpy.QtWidgets import (
        QCheckBox,
        QDoubleSpinBox,
        QFrame,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    # Reuse ConfirmOverlay's role QSS (same vocab as Train/Cancel).
    _ROLE_QSS = {
        "primary": (
            "QPushButton { background-color: #3a86ff; color: white; "
            "  border: 1px solid #2a76ef; padding: 6px 24px; "
            "  font-size: 11pt; font-weight: bold; }"
            "QPushButton:hover { background-color: #4a96ff; }"
            "QPushButton:pressed { background-color: #2a76ef; }"
            "QPushButton:disabled { background-color: #2a4a7a; color: #999999; "
            "  border: 1px solid #244270; }"
        ),
        "neutral": (
            "QPushButton { background-color: #555555; color: white; "
            "  border: 1px solid #444444; padding: 6px 24px; "
            "  font-size: 11pt; }"
            "QPushButton:hover { background-color: #666666; }"
            "QPushButton:pressed { background-color: #444444; }"
        ),
    }

    # Checkbox QSS so the indicator + label are legible on the dark
    # rgba(0,0,0,200) backdrop (Windows-native is too subtle per
    # [[feedback_dark_overlay_native_disabled_subtle]]).
    _CHECKBOX_QSS = (
        "QCheckBox { color: white; font-size: 11pt; }"
        "QCheckBox::indicator { width: 16px; height: 16px; }"
    )

    def _labeled_row(text: str, widget, tooltip: str = ""):
        """Helper: ``[QLabel(text)  widget  <stretch>]`` row layout.

        ``tooltip``, if non-empty, is attached to both the label and
        the widget so hovering either surface explains the knob. Plain
        text (no QSS); Qt renders it as a tooltip via the OS toolkit.
        """
        row = QHBoxLayout()
        lbl = QLabel(text + ":")
        lbl.setMinimumWidth(180)
        if tooltip:
            lbl.setToolTip(tooltip)
            widget.setToolTip(tooltip)
        row.addWidget(lbl)
        row.addWidget(widget)
        row.addStretch(1)
        return row

    class BlipOptionsDialog(QObject):
        """Synchronous two-stage modal for Detect blip outliers.

        ``exec_()`` returns ``(report, params_dict, drop_frame_if_any_blip)`` on Remove blips,
        ``None`` on Cancel. ``report`` is whatever the last Detect run
        produced (guaranteed to have ``len(blips) > 0`` because the
        Interpolate button is gated on non-empty results).
        """

        def __init__(
            self,
            main_window,
            *,
            ann,
            detect_fn,
        ):
            super().__init__(main_window)
            self._mw = main_window
            self._ann = ann
            self._detect_fn = detect_fn
            self._result = None
            self._loop = QEventLoop()
            self._last_report = None

            self._frame = QFrame(main_window)
            self._frame.setObjectName("dustrack_blip_options_overlay")
            self._frame.setStyleSheet(
                "#dustrack_blip_options_overlay { "
                "  background-color: rgba(0, 0, 0, 200); "
                "}"
                "QLabel { color: white; }"
                "#dustrack_blip_title { color: white; "
                "  font-size: 22pt; font-weight: bold; }"
                "#dustrack_blip_results { color: white; font-family: "
                "  'Consolas', 'Courier New', monospace; font-size: 10pt; }"
            )
            self._frame.setFocusPolicy(Qt.StrongFocus)

            outer = QVBoxLayout(self._frame)
            outer.setAlignment(Qt.AlignCenter)
            outer.addStretch(1)

            title_lbl = QLabel("Detect blip outliers")
            title_lbl.setObjectName("dustrack_blip_title")
            title_lbl.setAlignment(Qt.AlignCenter)
            outer.addWidget(title_lbl)

            # Inner content card -- no QWidget QSS so native spinboxes
            # keep their native rendering (per feedback_qt_qss_vs_palette).
            content = QWidget()
            content.setMaximumWidth(640)
            content_layout = QVBoxLayout(content)
            content_layout.setSpacing(12)

            # Active-layer context line.
            layer_name = getattr(ann, "name", None) or "<unnamed>"
            n_labels = len(getattr(ann, "labels", []) or [])
            n_frames = int(getattr(ann, "n_frames", 0) or 0)
            ctx_lbl = QLabel(
                f"Active layer: {layer_name!r}\n"
                f"  {n_labels} label{'s' if n_labels != 1 else ''} × {n_frames} frames"
            )
            content_layout.addWidget(ctx_lbl)

            # --- Detection-parameter spinboxes ---
            params_lbl = QLabel("Detection parameters")
            params_lbl.setStyleSheet("font-weight: bold;")
            content_layout.addWidget(params_lbl)

            # Spinbox ranges are generous on the upper end -- the
            # algorithm imposes no hard cap, and users iterating
            # against unusual data (very noisy traces, sustained
            # model failures) may legitimately want larger values
            # than the defaults suggest. Lower bounds are physically
            # meaningful: threshold_factor must be positive,
            # max_blip_length must be >=1 (single-frame blip),
            # return tolerance must be positive.
            self._threshold_spin = QDoubleSpinBox()
            self._threshold_spin.setRange(0.1, 1000.0)
            self._threshold_spin.setSingleStep(0.5)
            self._threshold_spin.setDecimals(1)
            self._threshold_spin.setValue(150.0)
            content_layout.addLayout(
                _labeled_row(
                    "Threshold factor",
                    self._threshold_spin,
                    tooltip=(
                        "How extreme a frame-to-frame jump has to be "
                        "to count as the start of a blip, in units of "
                        "robust sigma (1.4826 * MAD) above the per-"
                        "label median displacement. Higher = stricter "
                        "(fewer candidates flagged). The threshold is "
                        "computed per label, so high-motion and low-"
                        "motion labels get appropriately scaled cuts."
                    ),
                )
            )

            self._max_len_spin = QSpinBox()
            self._max_len_spin.setRange(1, 1000)
            self._max_len_spin.setValue(50)
            content_layout.addLayout(
                _labeled_row(
                    "Max blip length",
                    self._max_len_spin,
                    tooltip=(
                        "Maximum number of consecutive bad frames a "
                        "spike can span and still be bracketed as a "
                        "blip. Spikes whose forward scan doesn't find "
                        "a return-position anchor within this many "
                        "frames are skipped (counted as 'long' in the "
                        "results pane) -- they're usually sustained "
                        "model failures, not transient blips."
                    ),
                )
            )

            self._return_factor_spin = QDoubleSpinBox()
            self._return_factor_spin.setRange(0.1, 1000.0)
            self._return_factor_spin.setSingleStep(0.5)
            self._return_factor_spin.setDecimals(1)
            self._return_factor_spin.setValue(3.0)
            content_layout.addLayout(
                _labeled_row(
                    "Return tolerance",
                    self._return_factor_spin,
                    tooltip=(
                        "How close (in multiples of typical per-frame "
                        "displacement) the position after the spike "
                        "has to land near the position before the "
                        "spike for the run to count as a blip. The "
                        "tolerance scales with the run length, so a "
                        "5-frame spike is allowed to land further "
                        "from the pre-spike anchor than a 1-frame "
                        "spike. Higher = more permissive (more spikes "
                        "qualify as 'returning'); lower = stricter."
                    ),
                )
            )

            # Detect button (its own row, centered).
            detect_row = QHBoxLayout()
            detect_row.setAlignment(Qt.AlignCenter)
            self._detect_btn = QPushButton("Detect")
            self._detect_btn.setMinimumWidth(160)
            self._detect_btn.setStyleSheet(_ROLE_QSS["primary"])
            self._detect_btn.clicked.connect(self._on_detect_clicked)
            detect_row.addWidget(self._detect_btn)
            content_layout.addLayout(detect_row)

            # Results section: blank until first Detect.
            results_header = QLabel("Results")
            results_header.setStyleSheet("font-weight: bold;")
            content_layout.addWidget(results_header)
            self._results_lbl = QLabel(_format_blip_results_text(None))
            self._results_lbl.setObjectName("dustrack_blip_results")
            self._results_lbl.setWordWrap(False)
            self._results_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            content_layout.addWidget(self._results_lbl)

            # Output options
            options_header = QLabel("Output options")
            options_header.setStyleSheet("font-weight: bold;")
            content_layout.addWidget(options_header)
            self._drop_frame_chk = QCheckBox(
                "Drop all labels at frames where any blip was detected"
            )
            self._drop_frame_chk.setChecked(False)
            self._drop_frame_chk.setStyleSheet(_CHECKBOX_QSS)
            self._drop_frame_chk.setToolTip(
                "When unchecked (default): only the blipped label's entry "
                "is removed at each blip frame; other labels at the same "
                "frame are preserved. When checked: every label is "
                "dropped at any frame where any blip was detected -- "
                "useful when blips correlate with bad image quality "
                "(occlusion, motion blur) that compromises all labels at "
                "the same instant."
            )
            content_layout.addWidget(self._drop_frame_chk)

            # Cancel + Remove blips button row.
            button_row = QHBoxLayout()
            button_row.setAlignment(Qt.AlignCenter)
            self._cancel_btn = QPushButton("Cancel")
            self._cancel_btn.setMinimumWidth(160)
            self._cancel_btn.setStyleSheet(_ROLE_QSS["neutral"])
            self._cancel_btn.clicked.connect(self._on_cancel_clicked)
            self._apply_btn = QPushButton("Remove blips →")
            self._apply_btn.setMinimumWidth(160)
            self._apply_btn.setStyleSheet(_ROLE_QSS["primary"])
            self._apply_btn.setEnabled(False)
            self._apply_btn.clicked.connect(self._on_apply_clicked)
            button_row.addWidget(self._cancel_btn)
            button_row.addWidget(self._apply_btn)
            content_layout.addLayout(button_row)

            outer.addWidget(content, alignment=Qt.AlignCenter)
            outer.addStretch(1)

            main_window.installEventFilter(self)
            self._frame.show()
            self._reposition()
            self._frame.raise_()
            self._detect_btn.setFocus()

        # -- Slots -----------------------------------------------------

        def _current_knobs(self) -> dict:
            return dict(
                threshold_factor=float(self._threshold_spin.value()),
                max_blip_length=int(self._max_len_spin.value()),
                return_position_factor=float(self._return_factor_spin.value()),
            )

        def _drop_frame_state(self) -> bool:
            return bool(self._drop_frame_chk.isChecked())

        def _on_detect_clicked(self):
            knobs = self._current_knobs()
            try:
                self._last_report = self._detect_fn(self._ann, **knobs)
            except Exception as exc:  # noqa: BLE001
                # Surface the failure in the results pane rather than
                # tearing down the modal; detect is supposed to be
                # cheap + safe, so an exception here is a real signal
                # (e.g. ann has no labels). Disable Interpolate.
                self._last_report = None
                self._results_lbl.setText(f"Detection failed: {exc}")
                self._apply_btn.setEnabled(False)
                return
            self._results_lbl.setText(_format_blip_results_text(self._last_report))
            self._apply_btn.setEnabled(len(self._last_report) > 0)

        def _on_apply_clicked(self):
            if self._last_report is None or len(self._last_report) == 0:
                return  # button shouldn't have been clickable, but be defensive
            self._result = (
                self._last_report,
                self._current_knobs(),
                self._drop_frame_state(),
            )
            self._dismiss()
            self._loop.quit()

        def _on_cancel_clicked(self):
            self._result = None
            self._dismiss()
            self._loop.quit()

        # -- Lifecycle (mirror TrainingOptionsDialog) ------------------

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
            """Block until the user clicks Remove blips or Cancel.

            Returns ``(report, knobs_dict, drop_frame_if_any_blip)``
            on Remove blips; ``None``
            on Cancel.
            """
            self._loop.exec_()
            return self._result

    return BlipOptionsDialog


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
        QFileDialog,
        QFrame,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QPushButton,
        QVBoxLayout,
        QWidget,
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
                label = f"{b['name']}\n" f"  bodyparts: {bodyparts}\n" f"  {desc}"
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, b)
                self._list.addItem(item)
            self._list.setCurrentRow(0)
            self._list.itemDoubleClicked.connect(lambda _item: self._on_use_clicked())
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


# ---------------------------------------------------------------------
# 1.2.0a3: no-arg dustrack.open() welcome modal
# ---------------------------------------------------------------------


def _render_recent_session_label(session) -> str:
    """One-line label for a recent-sessions entry in the picker.

    Rendering rules (matches the spec captured during the 2026-05-22
    design conversation):

    - 1-element file -> the full path.
    - 1-element directory -> the full path + trailing slash
      (legacy ``recent_folders`` migration only; new code never
      writes a single-directory session).
    - N-element (N >= 2) -> ``<first>.<ext> + N-1 more`` plus the
      common parent folder when all entries share one.

    Extracted as a free function so the rendering can be tested
    without spinning up Qt.
    """
    first = Path(session[0])
    n = len(session)
    if n == 1:
        if first.is_dir():
            return f"{first}/"
        return str(first)
    # N-element session. Try to find a shared parent for context.
    paths = [Path(p) for p in session]
    try:
        common = os.path.commonpath([str(p) for p in paths])
        common_dir = Path(common)
        if common_dir.is_dir():
            return f"{first.name} + {n - 1} more  ({common_dir})"
    except (ValueError, OSError):
        pass
    return f"{first.name} + {n - 1} more"


def _make_open_video_overlay_class():
    """Lazy factory for the 1.2.0a3 no-arg :func:`dustrack.open`
    welcome modal.

    Mirrors :func:`_make_confirm_overlay_class`'s qtpy-import-on-demand
    pattern so importing ``dustrack`` stays Qt-free when only the
    library API is used. Returns a class that, when instantiated and
    ``exec_()``-ed, blocks on a synchronous Qt event loop and
    surfaces either:

    - ``list[Path]`` -- the user picked one or more videos (via
      Browse... or a recent-sessions row).
    - ``None`` -- the user dismissed the modal via the main window's
      close button (the spec deliberately omits a Quit button; the
      window X covers that path).
    """
    from qtpy.QtCore import QEvent, QEventLoop, QObject, Qt
    from qtpy.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QPushButton,
        QVBoxLayout,
    )

    _PRIMARY_QSS = (
        "QPushButton { background-color: #3a86ff; color: white; "
        "  border: 1px solid #2a76ef; padding: 8px 32px; "
        "  font-size: 12pt; font-weight: bold; }"
        "QPushButton:hover { background-color: #4a96ff; }"
        "QPushButton:pressed { background-color: #2a76ef; }"
        "QPushButton:disabled { background-color: #2a2a2a; "
        "  color: #777; border: 1px solid #333; }"
    )

    _HELP_NO_SELECTION = (
        "Pick a video file or a DLC config.yaml. " "Or click a recent session below."
    )
    _HELP_HAS_SELECTION = (
        "Click Load (or double-click the selected session) to open it."
    )

    class OpenVideoOverlay(QObject):
        """Welcome modal mounted on a freshly-constructed seed-mode
        DUSTrack window.

        Surface:

        - Title + subtitle ("Welcome to DUSTrack" / "Pick a video to
          get started").
        - Helpful message that flips with state -- "Pick a video or
          DLC config.yaml..." when nothing is selected, "Click Load
          to open..." when a recent row is selected.
        - One contextual primary button whose label flips:
            * No history selection -> "Open" -- pops the file dialog
              (filtered for videos + `config.yaml`); the dialog's own
              Open click commits immediately (no second click).
            * Recent row selected -> "Load" -- commits the selected
              row.
        - Secondary "Batch process..." button below the action button:
          opens the batch-process modal (convert-to-mono / build-TOC)
          on the same seed window without committing the welcome modal.
          Returns the sentinel string ``"batch_process"`` from
          :meth:`exec_` so the dispatcher in :mod:`dustrack._open` can
          loop the welcome modal back up after the batch run.
        - Recent-sessions list (max 20, most-recent first; hidden when
          empty). Single-click toggles a row's selection; clicking the
          same row again deselects. Double-click / Enter commits
          immediately.

        Same backdrop + parented-QFrame + event-filter + exec_-loop
        scaffolding as :class:`ConfirmOverlay`.
        """

        def __init__(self, main_window, *, recent_sessions):
            super().__init__(main_window)
            self._mw = main_window
            self._result = None  # type: list | None
            self._loop = QEventLoop()
            self._recent_sessions = list(recent_sessions or [])
            # Currently-selected recent index, or None if no recent
            # row is highlighted. Drives the action button's label
            # flip (Open <-> Load) and the help-line flip.
            self._selected_index = None  # type: int | None

            self._frame = QFrame(main_window)
            self._frame.setObjectName("dustrack_open_video_overlay")
            self._frame.setStyleSheet(
                "#dustrack_open_video_overlay { background-color: rgba(0, 0, 0, 200); }"
                "QLabel { color: white; }"
                "#dustrack_open_video_title { color: white; "
                "  font-size: 24pt; font-weight: bold; }"
                "#dustrack_open_video_subtitle { color: #cccccc; "
                "  font-size: 11pt; }"
                "#dustrack_open_video_recent_label { color: #cccccc; "
                "  font-size: 10pt; font-weight: bold; }"
                "#dustrack_open_video_help_label { color: #cccccc; "
                "  font-size: 10pt; }"
                "QListWidget { background-color: #1f1f1f; color: white; "
                "  font-size: 10pt; border: 1px solid #444; padding: 4px; }"
                "QListWidget::item { padding: 4px 8px; }"
                "QListWidget::item:hover { background-color: #2a2a2a; }"
                "QListWidget::item:selected { background-color: #3a86ff; "
                "  color: white; }"
            )
            self._frame.setFocusPolicy(Qt.StrongFocus)

            layout = QVBoxLayout(self._frame)
            layout.setAlignment(Qt.AlignCenter)

            # Top row: spacer + batch-process button in the corner. Sits
            # at the very top of the modal so it reads as unrelated to
            # the centered Open / Load action below.
            top_row = QHBoxLayout()
            top_row.setContentsMargins(16, 16, 16, 0)
            top_row.addStretch(1)
            self._batch_btn = QPushButton("Batch process...")
            # Faint blue-tint chrome: visible on the rgba(0,0,0,200)
            # backdrop without competing with the centered primary
            # Open / Load action. White text + #3a86ff accent border
            # carries the brand colour without filling like a primary.
            self._batch_btn.setStyleSheet(
                "QPushButton { background-color: rgba(58, 134, 255, 60); "
                "  color: white; border: 1px solid #3a86ff; "
                "  padding: 8px 20px; font-size: 11pt; "
                "  font-weight: bold; }"
                "QPushButton:hover { background-color: rgba(58, 134, 255, 120); "
                "  border-color: #5aa0ff; }"
                "QPushButton:pressed { background-color: rgba(58, 134, 255, 180); }"
            )
            self._batch_btn.clicked.connect(self._on_batch_clicked)
            top_row.addWidget(self._batch_btn)
            layout.addLayout(top_row)

            layout.addStretch(1)

            title_lbl = QLabel("Welcome to DUSTrack")
            title_lbl.setObjectName("dustrack_open_video_title")
            title_lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(title_lbl)

            subtitle = QLabel("Pick a video to get started")
            subtitle.setObjectName("dustrack_open_video_subtitle")
            subtitle.setAlignment(Qt.AlignCenter)
            layout.addWidget(subtitle)

            layout.addSpacing(20)

            # Contextual helper line above the action button.
            self._help_lbl = QLabel(_HELP_NO_SELECTION)
            self._help_lbl.setObjectName("dustrack_open_video_help_label")
            self._help_lbl.setAlignment(Qt.AlignCenter)
            self._help_lbl.setWordWrap(True)
            self._help_lbl.setMaximumWidth(720)
            layout.addWidget(self._help_lbl, alignment=Qt.AlignCenter)

            layout.addSpacing(10)

            # Single contextual action button: label flips with state.
            btn_row = QHBoxLayout()
            btn_row.setAlignment(Qt.AlignCenter)
            self._action_btn = QPushButton("Open")
            self._action_btn.setMinimumWidth(220)
            self._action_btn.setStyleSheet(_PRIMARY_QSS)
            self._action_btn.setDefault(True)
            self._action_btn.clicked.connect(self._on_action_clicked)
            btn_row.addWidget(self._action_btn)
            layout.addLayout(btn_row)

            # Recent-sessions list. Hidden entirely when empty so a
            # fresh-install launch shows just the title + button.
            self._recent_widget = None
            if self._recent_sessions:
                layout.addSpacing(24)
                recent_lbl = QLabel("Recent sessions")
                recent_lbl.setObjectName("dustrack_open_video_recent_label")
                recent_lbl.setAlignment(Qt.AlignCenter)
                layout.addWidget(recent_lbl)

                self._recent_widget = QListWidget()
                self._recent_widget.setFixedWidth(640)
                self._recent_widget.setMaximumHeight(280)
                for session in self._recent_sessions:
                    item = QListWidgetItem(_render_recent_session_label(session))
                    item.setData(Qt.UserRole, [str(p) for p in session])
                    self._recent_widget.addItem(item)
                # Single-click TOGGLES selection (click same row twice
                # to deselect); double-click / Enter COMMITS the row
                # in one gesture (muscle-memory shortcut). Selection
                # drives the action button's label flip.
                self._recent_widget.itemClicked.connect(self._on_recent_clicked)
                self._recent_widget.itemDoubleClicked.connect(
                    self._on_recent_activated,
                )
                self._recent_widget.itemActivated.connect(
                    self._on_recent_activated,
                )

                centered = QHBoxLayout()
                centered.addStretch(1)
                centered.addWidget(self._recent_widget)
                centered.addStretch(1)
                layout.addLayout(centered)

            layout.addStretch(1)

            main_window.installEventFilter(self)

            self._frame.show()
            self._reposition()
            self._frame.raise_()
            # Focus the action button so Enter triggers the dominant
            # action (Open / Load depending on state).
            self._action_btn.setFocus()

        def eventFilter(self, obj, event):  # noqa: N802 (Qt API)
            if obj is self._mw:
                t = event.type()
                if t == QEvent.Resize:
                    self._reposition()
                elif t == QEvent.Close:
                    # Window X during the modal -> abort with None.
                    # We don't ``event.ignore()`` here; the close
                    # event proceeds to the original closeEvent (the
                    # save-on-close guard short-circuits because the
                    # seed session has nothing to save and the
                    # history write short-circuits on
                    # _is_seed_session).
                    self._result = None
                    self._dismiss_no_filter_unhook()
                    if self._loop.isRunning():
                        self._loop.quit()
            return False

        def _reposition(self):
            self._frame.setGeometry(0, 0, self._mw.width(), self._mw.height())
            self._frame.raise_()

        def _refresh_action_state(self):
            """Update the action button's label + help text to reflect
            ``_selected_index``. Called after every recent-row toggle.
            """
            if self._selected_index is None:
                self._action_btn.setText("Open")
                self._help_lbl.setText(_HELP_NO_SELECTION)
            else:
                self._action_btn.setText("Load")
                self._help_lbl.setText(_HELP_HAS_SELECTION)

        def _select_recent_row(self, index):
            """Mark a recent row as the current selection (and
            highlight it in the list widget). ``None`` clears the
            selection."""
            self._selected_index = index
            widget = self._recent_widget
            if widget is None:
                self._refresh_action_state()
                return
            if index is None:
                widget.clearSelection()
            else:
                widget.setCurrentRow(index)
            self._refresh_action_state()

        def _commit_recent(self, index):
            """Commit the recent row at ``index`` and exit the modal
            loop. Used by both the action button's Load mode and the
            double-click / Enter shortcut.
            """
            if index is None or not (0 <= index < len(self._recent_sessions)):
                return
            session = self._recent_sessions[index]
            self._result = [Path(p) for p in session]
            self._dismiss()
            self._loop.quit()

        def _commit_picked(self, picked):
            """Commit a fresh file-dialog pick and exit the modal
            loop. The dialog's own Open click is the commit gesture;
            no second button click required."""
            if not picked:
                return
            self._result = [Path(p) for p in picked]
            self._dismiss()
            self._loop.quit()

        def _on_action_clicked(self):
            """The single action button: Open (no selection) or Load
            (recent row selected). Dispatch on state."""
            if self._selected_index is not None:
                self._commit_recent(self._selected_index)
                return
            picked = _prompt_for_videos(parent=self._frame)
            if picked is None:
                # QFileDialog cancel -> stay in the modal so the user
                # can try again or pick from the recent list.
                return
            self._commit_picked(picked)

        def _on_batch_clicked(self):
            """Route to the batch-process modal. Commits the sentinel
            string ``"batch_process"`` and exits the welcome modal's
            loop; the dispatcher in :mod:`dustrack._open` recognises
            the sentinel, opens the batch modal, and re-shows the
            welcome modal afterwards."""
            self._result = "batch_process"
            self._dismiss()
            self._loop.quit()

        def _on_recent_clicked(self, item):
            """Single-click on a recent row: toggle selection. Clicking
            an unselected row selects it; clicking the same selected
            row deselects."""
            widget = self._recent_widget
            if widget is None:
                return
            row = widget.row(item)
            if row < 0:
                return
            if self._selected_index == row:
                self._select_recent_row(None)
            else:
                self._select_recent_row(row)

        def _on_recent_activated(self, item):
            """Double-click / Enter on a recent row -- commits
            immediately (muscle-memory shortcut for power users).

            Reads the path list straight from the item's
            ``Qt.UserRole`` data so the commit doesn't depend on the
            widget's row-index lookup (which can return -1 under
            event-queue contention).
            """
            data = item.data(Qt.UserRole)
            if not data:
                return
            self._result = [Path(p) for p in data]
            self._dismiss()
            self._loop.quit()

        def _dismiss_no_filter_unhook(self):
            """Variant of :meth:`_dismiss` skipping the event-filter
            unhook -- used from the eventFilter itself, where
            removeEventFilter mid-dispatch is undefined behavior.
            """
            self._frame.hide()
            self._frame.deleteLater()

        def _dismiss(self):
            try:
                self._mw.removeEventFilter(self)
            except Exception:  # noqa: BLE001
                pass
            self._frame.hide()
            self._frame.deleteLater()

        def exec_(self):
            """Block until the user picks, hits Batch process, or
            dismisses.

            Returns:
                ``list[Path]`` if the user picked a video / recent
                session; ``"batch_process"`` (sentinel string) if the
                user clicked the secondary Batch button; ``None`` if
                the main-window close button was used.
            """
            self._loop.exec_()
            return self._result

    return OpenVideoOverlay


def _show_first_paint_notice(tracker) -> None:
    """One-shot modal asking the user to dismiss the dialog (or alt-tab
    away and back, or click any sidebar dropdown) to force a paint of
    the trace canvas.

    Works around the multi-video ``draw_idle`` delivery failure
    documented in ``feedback_mpl_canvas_warmup``: in multi-video Qt
    sessions the matplotlib trace canvas's deferred paint events
    aren't reliably delivered, so the trace pane can stay stale on
    first load until some external Qt event (window resize, modal
    dismiss, dropdown popup close, alt-tab) drains the paint queue.
    Clicking OK on this modal IS that external event -- the dismiss
    itself delivers a paintEvent and the trace refreshes.

    No-op on single-video sessions (no bug there) and when no Qt
    window is available (mpl-fallback / headless).
    """
    if os.environ.get("DUSTRACK_SUPPRESS_FIRST_PAINT_NOTICE"):
        return
    bundles = getattr(tracker, "_bundles", None)
    if not bundles or len(bundles) < 2:
        return
    qt_window = None
    try:
        qt_window = tracker._find_qt_window()
    except Exception:  # noqa: BLE001
        return
    if qt_window is None:
        return
    try:
        ConfirmOverlay = _make_confirm_overlay_class()
    except Exception:  # noqa: BLE001
        return
    try:
        ConfirmOverlay(
            qt_window,
            title="Click OK to finish loading",
            message=(
                "Click OK to start your multi-video session"
                # "Multi-video sessions need one Qt event to deliver the "
                # "first repaint of the trace canvas (technical detail: "
                # "mpl's deferred-draw chain is unreliable across multiple "
                # "matplotlib canvases on the same window).\n\n"
                # "Click OK -- the dismiss itself triggers the repaint. "
                # "Alternative gestures that also work: alt-tab away and "
                # "back, or click any sidebar dropdown.\n\n"
                # "One-time per session."
            ),
            buttons=[("OK", "primary")],
            default="OK",
            severity="info",
        ).exec_()
    except Exception:  # noqa: BLE001
        # Defensive: a failed notice should not block the session.
        pass


# ---------------------------------------------------------------------
# Video picker -- used by OpenVideoOverlay and also by dustrack.open()
# ---------------------------------------------------------------------
# Extensions surfaced as the "Videos" filter group in the no-arg
# :func:`open` picker. Anything Qt-renderable beyond this list is still
# reachable via the "All files" fallback. Kept conservative so users
# don't accidentally pick an audio/image asset and crash on construction.
_VIDEO_PICKER_EXTENSIONS = ("mp4", "avi", "mov", "mkv", "mts", "m4v", "wmv", "webm")


def _prompt_for_videos(parent=None):
    """Pop a Qt file-picker for one-or-more video files (or a DLC
    ``config.yaml``).

    Bootstraps a ``QApplication`` if one isn't already running (same
    pattern as :func:`_pin_qt_palette`), so this works as the very first
    Qt call in a fresh process. The picker opens at the last folder
    a video was picked from this machine (via
    :func:`dustrack._config.get_last_video_picker_dir`), or at the OS
    default on a fresh install.

    Three filter rows:

    - **Videos + DLC config** (default) -- the union; lets the user
      pick a `config.yaml` *or* one or more video files from the same
      dialog. Multi-select preserves click order (matters for the
      multi-video list-form dispatch where the user wants a specific
      order different from `config['video_sets']`).
    - **Videos only** -- restricts to recognised video extensions.
    - **All files** -- escape hatch.

    Returns:

    - ``list[Path]`` -- user picked one or more files (order preserved
      as the user clicked them). Folder-picker support is not exposed
      via this helper today; folders enter the modal via the recent-
      sessions list.
    - ``None`` -- user cancelled, OR qtpy isn't importable in the env
      (mpl-only install path); caller falls back to a no-op.
    """
    try:
        from qtpy.QtWidgets import QApplication, QFileDialog
    except ImportError:
        return None
    _ = QApplication.instance() or QApplication([])
    exts = " ".join(f"*.{e}" for e in _VIDEO_PICKER_EXTENSIONS)
    file_filter = (
        f"Videos and DLC config ({exts} config.yaml);;"
        f"Videos ({exts});;"
        "DLC config (config.yaml);;"
        "All files (*.*)"
    )
    start_dir = _config.get_last_video_picker_dir()
    paths, _selected_filter = QFileDialog.getOpenFileNames(
        parent,
        "Open video(s) or DLC config for DUSTrack",
        str(start_dir) if start_dir is not None else "",
        file_filter,
    )
    if not paths:
        return None
    return [Path(p) for p in paths]


def _make_decimate_gallery_class():
    """Build the cluster-per-row diverse-frame review modal lazily (qtpy on demand).

    Frames are embedded + clustered off-thread *before* this opens. Each
    appearance cluster is one row: its canonical (medoid) shown large, the
    diverse members kept at the current minimum feature-distance beside it, and
    a keep checkbox (default on). One knob -- the minimum distance between kept
    frames -- drives how many survive (data-driven; per-cluster counts vary with
    each cluster's spread), re-selecting over the cached farthest-point order
    with no re-embed. A member strip next to its canonical is also the "is the
    clustering working?" check. Unchecking a row drops that whole appearance
    group. ``exec_()`` returns the kept frame indices on confirm, ``None`` on
    cancel.
    """
    import numpy as np
    from qtpy.QtCore import QEvent, QEventLoop, QObject, Qt
    from qtpy.QtGui import QColor, QImage, QPainter, QPen, QPixmap
    from qtpy.QtWidgets import (
        QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
        QScrollArea, QSlider, QVBoxLayout, QWidget,
    )

    _PRIMARY = (
        "QPushButton { background-color: #3a86ff; color: white; "
        "border: 1px solid #2a76ef; padding: 6px 24px; font-size: 11pt; "
        "font-weight: bold; } QPushButton:hover { background-color: #4a96ff; }"
    )
    _NEUTRAL = (
        "QPushButton { background-color: #4a4a4a; color: white; "
        "border: 1px solid #3a3a3a; padding: 6px 24px; font-size: 11pt; } "
        "QPushButton:hover { background-color: #5a5a5a; }"
    )
    _CANON, _MEMBER = 200, 72
    _MEMBER_COLS, _MEMBER_ROWS = 7, 2
    _MEMBER_CAP = _MEMBER_COLS * _MEMBER_ROWS
    _ROW_H, _GUTTER_W = 208, 140

    def _to_pixmap(arr, size):
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.uint8))
        h, w = a.shape[:2]
        if a.ndim == 2:
            img = QImage(a.data, w, h, w, QImage.Format_Grayscale8)
        else:
            img = QImage(a.data, w, h, 3 * w, QImage.Format_RGB888)
        return QPixmap.fromImage(img.copy()).scaled(
            size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    class _DendrogramGutter(QWidget):
        """Paints the hierarchical tree over the cluster rows: leaves at the
        right edge (each aligned to a row centre), the root at the left, so
        sibling appearance groups are visibly joined low and dissimilar ones
        only merge near the root. Fed ``segments`` in (leaf_position, height01)
        space by :meth:`set_tree`."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self._segments = []
            self._n = 0
            self.setFixedWidth(_GUTTER_W)

        def set_tree(self, segments, n_leaves):
            self._segments = segments
            self._n = int(n_leaves)
            self.setFixedHeight(max(1, self._n) * _ROW_H)
            self.update()

        def paintEvent(self, _event):  # noqa: N802 (Qt API)
            if not self._segments or self._n < 2:
                return
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing, True)
            pen = QPen(QColor("#9fb3c8"))
            pen.setWidthF(1.4)
            p.setPen(pen)
            W = self.width()
            pad_l, pad_r = 10, 8
            usable = max(1, W - pad_l - pad_r)

            def x_of(h01):                     # 0 = leaf (right), 1 = root (left)
                return (W - pad_r) - h01 * usable

            def y_of(leafpos):                 # 0..n-1 -> row centre
                return leafpos * _ROW_H + _ROW_H / 2.0

            for seg in self._segments:
                pts = [(x_of(h), y_of(lp)) for (lp, h) in seg]
                for a, b in zip(pts, pts[1:]):
                    p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
            p.end()

    class DecimateGalleryDialog(QObject):
        def __init__(self, main_window, *, thumbs, recluster_fn, select_fn,
                     k_lo, k_hi, k_default, dmin_lo, dmin_hi, dmin_default,
                     n_frames):
            super().__init__(main_window)
            self._mw = main_window
            self._thumbs = thumbs
            self._recluster_fn = recluster_fn
            self._select_fn = select_fn
            self._lo, self._hi = float(dmin_lo), float(dmin_hi)
            self._n_frames = int(n_frames)
            self._labels = np.zeros(0, dtype=int)
            self._medoids = {}             # cluster -> frame index
            self._leaf_order = []          # cluster ids in tree (row) order
            self._selected = []            # global indices at the current d_min
            self._by_cluster = {}          # cluster -> [selected members, minus medoid]
            self._rows = {}                # cluster -> {frame,check,members,host,count}
            self._result = None
            self._loop = QEventLoop()

            self._frame = QFrame(main_window)
            self._frame.setObjectName("dustrack_decimate_overlay")
            self._frame.setStyleSheet(
                "#dustrack_decimate_overlay { background-color: rgba(0,0,0,205); }"
                "QLabel { color: white; }"
                "#dustrack_decimate_title { color: white; font-size: 22pt; "
                "  font-weight: bold; }"
                "QCheckBox::indicator { width: 18px; height: 18px; }"
                "QCheckBox::indicator:unchecked { background-color: transparent; "
                "  border: 2px solid white; border-radius: 3px; }"
                "QCheckBox::indicator:checked { background-color: #3a86ff; "
                "  border: 2px solid white; border-radius: 3px; }"
            )
            self._frame.setFocusPolicy(Qt.StrongFocus)

            outer = QVBoxLayout(self._frame)
            outer.setContentsMargins(40, 18, 40, 18)

            title = QLabel("Select diverse frames for training")
            title.setObjectName("dustrack_decimate_title")
            title.setAlignment(Qt.AlignCenter)
            outer.addWidget(title)

            # Knob 1: number of clusters (re-cuts the hierarchy; rebuilds rows).
            krow = QHBoxLayout()
            krow.setAlignment(Qt.AlignCenter)
            krow.addWidget(QLabel("Clusters"))
            self._k_slider = QSlider(Qt.Horizontal)
            self._k_slider.setRange(int(k_lo), int(k_hi))
            self._k_slider.setValue(int(k_default))
            self._k_slider.setFixedWidth(260)
            self._k_label = QLabel(str(int(k_default)))
            self._k_slider.valueChanged.connect(
                lambda v: self._k_label.setText(str(int(v))))
            self._k_slider.sliderReleased.connect(
                lambda: self._recluster(int(self._k_slider.value())))
            krow.addWidget(self._k_slider)
            krow.addWidget(self._k_label)
            outer.addLayout(krow)

            # Knob 2: minimum feature distance between kept frames. Left = smaller
            # distance = more (closer) frames; right = fewer.
            controls = QHBoxLayout()
            controls.setAlignment(Qt.AlignCenter)
            controls.addWidget(QLabel("← more frames"))
            self._slider = QSlider(Qt.Horizontal)
            self._slider.setRange(0, 1000)
            self._slider.setFixedWidth(380)
            self._slider.setValue(self._dmin_to_pos(float(dmin_default)))
            self._slider.valueChanged.connect(self._on_slider_moved)
            self._slider.sliderReleased.connect(self._render_members)
            controls.addWidget(self._slider)
            controls.addWidget(QLabel("fewer →"))
            outer.addLayout(controls)

            self._status = QLabel("")
            self._status.setAlignment(Qt.AlignCenter)
            outer.addWidget(self._status)

            self._scroll = QScrollArea()
            self._scroll.setWidgetResizable(True)
            self._scroll.setStyleSheet(
                "QScrollArea { border: none; background: transparent; }")
            body = QWidget()
            body_row = QHBoxLayout(body)
            body_row.setContentsMargins(0, 0, 0, 0)
            body_row.setSpacing(0)
            body_row.setAlignment(Qt.AlignTop)
            self._gutter = _DendrogramGutter()
            body_row.addWidget(self._gutter, alignment=Qt.AlignTop)
            self._rows_host = QWidget()
            self._rows_col = QVBoxLayout(self._rows_host)
            self._rows_col.setContentsMargins(0, 0, 0, 0)
            self._rows_col.setSpacing(0)
            self._rows_col.setAlignment(Qt.AlignTop)
            body_row.addWidget(self._rows_host, stretch=1)
            self._scroll.setWidget(body)
            outer.addWidget(self._scroll, stretch=1)

            btns = QHBoxLayout()
            btns.setAlignment(Qt.AlignCenter)
            self._cancel_btn = QPushButton("Cancel")
            self._cancel_btn.setStyleSheet(_NEUTRAL)
            self._cancel_btn.clicked.connect(self._on_cancel)
            self._confirm_btn = QPushButton("Create pruned layer →")
            self._confirm_btn.setStyleSheet(_PRIMARY)
            self._confirm_btn.clicked.connect(self._on_confirm)
            btns.addWidget(self._cancel_btn)
            btns.addWidget(self._confirm_btn)
            outer.addLayout(btns)

            main_window.installEventFilter(self)
            self._frame.show()
            self._reposition()
            self._frame.raise_()
            self._recluster(int(k_default))          # build rows + tree + select

        # -- knob geometry (slider position <-> minimum distance) ----------

        def _pos_to_dmin(self, pos):
            return self._lo + (self._hi - self._lo) * (pos / 1000.0)

        def _dmin_to_pos(self, d):
            if self._hi <= self._lo:
                return 0
            frac = (d - self._lo) / (self._hi - self._lo)
            return int(round(1000 * min(1.0, max(0.0, frac))))

        # -- clustering (re-cut the hierarchy at k; rebuild rows + tree) ----

        def _recluster(self, k):
            labels, medoids, leaf_order, segments = self._recluster_fn(k)
            self._labels = np.asarray(labels)
            self._medoids = {int(c): int(i) for c, i in medoids.items()}
            self._leaf_order = [int(c) for c in leaf_order]

            while self._rows_col.count():
                w = self._rows_col.takeAt(0).widget()
                if w is not None:
                    w.deleteLater()
            self._rows = {}
            for c in self._leaf_order:
                self._rows[c] = self._make_row(c)
                self._rows_col.addWidget(self._rows[c]["frame"])
            self._gutter.set_tree(segments, len(self._leaf_order))
            self._render_members()

        def _make_row(self, c):
            row = QFrame()
            row.setObjectName("decimate_row")
            row.setFixedHeight(_ROW_H)
            h = QHBoxLayout(row)
            h.setContentsMargins(8, 2, 8, 2)
            h.setSpacing(10)

            chk = QCheckBox()
            chk.setChecked(True)
            chk.toggled.connect(lambda _checked, cc=c: self._on_row_toggled(cc))
            h.addWidget(chk, alignment=Qt.AlignVCenter)

            canon = QLabel()
            canon.setPixmap(_to_pixmap(self._thumbs[self._medoids[c]], _CANON))
            canon.setToolTip(f"cluster {c} canonical (nearest the cluster centre)")
            h.addWidget(canon, alignment=Qt.AlignVCenter)

            count = QLabel("")
            count.setMinimumWidth(60)
            count.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            h.addWidget(count)

            host = QWidget()
            grid = QGridLayout(host)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setSpacing(3)
            grid.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            h.addWidget(host, stretch=1)

            info = dict(frame=row, check=chk, members=grid, host=host, count=count)
            self._style_row_frame(row, True)
            return info

        # -- selection (cheap: threshold the cached FPS order) -------------

        def _recompute(self):
            d = self._pos_to_dmin(self._slider.value())
            try:
                self._selected = [int(i) for i in self._select_fn(d)]
            except Exception as exc:  # noqa: BLE001
                self._status.setText(f"Selection failed: {exc}")
                self._selected = list(self._medoids.values())
            sel_by_c = {c: [] for c in self._medoids}
            for i in self._selected:
                c = int(self._labels[i])
                if c in sel_by_c and i != self._medoids[c]:
                    sel_by_c[c].append(i)
            self._by_cluster = sel_by_c
            self._update_status(d)

        def _update_status(self, d=None):
            if d is None:
                d = self._pos_to_dmin(self._slider.value())
            kept = sum(1 + len(m) for c, m in self._by_cluster.items()
                       if self._rows[c]["check"].isChecked())
            dropped = sum(1 for c in self._medoids
                          if not self._rows[c]["check"].isChecked())
            drop_txt = f"  ·  {dropped} cluster(s) dropped" if dropped else ""
            self._status.setText(
                f"{len(self._medoids)} clusters  ·  {kept} of {self._n_frames} "
                f"frames selected  ·  min distance {d:.3f}{drop_txt}")

        def _on_slider_moved(self, _value):
            # Live count on drag; thumbs redraw on release (cheap vs pixmaps).
            self._recompute()

        def _render_members(self):
            self._recompute()
            for c, info in self._rows.items():
                grid = info["members"]
                while grid.count():
                    w = grid.takeAt(0).widget()
                    if w is not None:
                        w.deleteLater()
                members = self._by_cluster.get(c, [])
                for k, i in enumerate(members[:_MEMBER_CAP]):
                    lbl = QLabel()
                    lbl.setPixmap(_to_pixmap(self._thumbs[int(i)], _MEMBER))
                    grid.addWidget(lbl, k // _MEMBER_COLS, k % _MEMBER_COLS)
                if len(members) > _MEMBER_CAP:
                    more = QLabel(f"+{len(members) - _MEMBER_CAP}")
                    more.setStyleSheet("color: #9fb3c8; font-size: 10pt;")
                    grid.addWidget(more, _MEMBER_ROWS - 1, _MEMBER_COLS)
                info["count"].setText(f"{1 + len(members)} kept")

        # -- keep/drop a whole appearance group ----------------------------

        def _style_row_frame(self, row, kept):
            row.setStyleSheet(
                "#decimate_row { background-color: rgba(255,255,255,%d); "
                "border-radius: 6px; }" % (16 if kept else 4))

        def _style_row(self, c):
            kept = self._rows[c]["check"].isChecked()
            self._style_row_frame(self._rows[c]["frame"], kept)
            self._rows[c]["host"].setEnabled(kept)
            self._rows[c]["count"].setEnabled(kept)

        def _on_row_toggled(self, c):
            self._style_row(c)
            self._update_status()

        # -- lifecycle -----------------------------------------------------

        def _on_confirm(self):
            keep = []
            for c, members in self._by_cluster.items():
                if self._rows[c]["check"].isChecked():
                    keep.append(self._medoids[c])
                    keep.extend(members)
            self._result = sorted({int(i) for i in keep})
            self._dismiss()
            self._loop.quit()

        def _on_cancel(self):
            self._result = None
            self._dismiss()
            self._loop.quit()

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

    return DecimateGalleryDialog


def _make_source_selection_class():
    """Build the diverse-selection *source* modal lazily (qtpy on demand).

    Pops up before the embed pass to pick WHAT is read. Scope is the primary
    choice: **this video** (one annotation layer -- the current one, or one
    picked from a dropdown) or **across all videos** (a union of project-wide
    sources chosen with checkboxes: labeled data, and -- soon -- blips +
    low-confidence frames from the autorefine machinery). ``exec_()`` returns a
    dict on Continue -- ``{"scope": "video", "layer": name}`` or
    ``{"scope": "across", "sources": [...]}`` -- or ``None`` on Cancel.
    """
    from qtpy.QtCore import QEvent, QEventLoop, QObject, Qt
    from qtpy.QtWidgets import (
        QButtonGroup, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel,
        QPushButton, QRadioButton, QVBoxLayout, QWidget,
    )

    _PRIMARY = (
        "QPushButton { background-color: #3a86ff; color: white; "
        "border: 1px solid #2a76ef; padding: 6px 24px; font-size: 11pt; "
        "font-weight: bold; } QPushButton:hover { background-color: #4a96ff; }"
    )
    _NEUTRAL = (
        "QPushButton { background-color: #4a4a4a; color: white; "
        "border: 1px solid #3a3a3a; padding: 6px 24px; font-size: 11pt; }"
    )

    class SourceSelectionDialog(QObject):
        def __init__(self, main_window, *, layer_names, current_layer, has_project):
            super().__init__(main_window)
            self._mw = main_window
            self._current_layer = current_layer
            self._result = None
            self._loop = QEventLoop()

            self._frame = QFrame(main_window)
            self._frame.setObjectName("dustrack_source_overlay")
            self._frame.setStyleSheet(
                "#dustrack_source_overlay { background-color: rgba(0,0,0,205); }"
                "QLabel { color: white; font-size: 11pt; }"
                "#dustrack_source_title { color: white; font-size: 20pt; "
                "  font-weight: bold; }"
                "#dustrack_section { color: #9fb3c8; font-size: 10pt; "
                "  font-weight: bold; }"
                "QComboBox { font-size: 11pt; }"
                "QRadioButton { color: white; font-size: 11pt; }"
                "QCheckBox { color: white; font-size: 11pt; }"
                # Native checked mark is invisible on this dark overlay; render
                # a white-bordered indicator filled primary-blue when checked
                # (circle for radios, square for checkboxes).
                "QRadioButton::indicator { width: 14px; height: 14px; }"
                "QRadioButton::indicator:unchecked { "
                "  background-color: transparent; border: 2px solid white; "
                "  border-radius: 8px; }"
                "QRadioButton::indicator:checked { "
                "  background-color: #3a86ff; border: 2px solid white; "
                "  border-radius: 8px; }"
                "QRadioButton:disabled, QCheckBox:disabled { color: #777777; }"
                "QRadioButton::indicator:disabled { "
                "  background-color: transparent; border: 2px solid #666666; "
                "  border-radius: 8px; }"
                "QRadioButton::indicator:checked:disabled { "
                "  background-color: #4a5a7a; border: 2px solid #666666; "
                "  border-radius: 8px; }"
                "QCheckBox::indicator { width: 15px; height: 15px; }"
                "QCheckBox::indicator:unchecked { "
                "  background-color: transparent; border: 2px solid white; "
                "  border-radius: 3px; }"
                "QCheckBox::indicator:checked { "
                "  background-color: #3a86ff; border: 2px solid white; "
                "  border-radius: 3px; }"
                "QCheckBox::indicator:disabled { "
                "  background-color: transparent; border: 2px solid #666666; "
                "  border-radius: 3px; }")
            self._frame.setFocusPolicy(Qt.StrongFocus)

            outer = QVBoxLayout(self._frame)
            outer.setAlignment(Qt.AlignCenter)
            outer.addStretch(1)
            title = QLabel("Diverse selection — source")
            title.setObjectName("dustrack_source_title")
            title.setAlignment(Qt.AlignCenter)
            outer.addWidget(title)

            content = QWidget()
            content.setMaximumWidth(580)
            col = QVBoxLayout(content)
            col.setSpacing(8)

            # 1) SCOPE -- the primary decision: it governs which sources are
            #    even available (this-video reads one layer; across-videos reads
            #    a union of project-wide sources).
            scope_lbl = QLabel("SCOPE")
            scope_lbl.setObjectName("dustrack_section")
            col.addWidget(scope_lbl)
            self._scope_group = QButtonGroup(self._frame)
            self._this_radio = QRadioButton("This video")
            self._this_radio.setChecked(True)
            self._this_radio.toggled.connect(self._on_scope_changed)
            self._scope_group.addButton(self._this_radio)
            col.addWidget(self._this_radio)
            self._across_radio = QRadioButton("Across all videos in the project")
            self._across_radio.toggled.connect(self._on_scope_changed)
            self._scope_group.addButton(self._across_radio)
            col.addWidget(self._across_radio)

            col.addSpacing(12)

            self._read_lbl = QLabel("READ FROM  ·  this video")
            self._read_lbl.setObjectName("dustrack_section")
            col.addWidget(self._read_lbl)

            # 2a) this-video source: one annotation layer (current, or picked).
            self._video_src = QWidget()
            vcol = QVBoxLayout(self._video_src)
            vcol.setContentsMargins(0, 0, 0, 0)
            vcol.setSpacing(6)
            self._layer_group = QButtonGroup(self._frame)
            self._current_radio = QRadioButton(f"Current layer  ({current_layer})")
            self._current_radio.setChecked(True)
            self._layer_group.addButton(self._current_radio)
            vcol.addWidget(self._current_radio)
            pick_row = QHBoxLayout()
            self._pick_radio = QRadioButton("Layer:")
            self._layer_group.addButton(self._pick_radio)
            self._combo = QComboBox()
            self._combo.addItems(list(layer_names))
            self._combo.setEnabled(False)
            self._pick_radio.toggled.connect(self._combo.setEnabled)
            pick_row.addWidget(self._pick_radio)
            pick_row.addWidget(self._combo, stretch=1)
            vcol.addLayout(pick_row)
            col.addWidget(self._video_src)

            # 2b) across-videos source: a union of project-wide sources. Labeled
            #     data needs a DLC project; blips + low-confidence come from the
            #     autorefine machinery and aren't wired yet.
            self._across_src = QWidget()
            acol = QVBoxLayout(self._across_src)
            acol.setContentsMargins(0, 0, 0, 0)
            acol.setSpacing(6)
            self._src_labeled = QCheckBox("Labeled data — every video in the project")
            self._src_labeled.setChecked(has_project)
            self._src_labeled.setEnabled(has_project)
            if not has_project:
                self._src_labeled.setToolTip(
                    "Needs a DLC project — labeled data lives under its "
                    "labeled-data/ folder. Create/open a project first.")
            acol.addWidget(self._src_labeled)
            self._src_blips = QCheckBox("Blips (confident-wrong frames) — coming soon")
            self._src_blips.setEnabled(False)
            self._src_blips.setToolTip("From the autorefine blip detector — not "
                                       "wired into this picker yet.")
            acol.addWidget(self._src_blips)
            self._src_lowconf = QCheckBox("Low-confidence frames — coming soon")
            self._src_lowconf.setEnabled(False)
            self._src_lowconf.setToolTip("From the autorefine likelihood "
                                         "detector — not wired in yet.")
            acol.addWidget(self._src_lowconf)
            self._across_src.setVisible(False)
            col.addWidget(self._across_src)

            btn_row = QHBoxLayout()
            btn_row.setAlignment(Qt.AlignCenter)
            cancel = QPushButton("Cancel")
            cancel.setStyleSheet(_NEUTRAL)
            cancel.clicked.connect(self._on_cancel)
            cont = QPushButton("Continue →")
            cont.setStyleSheet(_PRIMARY)
            cont.clicked.connect(self._on_ok)
            btn_row.addWidget(cancel)
            btn_row.addWidget(cont)
            col.addLayout(btn_row)

            outer.addWidget(content, alignment=Qt.AlignCenter)
            outer.addStretch(1)

            self._on_scope_changed()          # establish initial section visibility

            main_window.installEventFilter(self)
            self._frame.show()
            self._reposition()
            self._frame.raise_()

        def _on_scope_changed(self, *_):
            across = self._across_radio.isChecked()
            self._read_lbl.setText("READ FROM  ·  across all videos"
                                   if across else "READ FROM  ·  this video")
            self._video_src.setVisible(not across)
            self._across_src.setVisible(across)

        def _on_ok(self):
            if self._across_radio.isChecked():
                sources = []
                if self._src_labeled.isChecked():
                    sources.append("labeled_data")
                self._result = {"scope": "across", "sources": sources}
            else:
                layer = (self._current_layer if self._current_radio.isChecked()
                         else self._combo.currentText())
                self._result = {"scope": "video", "layer": layer}
            self._dismiss()
            self._loop.quit()

        def _on_cancel(self):
            self._result = None
            self._dismiss()
            self._loop.quit()

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

    return SourceSelectionDialog
