"""Batch-process modal -- Qt overlay wrapping ``dustrack.batch.*``.

Surfaces :func:`dustrack.convert_to_mono` and :func:`dustrack.build_toc`
behind a single click-driven UI so users on the no-CLI path can warm a
folder (or DLC project) before annotation. Mounted as an overlay on the
seed-mode DUSTrack window from the welcome modal's "Batch process..."
button, and also reachable from the main window's Tools menu while a
real session is open.

The actual batch work runs on a :class:`QThread` worker; per-file
progress flows back via Qt signals. The batch.py primitives accept
``progress_callback`` and ``cancel_check`` kwargs so this modal can
drive them without re-implementing the file-walking logic.

The dispatch logic that decides which batch ops to call is extracted
into :func:`run_batch_jobs` so it's testable without spinning up Qt.

Mirrors :func:`dustrack._overlays._make_open_video_overlay_class`'s
qtpy-import-on-demand pattern so importing ``dustrack`` stays Qt-free
when only the library API is used.
"""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

from . import batch as _batch


# Status strings the modal renders as user-facing tags. Anything not in
# this map (e.g. ``"error: ..."`` from dnav) is shown verbatim.
_STATUS_LABELS = {
    "ok": "converted",
    "skip_missing": "skip (missing)",
    "skip_overwrite": "skip (would overwrite)",
    "skip_existing": "skip (mono exists)",
    "skip_already_mono": "skip (already mono)",
    "failed": "FAILED",
    "hit": "TOC hit",
    "built": "TOC built",
    "built (uncached)": "TOC built (uncached)",
}


@dataclass
class BatchJobSpec:
    """What the user picked in the modal's setup view.

    ``source`` is a folder of videos or a single video file -- the same
    shapes :func:`dustrack.convert_to_mono` and :func:`dustrack.build_toc`
    accept. DLC projects are not a special case: point the picker at
    ``<project>/videos`` to TOC the in-project copies (use
    ``recursive=False`` semantics implicitly by picking the leaf
    folder).
    """

    source: Path
    convert_to_mono: bool = True
    build_toc: bool = True


@dataclass
class BatchRunResults:
    """What the worker returns when it's done (or cancelled)."""

    converted: list[Path] = field(default_factory=list)
    toc_results: dict = field(default_factory=dict)
    cancelled: bool = False
    error: Optional[str] = None


def run_batch_jobs(
    spec: BatchJobSpec,
    *,
    progress_callback: Optional[Callable[[str, int, int, Path, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> BatchRunResults:
    """Dispatch a :class:`BatchJobSpec` to the underlying batch helpers.

    The callback receives ``(phase, idx, total, path, status)`` where
    ``phase`` is ``"convert"`` or ``"toc"``; the modal renders the phase
    tag in its current-file label. Errors from a phase are captured into
    :attr:`BatchRunResults.error` (the offending phase aborts; subsequent
    phases still run if not blocked).

    Extracted from the modal class so it can be unit-tested against
    synthetic specs without Qt.
    """
    results = BatchRunResults()

    def _phase_cb(phase: str):
        if progress_callback is None:
            return None
        return lambda idx, total, path, status: progress_callback(
            phase, idx, total, path, status
        )

    # Convert first (so TOC builds on the new mono outputs), then build
    # TOC for both the mono outputs and any videos that didn't need
    # converting.
    if spec.convert_to_mono:
        try:
            results.converted = _batch.convert_to_mono(
                spec.source,
                verbose=False,
                show_progress=False,
                progress_callback=_phase_cb("convert"),
                cancel_check=cancel_check,
            )
        except Exception as e:  # noqa: BLE001
            results.error = f"convert_to_mono failed: {e}"
    if cancel_check is not None and cancel_check():
        results.cancelled = True
        return results

    if spec.build_toc:
        try:
            results.toc_results = _batch.build_toc(
                spec.source,
                show_progress=False,
                progress_callback=_phase_cb("toc"),
                cancel_check=cancel_check,
            )
        except Exception as e:  # noqa: BLE001
            # Don't clobber an earlier error.
            results.error = results.error or f"build_toc failed: {e}"
    if cancel_check is not None and cancel_check():
        results.cancelled = True
    return results


def _make_batch_modal_class():
    """Lazy factory for the batch-process modal class.

    Same qtpy-import-on-demand pattern as
    :func:`dustrack._overlays._make_open_video_overlay_class`. Returns
    a class that, when instantiated and ``exec_()``-ed, blocks on a
    synchronous Qt event loop until the user cancels or finishes a run.
    """
    from qtpy.QtCore import QEvent, QEventLoop, QObject, Qt, QThread, Signal
    from qtpy.QtWidgets import (
        QCheckBox,
        QFileDialog,
        QFrame,
        QHBoxLayout,
        QLabel,
        QProgressBar,
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
    _SECONDARY_QSS = (
        "QPushButton { background-color: #2a2a2a; color: white; "
        "  border: 1px solid #444; padding: 8px 24px; font-size: 11pt; }"
        "QPushButton:hover { background-color: #3a3a3a; }"
        "QPushButton:pressed { background-color: #1a1a1a; }"
    )

    class _BatchWorker(QThread):
        """Runs :func:`run_batch_jobs` on a background thread and emits
        per-file progress + a finished signal.

        Cancel is signalled via a :class:`threading.Event` (Qt-thread
        safe; the worker polls it between files).
        """

        progress = Signal(str, int, int, str, str)  # phase, idx, total, path, status
        finished_results = Signal(object)

        def __init__(self, spec: BatchJobSpec, parent=None):
            super().__init__(parent)
            self._spec = spec
            self._cancel_evt = threading.Event()

        def request_cancel(self):
            self._cancel_evt.set()

        def run(self):  # noqa: D401 -- QThread API
            def _cb(phase, idx, total, path, status):
                # path may be a Path; signals serialise it as str for
                # the slot's convenience.
                self.progress.emit(phase, idx, total, str(path), status)

            try:
                results = run_batch_jobs(
                    self._spec,
                    progress_callback=_cb,
                    cancel_check=self._cancel_evt.is_set,
                )
            except Exception as e:  # noqa: BLE001
                results = BatchRunResults(error=f"{e}\n{traceback.format_exc()}")
            self.finished_results.emit(results)

    class BatchModal(QObject):
        """Overlay-style modal: setup view -> running view -> done view.

        Surface (setup view):

        - Title + subtitle.
        - Source row: "Source: <none picked>" label + "Pick folder..."
          / "Pick DLC project..." buttons.
        - Two checkboxes: Convert to mono, Build TOC sidecars. The
          mono checkbox auto-disables when the source is a DLC project.
        - Run button (disabled until a source is picked) + Close button.

        Surface (running view):

        - Phase + current-file label (replaces the source row).
        - Progress bar (idx / total within the current phase).
        - Status feed (last few status lines).
        - Cancel button (replaces Run).

        Surface (done view):

        - Summary line ("Converted N files, built M TOCs.") + Close.
        """

        def __init__(self, main_window, *, initial_source: Optional[Path] = None):
            super().__init__(main_window)
            self._mw = main_window
            self._loop = QEventLoop()
            self._spec_source: Optional[Path] = (
                Path(initial_source) if initial_source else None
            )
            self._worker: Optional[_BatchWorker] = None
            self._results: Optional[BatchRunResults] = None
            # Recent status lines so the feed stays bounded (avoid
            # unbounded growth on a 1600-video corpus).
            self._status_lines: list[str] = []
            self._STATUS_FEED_MAX = 8

            self._frame = QFrame(main_window)
            self._frame.setObjectName("dustrack_batch_modal")
            self._frame.setStyleSheet(
                "#dustrack_batch_modal { background-color: rgba(0, 0, 0, 200); }"
                "QLabel { color: white; }"
                "#dustrack_batch_title { color: white; "
                "  font-size: 22pt; font-weight: bold; }"
                "#dustrack_batch_subtitle { color: #cccccc; "
                "  font-size: 11pt; }"
                "#dustrack_batch_source { color: #cccccc; "
                "  font-size: 10pt; padding: 6px; }"
                "#dustrack_batch_feed { color: #bbbbbb; "
                "  font-family: Consolas, 'Courier New', monospace; "
                "  font-size: 9pt; background-color: #1a1a1a; "
                "  border: 1px solid #333; padding: 6px; }"
                "QCheckBox { color: white; font-size: 11pt; "
                "  padding: 4px; }"
                "QCheckBox:disabled { color: #777; }"
                "QProgressBar { background-color: #1f1f1f; "
                "  border: 1px solid #444; color: white; "
                "  text-align: center; min-height: 20px; }"
                "QProgressBar::chunk { background-color: #3a86ff; }"
            )
            self._frame.setFocusPolicy(Qt.StrongFocus)

            layout = QVBoxLayout(self._frame)
            layout.setAlignment(Qt.AlignCenter)
            layout.addStretch(1)

            title = QLabel("Batch process videos")
            title.setObjectName("dustrack_batch_title")
            title.setAlignment(Qt.AlignCenter)
            layout.addWidget(title)

            subtitle = QLabel(
                "Convert ultrasound clips to h265 monochrome and/or pre-build "
                "PyAV+TOC sidecars so first-open is instant."
            )
            subtitle.setObjectName("dustrack_batch_subtitle")
            subtitle.setAlignment(Qt.AlignCenter)
            subtitle.setWordWrap(True)
            subtitle.setMaximumWidth(720)
            layout.addWidget(subtitle, alignment=Qt.AlignCenter)

            layout.addSpacing(18)

            # Source picker row.
            self._source_lbl = QLabel(self._source_label_text())
            self._source_lbl.setObjectName("dustrack_batch_source")
            self._source_lbl.setAlignment(Qt.AlignCenter)
            self._source_lbl.setMinimumWidth(640)
            self._source_lbl.setMaximumWidth(720)
            self._source_lbl.setWordWrap(True)
            layout.addWidget(self._source_lbl, alignment=Qt.AlignCenter)

            picker_row = QHBoxLayout()
            picker_row.setAlignment(Qt.AlignCenter)
            self._pick_folder_btn = QPushButton("Pick folder of videos...")
            self._pick_folder_btn.setStyleSheet(_SECONDARY_QSS)
            self._pick_folder_btn.clicked.connect(self._on_pick_folder)
            picker_row.addWidget(self._pick_folder_btn)
            layout.addLayout(picker_row)

            layout.addSpacing(14)

            # Operation checkboxes.
            ops_row = QHBoxLayout()
            ops_row.setAlignment(Qt.AlignCenter)
            self._mono_cb = QCheckBox("Convert to mono (h265)")
            self._mono_cb.setChecked(True)
            self._mono_cb.stateChanged.connect(self._refresh_run_state)
            self._toc_cb = QCheckBox("Build TOC sidecars")
            self._toc_cb.setChecked(True)
            self._toc_cb.stateChanged.connect(self._refresh_run_state)
            ops_row.addWidget(self._mono_cb)
            ops_row.addSpacing(24)
            ops_row.addWidget(self._toc_cb)
            layout.addLayout(ops_row)

            layout.addSpacing(18)

            # Progress widgets (hidden in setup view).
            self._phase_lbl = QLabel("")
            self._phase_lbl.setObjectName("dustrack_batch_source")
            self._phase_lbl.setAlignment(Qt.AlignCenter)
            self._phase_lbl.setMinimumWidth(640)
            self._phase_lbl.setMaximumWidth(720)
            self._phase_lbl.setWordWrap(True)
            self._phase_lbl.hide()
            layout.addWidget(self._phase_lbl, alignment=Qt.AlignCenter)

            self._progress = QProgressBar()
            self._progress.setMinimumWidth(640)
            self._progress.setMaximumWidth(720)
            self._progress.setRange(0, 100)
            self._progress.setValue(0)
            self._progress.hide()
            layout.addWidget(self._progress, alignment=Qt.AlignCenter)

            self._feed_lbl = QLabel("")
            self._feed_lbl.setObjectName("dustrack_batch_feed")
            self._feed_lbl.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            self._feed_lbl.setMinimumWidth(640)
            self._feed_lbl.setMaximumWidth(720)
            self._feed_lbl.setMinimumHeight(140)
            self._feed_lbl.setWordWrap(False)
            self._feed_lbl.hide()
            layout.addWidget(self._feed_lbl, alignment=Qt.AlignCenter)

            layout.addSpacing(14)

            # Action row.
            btn_row = QHBoxLayout()
            btn_row.setAlignment(Qt.AlignCenter)
            self._run_btn = QPushButton("Run")
            self._run_btn.setMinimumWidth(180)
            self._run_btn.setStyleSheet(_PRIMARY_QSS)
            self._run_btn.clicked.connect(self._on_run)
            self._close_btn = QPushButton("Close")
            self._close_btn.setMinimumWidth(140)
            self._close_btn.setStyleSheet(_SECONDARY_QSS)
            self._close_btn.clicked.connect(self._on_close)
            btn_row.addWidget(self._run_btn)
            btn_row.addSpacing(10)
            btn_row.addWidget(self._close_btn)
            layout.addLayout(btn_row)

            layout.addStretch(1)

            self._refresh_run_state()
            main_window.installEventFilter(self)
            self._frame.show()
            self._reposition()
            self._frame.raise_()

        # --- source / state helpers ---

        def _source_label_text(self) -> str:
            if self._spec_source is None:
                return "Source: (none picked)"
            return f"Source: {self._spec_source}"

        def _current_spec(self) -> Optional[BatchJobSpec]:
            if self._spec_source is None:
                return None
            spec = BatchJobSpec(
                source=self._spec_source,
                convert_to_mono=self._mono_cb.isChecked(),
                build_toc=self._toc_cb.isChecked(),
            )
            return spec

        def _refresh_run_state(self):
            spec = self._current_spec()
            if spec is None:
                self._run_btn.setEnabled(False)
                return
            self._run_btn.setEnabled(
                self._mono_cb.isChecked() or self._toc_cb.isChecked()
            )

        # --- pickers ---

        def _on_pick_folder(self):
            picked = QFileDialog.getExistingDirectory(
                self._frame,
                "Pick a folder of videos to batch-process",
                str(self._spec_source) if self._spec_source else "",
            )
            if not picked:
                return
            self._spec_source = Path(picked)
            self._source_lbl.setText(self._source_label_text())
            self._refresh_run_state()

        # --- run / cancel ---

        def _on_run(self):
            spec = self._current_spec()
            if spec is None:
                return
            # Switch to running view.
            self._pick_folder_btn.setEnabled(False)
            self._mono_cb.setEnabled(False)
            self._toc_cb.setEnabled(False)
            self._run_btn.setText("Cancel")
            self._run_btn.setEnabled(True)
            try:
                self._run_btn.clicked.disconnect()
            except (TypeError, RuntimeError):
                pass
            self._run_btn.clicked.connect(self._on_cancel)
            self._close_btn.setEnabled(False)
            self._phase_lbl.setText("Starting...")
            self._phase_lbl.show()
            self._progress.setValue(0)
            self._progress.show()
            self._feed_lbl.setText("")
            self._feed_lbl.show()
            self._status_lines = []

            self._worker = _BatchWorker(spec, parent=self)
            self._worker.progress.connect(self._on_progress)
            self._worker.finished_results.connect(self._on_finished)
            self._worker.start()

        def _on_cancel(self):
            if self._worker is None:
                return
            self._worker.request_cancel()
            self._phase_lbl.setText("Cancelling -- waiting for current file to finish...")
            self._run_btn.setEnabled(False)

        def _on_progress(self, phase: str, idx: int, total: int, path_str: str, status: str):
            tag = _STATUS_LABELS.get(status, status)
            name = Path(path_str).name
            phase_human = {"convert": "Converting", "toc": "Building TOC"}.get(
                phase, phase
            )
            self._phase_lbl.setText(
                f"{phase_human} ({idx + 1}/{total}): {name}"
            )
            if total > 0:
                self._progress.setValue(int(100 * (idx + 1) / total))
            self._status_lines.append(f"  [{tag}] {name}")
            if len(self._status_lines) > self._STATUS_FEED_MAX:
                self._status_lines = self._status_lines[-self._STATUS_FEED_MAX:]
            self._feed_lbl.setText("\n".join(self._status_lines))

        def _on_finished(self, results: BatchRunResults):
            self._results = results
            n_converted = len(results.converted)
            toc_status_counts: dict[str, int] = {}
            for status in results.toc_results.values():
                # Bucket "error: ..." under a single "error" key.
                key = status if not str(status).startswith("error") else "error"
                toc_status_counts[key] = toc_status_counts.get(key, 0) + 1
            n_toc_built = toc_status_counts.get("built", 0) + toc_status_counts.get(
                "built (uncached)", 0
            )
            n_toc_hit = toc_status_counts.get("hit", 0)
            n_toc_err = toc_status_counts.get("error", 0)

            if results.cancelled:
                summary = (
                    f"Cancelled. Converted {n_converted}; TOC built "
                    f"{n_toc_built}, hits {n_toc_hit}, errors {n_toc_err}."
                )
            elif results.error:
                summary = f"Error: {results.error}"
            else:
                summary = (
                    f"Done. Converted {n_converted} file(s); TOC built "
                    f"{n_toc_built}, hits {n_toc_hit}, errors {n_toc_err}."
                )
            self._phase_lbl.setText(summary)
            self._progress.setValue(100)
            try:
                self._run_btn.clicked.disconnect()
            except (TypeError, RuntimeError):
                pass
            self._run_btn.hide()
            self._close_btn.setEnabled(True)
            self._close_btn.setText("Done")
            self._close_btn.setStyleSheet(_PRIMARY_QSS)
            self._close_btn.setFocus()

        # --- dismiss / event-filter scaffolding ---

        def _on_close(self):
            # If the worker is still running, ask it to stop; the
            # finished slot will fire and tear us down. Otherwise
            # dismiss immediately.
            if self._worker is not None and self._worker.isRunning():
                self._worker.request_cancel()
                self._worker.wait(5000)
            self._dismiss()
            if self._loop.isRunning():
                self._loop.quit()

        def eventFilter(self, obj, event):  # noqa: N802 (Qt API)
            if obj is self._mw:
                t = event.type()
                if t == QEvent.Resize:
                    self._reposition()
                elif t == QEvent.Close:
                    if self._worker is not None and self._worker.isRunning():
                        self._worker.request_cancel()
                        self._worker.wait(5000)
                    self._dismiss_no_filter_unhook()
                    if self._loop.isRunning():
                        self._loop.quit()
            return False

        def _reposition(self):
            self._frame.setGeometry(0, 0, self._mw.width(), self._mw.height())
            self._frame.raise_()

        def _dismiss_no_filter_unhook(self):
            self._frame.hide()
            self._frame.deleteLater()

        def _dismiss(self):
            try:
                self._mw.removeEventFilter(self)
            except Exception:  # noqa: BLE001
                pass
            self._frame.hide()
            self._frame.deleteLater()

        def exec_(self) -> Optional[BatchRunResults]:
            """Block until the user closes the modal. Returns the
            :class:`BatchRunResults` from the last run, or ``None`` if
            the user closed without running anything.
            """
            self._loop.exec_()
            return self._results

    return BatchModal


def open_batch_modal(
    main_window,
    *,
    initial_source: Optional[Union[str, Path]] = None,
) -> Optional[BatchRunResults]:
    """Convenience launcher used by the welcome modal + Tools menu.

    Constructs the lazy modal class, mounts it on ``main_window``, and
    blocks on its event loop. Returns the run results (or ``None`` if
    the user closed without running anything).
    """
    BatchModal = _make_batch_modal_class()
    modal = BatchModal(
        main_window,
        initial_source=Path(initial_source) if initial_source else None,
    )
    return modal.exec_()
