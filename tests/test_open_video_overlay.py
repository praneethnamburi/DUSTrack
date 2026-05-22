"""Tests for the 1.2.0a3 ``OpenVideoOverlay`` welcome-modal contextual-
button + history-toggle behaviour (revised 2026-05-22 follow-up).

Surface under test:

- Initial state: action button labelled ``Open``; helpful message asks
  the user to pick a video / config.yaml or click a recent row.
- Recent-row single-click toggles selection (click same row twice =
  deselect). Selection flips the action button label to ``Load`` and
  the help text to the "click Load to open" form.
- Double-click / Enter on a recent row commits immediately (muscle-
  memory shortcut, bypasses the staged-Load step).
- Action button dispatches on state: ``Open`` mode pops the file
  dialog and commits the dialog return; ``Load`` mode commits the
  selected recent row.

``exec_()`` itself blocks on a Qt event loop and is left to manual
smoke -- we exercise the public-ish staging methods directly.
"""
from __future__ import annotations

from pathlib import Path

import pytest

try:
    from qtpy.QtCore import QCoreApplication
    from qtpy.QtWidgets import QApplication, QMainWindow
except ImportError:  # pragma: no cover
    pytest.skip("qtpy not installed", allow_module_level=True)


@pytest.fixture(scope="module")
def qapp():
    app = QCoreApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def main_window(qapp):
    win = QMainWindow()
    win.resize(800, 600)
    win.show()
    qapp.processEvents()
    yield win
    win.close()


@pytest.fixture
def overlay_cls():
    from dustrack.dlcinterface import _make_open_video_overlay_class
    return _make_open_video_overlay_class()


# ---------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------


class TestInitialState:
    def test_button_label_is_open_with_no_selection(
        self, main_window, overlay_cls,
    ):
        ov = overlay_cls(main_window, recent_sessions=[])
        try:
            assert ov._action_btn.text() == "Open"
            assert ov._selected_index is None
            # Help text in "no selection" mode.
            assert "video" in ov._help_lbl.text().lower()
            # Empty history -> no recent widget at all.
            assert ov._recent_widget is None
        finally:
            ov._frame.deleteLater()

    def test_with_recent_history_button_still_open_until_select(
        self, main_window, overlay_cls, tmp_path,
    ):
        v = tmp_path / "v.mp4"
        v.write_bytes(b"")
        ov = overlay_cls(main_window, recent_sessions=[[v]])
        try:
            assert ov._action_btn.text() == "Open"
            assert ov._selected_index is None
            assert ov._recent_widget is not None
            assert ov._recent_widget.count() == 1
        finally:
            ov._frame.deleteLater()


# ---------------------------------------------------------------------
# Recent-row toggle
# ---------------------------------------------------------------------


class TestRecentToggle:
    def test_click_selects_and_flips_button_to_load(
        self, main_window, overlay_cls, tmp_path,
    ):
        v = tmp_path / "r.mp4"
        v.write_bytes(b"")
        ov = overlay_cls(main_window, recent_sessions=[[v]])
        try:
            item = ov._recent_widget.item(0)
            ov._on_recent_clicked(item)
            assert ov._selected_index == 0
            assert ov._action_btn.text() == "Load"
            assert "load" in ov._help_lbl.text().lower()
        finally:
            ov._frame.deleteLater()

    def test_click_same_row_deselects(
        self, main_window, overlay_cls, tmp_path,
    ):
        v = tmp_path / "r.mp4"
        v.write_bytes(b"")
        ov = overlay_cls(main_window, recent_sessions=[[v]])
        try:
            item = ov._recent_widget.item(0)
            ov._on_recent_clicked(item)
            assert ov._selected_index == 0
            ov._on_recent_clicked(item)  # toggle off
            assert ov._selected_index is None
            assert ov._action_btn.text() == "Open"
        finally:
            ov._frame.deleteLater()

    def test_click_different_row_swaps_selection(
        self, main_window, overlay_cls, tmp_path,
    ):
        v0 = tmp_path / "a.mp4"
        v1 = tmp_path / "b.mp4"
        for v in (v0, v1):
            v.write_bytes(b"")
        ov = overlay_cls(main_window, recent_sessions=[[v0], [v1]])
        try:
            ov._on_recent_clicked(ov._recent_widget.item(0))
            assert ov._selected_index == 0
            ov._on_recent_clicked(ov._recent_widget.item(1))
            assert ov._selected_index == 1
            assert ov._action_btn.text() == "Load"
        finally:
            ov._frame.deleteLater()


# ---------------------------------------------------------------------
# Commit pathways
# ---------------------------------------------------------------------


class TestCommit:
    def test_load_mode_commits_selected_row(
        self, main_window, overlay_cls, tmp_path,
    ):
        v = tmp_path / "r.mp4"
        v.write_bytes(b"")
        ov = overlay_cls(main_window, recent_sessions=[[v]])
        try:
            ov._on_recent_clicked(ov._recent_widget.item(0))
            assert ov._action_btn.text() == "Load"
            ov._on_action_clicked()
            assert ov._result == [v]
        finally:
            ov._frame.deleteLater()

    def test_double_click_commits_without_first_selecting(
        self, main_window, overlay_cls, tmp_path,
    ):
        v = tmp_path / "r.mp4"
        v.write_bytes(b"")
        ov = overlay_cls(main_window, recent_sessions=[[v]])
        try:
            ov._on_recent_activated(ov._recent_widget.item(0))
            assert ov._result == [v]
        finally:
            ov._frame.deleteLater()

    def test_open_mode_dispatches_to_file_dialog(
        self, main_window, overlay_cls, tmp_path, monkeypatch,
    ):
        v = tmp_path / "picked.mp4"
        v.write_bytes(b"")
        from dustrack import _overlays
        monkeypatch.setattr(
            _overlays, "_prompt_for_videos",
            lambda parent=None: [v],
        )
        ov = overlay_cls(main_window, recent_sessions=[])
        try:
            assert ov._action_btn.text() == "Open"
            ov._on_action_clicked()
            assert ov._result == [v]
        finally:
            ov._frame.deleteLater()

    def test_open_mode_cancel_stays_in_modal(
        self, main_window, overlay_cls, monkeypatch,
    ):
        from dustrack import _overlays
        monkeypatch.setattr(
            _overlays, "_prompt_for_videos",
            lambda parent=None: None,
        )
        ov = overlay_cls(main_window, recent_sessions=[])
        try:
            ov._on_action_clicked()
            # Modal still alive; no result committed.
            assert ov._result is None
        finally:
            ov._frame.deleteLater()

    def test_commit_recent_out_of_range_is_noop(
        self, main_window, overlay_cls, tmp_path,
    ):
        v = tmp_path / "r.mp4"
        v.write_bytes(b"")
        ov = overlay_cls(main_window, recent_sessions=[[v]])
        try:
            ov._commit_recent(None)
            assert ov._result is None
            ov._commit_recent(99)
            assert ov._result is None
        finally:
            ov._frame.deleteLater()
