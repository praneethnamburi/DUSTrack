"""Qt styling helpers: QSS for sidebar button groups + light/dark palette pin.

Two concerns:

* :func:`_qss_for_group` / :func:`_make_group_styler` -- per-group QSS
  factory closed over a palette ``spec`` (bg / fg / border / hover /
  pressed / disabled). Used by :meth:`DUSTrack._add_default_buttons`
  to paint the Workflow / Display / Niche / Utilities / Swap sidebar
  bands.

* :func:`_pin_qt_palette` -- deterministic Fusion-styled
  ``QApplication`` palette so DUSTrack paints the same colors
  regardless of Qt binding (PySide6 6.5+ honors the OS color scheme
  by default; PyQt6 does not) and regardless of Windows system
  theme. See :memory:`feedback_qt_fusion_standardpalette_os_theme`.

Extracted from ``dlcinterface.py`` in dustrack 1.2.0rc1.
"""

from __future__ import annotations


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
        pal.setColor(QPalette.Window, QColor(45, 45, 45))
        pal.setColor(QPalette.WindowText, QColor(220, 220, 220))
        pal.setColor(QPalette.Base, QColor(30, 30, 30))
        pal.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
        pal.setColor(QPalette.Text, QColor(220, 220, 220))
        pal.setColor(QPalette.Button, QColor(60, 60, 60))
        pal.setColor(QPalette.ButtonText, QColor(220, 220, 220))
        pal.setColor(QPalette.ToolTipBase, QColor(45, 45, 45))
        pal.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
        pal.setColor(QPalette.Highlight, QColor(70, 110, 180))
        pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    else:
        # Explicit light palette. Do NOT use ``app.style().standardPalette()``
        # -- in Qt 6.5+ Fusion's standard palette follows the OS color
        # scheme, so on a Windows-dark-mode machine it returns dark
        # colors and the whole point of the pin is lost.
        pal.setColor(QPalette.Window, QColor(240, 240, 240))
        pal.setColor(QPalette.WindowText, QColor(0, 0, 0))
        pal.setColor(QPalette.Base, QColor(255, 255, 255))
        pal.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
        pal.setColor(QPalette.Text, QColor(0, 0, 0))
        pal.setColor(QPalette.Button, QColor(240, 240, 240))
        pal.setColor(QPalette.ButtonText, QColor(0, 0, 0))
        pal.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
        pal.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
        pal.setColor(QPalette.Highlight, QColor(70, 110, 180))
        pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(pal)
