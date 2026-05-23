"""Multi-video navigation row: ``◀ <video dropdown> ▶``.

The nav row mounts at the TOP of the rc2 left-column dock. The
central QComboBox lists every bundle's video as ``"i. <stem>"``;
the arrows step prev/next (also bound to Alt+Left / Alt+Right).
Trailing marker on dropdown items reflects hydration state
(``…`` for pending/hydrating, ``✗`` for failed).

Always rendered -- in a single-video session the arrows are
disabled and the dropdown shows one entry, but the row stays
visible so the affordance is discoverable when a multi-video
session is opened later.

Extracted from ``gui.DUSTrack`` in the 1.2.0rc1 follow-up.
"""

from __future__ import annotations

from pathlib import Path

from ._bundle import HYDRATION_FAILED, HYDRATION_HYDRATING, HYDRATION_PENDING


def add_nav_widget(dustrack) -> None:
    """Mount the nav row at the top of the left-column dock.

    No-op on the mpl-fallback path (no Qt main window / left
    column). On success, populates the following attributes on the
    tracker shell: ``_nav_widget``, ``_nav_prev_btn``,
    ``_nav_next_btn``, ``_nav_combo``.
    """
    qt_window = dustrack._find_qt_window()
    if qt_window is None:
        return
    col = getattr(qt_window, "_dnav_left_column", None)
    if col is None:
        return
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QColor
    from qtpy.QtWidgets import (
        QComboBox,
        QFrame,
        QHBoxLayout,
        QSizePolicy,
        QToolButton,
        QWidget,
    )

    row = QWidget(col.host)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(6, 4, 6, 4)
    layout.setSpacing(4)

    prev_btn = QToolButton(row)
    prev_btn.setText("◀")
    prev_btn.setFocusPolicy(Qt.NoFocus)
    prev_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    prev_btn.clicked.connect(lambda _checked=False: dustrack.swap_prev())

    combo = QComboBox(row)
    combo.setFocusPolicy(Qt.NoFocus)
    combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    # ``activated[int]`` fires only on user interaction (click /
    # keyboard selection) -- not on programmatic ``setCurrentIndex``,
    # which the post-swap sync uses. A sync-after-swap would otherwise
    # recurse into ``swap_to``.
    combo.activated.connect(dustrack._on_nav_combo_activated)

    next_btn = QToolButton(row)
    next_btn.setText("▶")
    next_btn.setFocusPolicy(Qt.NoFocus)
    next_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    next_btn.clicked.connect(lambda _checked=False: dustrack.swap_next())

    layout.addWidget(prev_btn)
    layout.addWidget(combo, stretch=1)
    layout.addWidget(next_btn)

    # Pale-blue palette echoing the Workflow group bg so the nav
    # row reads as "header above the workflow column" rather than
    # a stranded widget.
    row.setAutoFillBackground(True)
    pal = row.palette()
    pal.setColor(row.backgroundRole(), QColor("#cfdef3"))
    pal.setColor(row.foregroundRole(), QColor("#2c3e50"))
    row.setPalette(pal)
    sep = QFrame(col.host)
    sep.setFrameShape(QFrame.HLine)
    sep.setFrameShadow(QFrame.Sunken)

    col.outer_layout.insertWidget(0, row)
    col.outer_layout.insertWidget(1, sep)

    dustrack._nav_widget = row
    dustrack._nav_prev_btn = prev_btn
    dustrack._nav_next_btn = next_btn
    dustrack._nav_combo = combo


def on_nav_combo_activated(dustrack, index: int) -> None:
    """User-triggered dropdown selection -> swap to that bundle.

    On a rejected swap (out-of-bounds, hydration-failed) we re-sync
    the combo back to the still-active index so the visible selection
    matches reality.
    """
    if index == dustrack._active_index:
        return
    ok = dustrack.swap_to(index)
    if not ok:
        dustrack._refresh_nav_buttons()


def add_video_nav_key_bindings(dustrack) -> None:
    """Register ``Alt+Left`` / ``Alt+Right`` for previous / next
    video. Verified unbound in dnav core key bindings -- bare
    arrows are taken for frame nav.
    """
    try:
        dustrack.add_key_binding(
            "alt+left",
            dustrack.swap_prev,
            "Previous video",
            group="0. Video navigation",
        )
        dustrack.add_key_binding(
            "alt+right",
            dustrack.swap_next,
            "Next video",
            group="0. Video navigation",
        )
    except Exception:  # noqa: BLE001 - older dnav signature / no method
        pass


def refresh_nav_buttons(dustrack) -> None:
    """Sync the nav row's dropdown + enable states to
    ``dustrack._bundles`` + ``dustrack._active_index``. Idempotent;
    cheap; safe to call from any state-change site (swap, bundle
    init, bg-hydration progress tick).
    """
    if dustrack._nav_widget is None:
        return
    n = max(len(dustrack._bundles), 1)
    i = dustrack._active_index
    combo = getattr(dustrack, "_nav_combo", None)
    if combo is not None:
        sync_nav_combo(dustrack, combo, n=n, active=i)
    if dustrack._nav_prev_btn is not None:
        dustrack._nav_prev_btn.setEnabled(i > 0)
    if dustrack._nav_next_btn is not None:
        dustrack._nav_next_btn.setEnabled(i < n - 1)


def format_nav_combo_item(bundle, idx: int) -> str:
    """Format one dropdown row as ``"i. <stem>"`` with a trailing
    marker for non-ready bundles. Pure -- testable from a stub
    bundle with ``fname`` + ``hydration_state``.
    """
    stem = Path(bundle.fname).stem
    label = f"{idx + 1}. {stem}"
    state = bundle.hydration_state
    if state == HYDRATION_HYDRATING or state == HYDRATION_PENDING:
        return f"{label}  …"
    if state == HYDRATION_FAILED:
        return f"{label}  ✗"
    return label


def sync_nav_combo(dustrack, combo, *, n: int, active: int) -> None:
    """Bring the dropdown's items + selection + tooltips in line
    with the current bundle list.

    Programmatic mutations are wrapped in ``blockSignals`` so the
    ``activated`` connection (user-only) is never re-entered from
    this path. When the bundle identity list is unchanged, only
    per-item suffixes + tooltips + the active selection are touched
    -- a hot path during bg-hydration progress ticks.
    """
    try:
        from qtpy.QtCore import Qt

        tooltip_role = Qt.ToolTipRole
    except Exception:  # noqa: BLE001 -- no qtpy in this env
        tooltip_role = 3  # Qt::ToolTipRole

    bundles = dustrack._bundles
    # Snapshot the fname list so a count-only check below is robust
    # to in-place mutations of dustrack._bundles.
    fnames = [str(b.fname) for b in bundles]
    signature = tuple(fnames)
    prior_signature = getattr(dustrack, "_nav_combo_signature", None)

    combo.blockSignals(True)
    try:
        if signature != prior_signature:
            combo.clear()
            for j, b in enumerate(bundles):
                combo.addItem(format_nav_combo_item(b, j))
                combo.setItemData(j, fnames[j], tooltip_role)
            if not bundles:
                # Placeholder for the (rare) zero-bundle stub state
                # so the widget isn't empty.
                combo.addItem("(no videos)")
            dustrack._nav_combo_signature = signature
        else:
            # Same bundles, possibly different hydration states.
            for j, b in enumerate(bundles):
                text = format_nav_combo_item(b, j)
                if combo.itemText(j) != text:
                    combo.setItemText(j, text)
                combo.setItemData(j, fnames[j], tooltip_role)
        if bundles:
            clamped = max(0, min(active, len(bundles) - 1))
            if combo.currentIndex() != clamped:
                combo.setCurrentIndex(clamped)
            # Combo's own hover tooltip: full path of the currently-
            # displayed video.
            combo.setToolTip(fnames[clamped])
        else:
            combo.setToolTip("")
    finally:
        combo.blockSignals(False)
