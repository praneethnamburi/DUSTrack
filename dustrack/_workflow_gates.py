"""Workflow-button enable/disable gates.

Three buttons in the Workflow sidebar group have preconditions that
must be re-checked whenever the relevant session state changes:

* **Create DLC Project** -- disabled inside an existing DLC project
  (a nested ``copy_videos`` scaffold would have unhandled paths)
  and while the lazy DLC import is still loading or failed.
* **Train DLC model** -- disabled until a ``DLCProject`` is created
  / opened. Replaces the click-time ``ValueError`` with a greyed-out
  button + tooltip.
* **Apply manual corrections** -- disabled without an overlay set,
  or when the active layer is already the corrections output
  (circular splice).

:func:`evaluate_workflow_gates` is the pure decision logic --
data in, gate dict out, no Qt. :func:`refresh_workflow_button_state`
applies the gate dict to the live buttons. :func:`install_dlc_load_gate_refresh`
arms a QTimer that re-fires the refresh once the lazy DLC import
resolves.

Extracted from ``gui.DUSTrack`` in the 1.2.0rc1 follow-up.
"""
from __future__ import annotations

import traceback

from .dlcloader import HAS_DLC, _dlc_load_state


def evaluate_workflow_gates(dustrack) -> dict:
    """Compute ``{button_label: (enabled, tooltip)}`` for the gated buttons.

    Reads ``dustrack._dlcproject``, ``dustrack._current_overlay``,
    ``dustrack.ann``, ``dustrack.fname``, and the lazy-loader's
    ``_dlc_load_state()``. Pure -- no widget mutations. Testable
    with a mock dustrack (any object with the right attrs).
    """
    from .dlcinterface import _session_inside_dlc_project

    gates: dict = {}
    corrections_layer = type(dustrack).CORRECTIONS_LAYER_NAME

    # --- Create DLC Project --------------------------------------
    proj_root = _session_inside_dlc_project(dustrack)
    dlc_state = _dlc_load_state()
    if proj_root is not None:
        # "Inside a project" wins over "still loading" -- the click
        # would refuse on that ground first.
        gates["Create DLC Project"] = (
            False,
            f"Already inside DLC project {proj_root.name!r} — "
            "use Train DLC model to extend it.",
        )
    elif dlc_state in ("pending", "loading"):
        gates["Create DLC Project"] = (
            False,
            "Loading DeepLabCut… (this button enables once the "
            "import completes -- typically a few seconds after "
            "DUSTrack launches).",
        )
    elif dlc_state == "missing":
        gates["Create DLC Project"] = (
            False,
            "DeepLabCut failed to load. Check the launching "
            "terminal for the import error.",
        )
    else:
        gates["Create DLC Project"] = (True, "")

    # --- Train DLC model -----------------------------------------
    if dustrack._dlcproject is None:
        gates["Train DLC model"] = (
            False,
            "Create a DLC project first.",
        )
    else:
        gates["Train DLC model"] = (True, "")

    # --- Apply manual corrections --------------------------------
    ann = getattr(dustrack, "ann", None)
    ann_name = getattr(ann, "name", None) if ann is not None else None
    if dustrack._current_overlay is None:
        gates["Apply manual corrections"] = (
            False,
            "Set an overlay layer (typically a 'dlc_*' trace) "
            "first.",
        )
    elif ann_name == corrections_layer:
        gates["Apply manual corrections"] = (
            False,
            "Switch the active layer back to your manual "
            f"annotations — {corrections_layer!r} is the "
            "output, not the input.",
        )
    else:
        gates["Apply manual corrections"] = (True, "")

    return gates


def refresh_workflow_button_state(dustrack) -> None:
    """Enable / disable Workflow-group buttons based on session state.

    Qt-only: walks ``dustrack.buttons`` and writes ``setEnabled`` +
    ``setToolTip`` on each Button's ``_qt_btn`` attribute. No-op on
    any button whose Qt handle is missing (the legacy mpl-fallback
    path; no longer supported as a first-class deployment, kept
    working for users on pinned older versions).

    ``Reduce jitter`` is intentionally *not* gated here: its real
    precondition is "every frame in the active layer is fully
    annotated", which is a data property rather than a name-pattern
    property.
    """
    if not HAS_DLC:
        # The Workflow group's DLC buttons aren't added when
        # deeplabcut is missing; nothing to gate.
        return
    gates = evaluate_workflow_gates(dustrack)
    for label, (enabled, tooltip) in gates.items():
        if label not in dustrack.buttons:
            continue
        btn = dustrack.buttons[label]
        qt_btn = getattr(btn, "_qt_btn", None)
        if qt_btn is None:
            continue
        qt_btn.setEnabled(enabled)
        qt_btn.setToolTip(tooltip)


def install_dlc_load_gate_refresh(dustrack) -> None:
    """Re-evaluate workflow gates once the lazy DLC import resolves.

    Poll-based: starts a 250 ms ``QTimer`` on the Qt main thread
    that watches :func:`_dlc_load_state`; when state transitions
    out of ``"pending"`` / ``"loading"`` the timer fires
    :func:`refresh_workflow_button_state` once and stops itself.
    Polling (vs. a cross-thread signal hop from the loader thread)
    keeps every Qt touch on the main thread.

    No-op on the mpl-fallback path (no Qt window) and on the
    ``HAS_DLC=False`` path (Workflow buttons aren't created).
    """
    if not HAS_DLC:
        return
    if _dlc_load_state() in ("done", "missing"):
        # Already resolved; gates are in their final state.
        return
    try:
        from qtpy.QtCore import QTimer
    except Exception:  # noqa: BLE001 -- mpl-only / pre-Qt teardown
        return

    qt_window = dustrack._find_qt_window()
    if qt_window is None:
        return

    timer = QTimer(qt_window)
    timer.setInterval(250)

    def _tick():
        if _dlc_load_state() in ("done", "missing"):
            timer.stop()
            try:
                refresh_workflow_button_state(dustrack)
            except Exception:  # noqa: BLE001 -- defensive
                traceback.print_exc()

    timer.timeout.connect(_tick)
    timer.start()
    # Keep a reference so Qt doesn't garbage-collect the timer
    # mid-poll. Same convention as the overlay-worker timer.
    dustrack._dlc_load_gate_timer = timer
