"""Qt UI for the Train pre-flight: confirm modals + remediation orchestration.

Three modal prompts and one orchestrator, all keyed on the
``issues`` dict produced by
:func:`._preflight.scan_unsaved_and_incomplete`:

* :func:`prompt_unified_pre_flight` -- the main "Save and clean"
  modal, shown when any manual layer has unsaved edits and/or
  incomplete frames. Returns ``True`` iff the user picked
  *Save and clean*.

* :func:`prompt_no_trainable_labels` -- hard-block overlay shown
  when no labels exist anywhere in the project (the freshly-seeded
  iteration-1 case before the user has annotated anything).

* :func:`prompt_empty_layer_train_confirm` -- soft confirm shown
  when the active layer is empty but other label sources exist
  ("continue training without new data?"). Returns ``True`` iff
  *Continue training*.

* :func:`apply_pre_flight_remediations` -- thin wrapper around the
  logic-side remediation in :mod:`._preflight`, injecting the
  canonical ``make_annotation_file_name`` constructor so the
  logic module doesn't need to import from ``_file_management``.

Pure UI -- all logic (the scans, diffs, formatters, sidecar
writers) lives in :mod:`._preflight`. The DUSTrack class is a
thin coordinator on top of both.

Extracted from ``gui.DUSTrack`` in the 1.2.0rc1 follow-up.
"""
from __future__ import annotations

from ._overlays import _make_confirm_overlay_class
from ._preflight import (
    apply_pre_flight_remediations as _apply_remediations_logic,
    format_pre_flight_summary,
)


def prompt_unified_pre_flight(qt_window, issues: dict) -> bool:
    """Single modal for the combined save-state + incompleteness
    pre-flight. Returns True iff the user picked *Save and clean*.

    Routes through :class:`ConfirmOverlay` (rc2) so the modal
    shares visual vocabulary with the new ``Discard unsaved`` /
    ``Remove layer`` confirms. The per-layer breakdown is shown
    inline rather than behind a collapsed "Show Details..." toggle
    -- the breakdown is the substance the user needs to decide on,
    not optional extra.
    """
    ConfirmOverlay = _make_confirm_overlay_class()
    n = len(issues)
    header = (
        f"{n} manual annotation layer{'s' if n != 1 else ''} "
        f"{'have' if n != 1 else 'has'} unsaved changes and/or "
        "incomplete frames."
    )
    breakdown = format_pre_flight_summary(issues)
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


def prompt_no_trainable_labels(qt_window, active_layer_name: str) -> None:
    """Hard-block overlay for the Train DLC path when the active
    manual layer is empty AND no other source of labels exists in
    the project (no other non-empty manual layer, no
    ``labeled-data/*.h5``). Distinct from
    :func:`prompt_empty_layer_train_confirm` -- there's nothing
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
            f"Active layer {active_layer_name!r} has no labels, and no "
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


def prompt_empty_layer_train_confirm(qt_window, active_layer_name: str) -> bool:
    """Modal that fires when Train DLC is clicked with an empty
    active manual layer. Returns True iff the user confirmed
    ``Continue training``.

    User intent in this state: "train for more iterations without
    new labels." Training will reuse whatever labels already
    exist in ``labeled-data/``; if none do (e.g. a freshly-seeded
    iteration-1), DLC will fail downstream with its own error.
    """
    ConfirmOverlay = _make_confirm_overlay_class()
    body = (
        f"Active layer {active_layer_name!r} has no labels.\n\n"
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


def apply_pre_flight_remediations(annotations, video_fname, issues: dict) -> None:
    """Wrapper around :func:`._preflight.apply_pre_flight_remediations`
    that injects the ``make_annotation_file_name`` builder (kept
    out of the logic module to avoid an import cycle with
    ``_file_management``).
    """
    from ._file_management import make_annotation_file_name
    _apply_remediations_logic(
        annotations, video_fname, issues,
        make_annotation_file_name=make_annotation_file_name,
    )
