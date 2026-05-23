"""Qt UI for the Train pre-flight: confirm modals + remediation orchestration.

Three modal prompts and one orchestrator, all keyed on the
``issues`` dict produced by
:func:`._preflight.scan_unsaved_and_incomplete`:

* :func:`prompt_unified_pre_flight` -- the main "Save and clean"
  modal, shown when any manual layer has unsaved edits, incomplete
  frames, or stray labels (labels with annotations that aren't in
  the project's bodyparts). Returns a :class:`PreFlightDecision`
  with ``proceed`` (user picked *Save and clean*) and
  ``keep_strays`` (the optional checkbox state, default False).

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

from dataclasses import dataclass

from ._overlays import _make_confirm_overlay_class
from ._preflight import (
    apply_pre_flight_remediations as _apply_remediations_logic,
    format_pre_flight_summary,
    has_strays,
)


@dataclass(frozen=True)
class PreFlightDecision:
    """User's decision out of the unified pre-flight modal.

    ``proceed`` is True iff the user clicked *Save and clean*.
    ``keep_strays`` is the checkbox state -- True means "keep stray
    labels in the saved JSON file", False (default, the unchecked
    box) means "strip them on save". The checkbox is only rendered
    when the issues dict contains strays, but the field is always
    populated on the decision (False when no checkbox was shown).
    """

    proceed: bool
    keep_strays: bool = False


# Canonical phrasing for the stray-labels checkbox. Multi-naming
# alternatives considered: "Keep extra labels in the saved file
# (they won't be trained)", "Preserve labels not in this DLC
# project". The canonical form names the action ("save") and the
# user-side criterion ("not in this DLC project", avoiding the
# DLC-vocabulary word "bodyparts") and reassures about the
# consequence ("won't affect training").
_KEEP_STRAYS_CHECKBOX_LABEL = (
    "Save labels not in this DLC project (won't affect training)"
)


def prompt_unified_pre_flight(qt_window, issues: dict) -> PreFlightDecision:
    """Single modal for the combined save-state + incompleteness +
    stray-labels pre-flight. Returns a :class:`PreFlightDecision`.

    Routes through :class:`ConfirmOverlay` (rc2) so the modal
    shares visual vocabulary with the new ``Discard unsaved`` /
    ``Remove layer`` confirms. The per-layer breakdown is shown
    inline rather than behind a collapsed "Show Details..." toggle
    -- the breakdown is the substance the user needs to decide on,
    not optional extra.

    When any layer has stray labels, an opt-in checkbox is added
    asking the user whether to keep them in the saved file. Default
    unchecked: the user must deliberately press the checkbox to
    preserve work on extra labels. The expected workflow is "Save
    stray labels, then later fork a new DLC project with the
    expanded label set" -- spelling this out in the body so the
    decision has context.
    """
    ConfirmOverlay = _make_confirm_overlay_class()
    n = len(issues)
    header = (
        f"{n} manual annotation layer{'s' if n != 1 else ''} "
        f"{'have' if n != 1 else 'has'} unsaved changes, "
        "incomplete frames, or labels outside this DLC project."
    )
    breakdown = format_pre_flight_summary(issues)
    strays_present = has_strays(issues)
    actions = [
        " - save in-memory edits to disk for the listed layer(s),",
        " - drop frames missing one or more bodyparts (per-layer "
        "recovery sidecars written next to each annotation file),",
    ]
    if strays_present:
        actions.append(
            " - drop annotations on labels not in this DLC project "
            "(opt out via the checkbox below to keep them in the "
            "saved file),"
        )
    actions.append(" - then start training.")
    body = (
        f"{header}\n\n"
        f"{breakdown}\n\n"
        "Save and clean will:\n" + "\n".join(actions) + "\n\n"
        "Cancel returns to the UI without changes."
    )
    checkboxes = []
    if strays_present:
        checkboxes.append({
            "key": "keep_strays",
            "label": _KEEP_STRAYS_CHECKBOX_LABEL,
            "default_checked": False,
        })
    overlay = ConfirmOverlay(
        qt_window,
        title="Pre-flight issues",
        message=body,
        buttons=[
            ("Save and clean", "primary"),
            ("Cancel", "neutral"),
        ],
        default="Cancel",
        severity="warning",
        checkboxes=checkboxes,
    )
    result = overlay.exec_()
    return PreFlightDecision(
        proceed=(result == "Save and clean"),
        keep_strays=bool(overlay.checkbox_states.get("keep_strays", False)),
    )


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


def apply_pre_flight_remediations(
    annotations, video_fname, issues: dict, *, strip_strays: bool = True,
) -> None:
    """Wrapper around :func:`._preflight.apply_pre_flight_remediations`
    that injects the ``make_annotation_file_name`` builder (kept
    out of the logic module to avoid an import cycle with
    ``_file_management``) and forwards the ``strip_strays`` decision
    from the modal's checkbox.
    """
    from ._file_management import make_annotation_file_name

    _apply_remediations_logic(
        annotations,
        video_fname,
        issues,
        make_annotation_file_name=make_annotation_file_name,
        strip_strays=strip_strays,
    )
