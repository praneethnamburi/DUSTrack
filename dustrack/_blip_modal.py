"""Qt UI wrapper for the Detect blip outliers options modal.

Two-stage modal: user tunes the three detection knobs (threshold
factor, max blip length, return position factor) + clicks Detect to
populate the in-modal results pane; then Interpolate kicks off the
slow LK pass via the caller's :class:`ProgressOverlay`.

Pairs with :mod:`dustrack.blip` (the algorithm) and :mod:`dustrack.gui`
(the workflow method that composes this modal with the async LK
overlay). The dialog factory itself lives in :mod:`._overlays`
alongside ``TrainingOptionsDialog`` so all Qt dialog factories share
one home.

Mirrors :mod:`dustrack._train_modal`'s shape (thin wrapper that builds
the lazy-imported class then runs ``exec_()``).
"""

from __future__ import annotations

from ._overlays import _make_blip_options_class
from . import blip as _blip_mod


def prompt_blip_options(qt_window, ann):
    """Show the Detect blip outliers modal and return what the user picked.

    Args:
        qt_window: The DUSTrack QMainWindow (overlay parent).
        ann: The active :class:`VideoAnnotation` to scan.

    Returns:
        ``(report, knobs, drop_frame_if_any_blip)`` on Remove blips,
        where ``report`` is a :class:`dustrack.blip.BlipReport` with
        at least one blip, ``knobs`` is the dict of detection-knob
        values used to produce it, and ``drop_frame_if_any_blip`` is
        the checkbox state (``False`` = drop only the blipped label
        at each blip frame; ``True`` = drop every label at any frame
        where any blip was detected). ``None`` if the user clicked
        Cancel.
    """
    BlipOptionsDialog = _make_blip_options_class()
    dialog = BlipOptionsDialog(
        qt_window,
        ann=ann,
        detect_fn=_blip_mod.detect_blips,
    )
    return dialog.exec_()
