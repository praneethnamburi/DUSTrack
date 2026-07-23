"""Qt UI wrapper for the diverse-frame selection (decimation) review modal.

The DINOv3-index feature (spec roadmap #1): the frame set is embedded
off-thread first (in the ``gui`` workflow method, under a ProgressOverlay),
then this modal lets the user set the target count + auto-balance toggle,
re-run the cheap selection over the cached embeddings, review the picks as a
gallery, and confirm. Pairs with :mod:`dustrack.imagesimilarity` (the engine)
and :mod:`dustrack.gui` (the workflow method that composes the embed pass with
this modal). The dialog factory lives in :mod:`._overlays` alongside the other
dialog factories. Mirrors :mod:`dustrack._blip_modal`'s thin-wrapper shape.
"""
from __future__ import annotations

from ._overlays import _make_decimate_gallery_class, _make_source_selection_class


def prompt_source_selection(qt_window, *, layer_names, current_layer,
                            across_enabled=False):
    """Ask WHAT the diverse selection reads before the embed pass runs.

    Args:
        qt_window: The DUSTrack QMainWindow (overlay parent).
        layer_names: Every annotation-layer name (the "pick a layer" dropdown).
        current_layer: The active layer's name (the default source).
        across_enabled: Whether the "across all videos" scope is offered yet
            (the single-video case ships first; this stays ``False`` until the
            multi-video generator lands).

    Returns:
        ``(layer_name, scope)`` on Continue -- ``scope`` is ``"video"`` or
        ``"across"`` -- or ``None`` on Cancel.
    """
    Dialog = _make_source_selection_class()
    return Dialog(
        qt_window,
        layer_names=layer_names,
        current_layer=current_layer,
        across_enabled=across_enabled,
    ).exec_()


def prompt_decimate_gallery(qt_window, *, total, thumbs, select_fn, default_count):
    """Show the diverse-selection review modal and return the chosen indices.

    Args:
        qt_window: The DUSTrack QMainWindow (overlay parent).
        total: Number of frames in the embedded set (spinbox upper bound).
        thumbs: One small ``uint8`` gray thumbnail array per embedded frame,
            index-aligned to the embeddings the ``select_fn`` selects over.
        select_fn: ``(count, balance) -> row indices``; re-runs
            :func:`dustrack.imagesimilarity.select_diverse` over the cached
            embeddings (cheap, no re-embed).
        default_count: Initial spinbox value (the 50%-of-set default).

    Returns:
        The list of selected row indices on confirm, ``None`` on cancel.
    """
    Dialog = _make_decimate_gallery_class()
    return Dialog(
        qt_window,
        total=total,
        thumbs=thumbs,
        select_fn=select_fn,
        default_count=default_count,
    ).exec_()
