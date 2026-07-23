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


def prompt_decimate_gallery(qt_window, *, thumbs, recluster_fn, select_fn,
                            k_lo, k_hi, k_default, dmin_lo, dmin_hi,
                            dmin_default, n_frames):
    """Show the cluster-per-row review modal and return the kept frame indices.

    Args:
        qt_window: The DUSTrack QMainWindow (overlay parent).
        thumbs: One small ``uint8`` gray thumbnail per embedded frame,
            index-aligned to the embeddings.
        recluster_fn: ``k -> (labels, medoids, leaf_order, segments)`` -- re-cuts
            the precomputed hierarchy at ``k`` clusters (cheap, no re-embed):
            per-frame ``labels``, ``{cluster: frame_index}`` medoids, the
            dendrogram ``leaf_order`` (row order that groups siblings), and the
            tree ``segments`` for the left gutter. Drives the Clusters knob.
        select_fn: ``min_dist -> sorted frame indices``; thresholds the cached
            farthest-point capture radii and includes the medoid floor. Drives
            the minimum-distance knob (independent of ``k``).
        k_lo, k_hi, k_default: Clusters-slider range + initial value.
        dmin_lo, dmin_hi, dmin_default: Minimum-distance range (lo = more/closer
            frames, hi = fewer) + initial value.
        n_frames: Total embedded frames (the "of N" in the status line).

    Returns:
        The kept frame indices (from checked clusters) on confirm, ``None`` on
        cancel.
    """
    Dialog = _make_decimate_gallery_class()
    return Dialog(
        qt_window,
        thumbs=thumbs,
        recluster_fn=recluster_fn,
        select_fn=select_fn,
        k_lo=k_lo,
        k_hi=k_hi,
        k_default=k_default,
        dmin_lo=dmin_lo,
        dmin_hi=dmin_hi,
        dmin_default=dmin_default,
        n_frames=n_frames,
    ).exec_()
