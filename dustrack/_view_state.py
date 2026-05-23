"""Per-bundle viewport + enhancement snapshot / restore helpers.

Three independent state buckets are captured on swap-out and
restored on swap-in so each bundle keeps its own pan/zoom/sliders
across multi-video sessions:

* **Image-pane viewport** -- Tier 2 (Qt-native) reads/writes via
  the pane's ``get_view_state`` / ``set_view_state``; Tier 1
  (matplotlib) snapshots the image axis's xlim/ylim. Opaque blob
  in both cases.
* **Trace-axes viewport** -- the two trace axes' xlim/ylim.
* **Enhance sliders** -- CLAHE clip + gamma + brightness.

All six functions take ``dustrack`` and read the attrs they need
off it. They live here (vs. on the class) so the class file stays
under 4 k lines and the swap-state machinery has one canonical
home.

Extracted from ``gui.DUSTrack`` in the 1.2.0rc1 follow-up.
"""
from __future__ import annotations


# ---------------------------------------------------------------------
# Image-pane viewport (Tier 1 + Tier 2 dispatch)
# ---------------------------------------------------------------------


def get_image_view_state(dustrack):
    """Snapshot the current image pane's zoom / pan state.

    Returns an opaque blob the matching :func:`set_image_view_state`
    understands. Tier 2 (Qt-native) wraps QGraphicsView's
    transform + scrollbar positions; Tier 1 (matplotlib) wraps
    the image axis's xlim / ylim. ``None`` = no viewport saved /
    nothing rendered yet (caller restores to fit-frame).
    """
    if dustrack._fast_render:
        pane = dustrack._image_pane
        getter = getattr(pane, "get_view_state", None)
        if getter is None:
            return None
        try:
            return getter()
        except Exception:  # noqa: BLE001 - defensive
            return None
    ax = dustrack._ax_image
    if ax is None:
        return None
    try:
        xlim = tuple(ax.get_xlim())
        ylim = tuple(ax.get_ylim())
    except Exception:  # noqa: BLE001
        return None
    return {"kind": "mpl", "xlim": xlim, "ylim": ylim}


def set_image_view_state(dustrack, state) -> None:
    """Restore a previously-snapshotted viewport. ``None`` falls
    back to fit-frame on Tier 2 (pane's ``reset_view``) or a
    no-op autoscale on Tier 1.
    """
    if dustrack._fast_render:
        pane = dustrack._image_pane
        setter = getattr(pane, "set_view_state", None)
        if setter is not None:
            try:
                setter(state)
            except Exception:  # noqa: BLE001
                pass
        elif state is None:
            reset = getattr(pane, "reset_view", None)
            if reset is not None:
                try:
                    reset()
                except Exception:  # noqa: BLE001
                    pass
        return
    ax = dustrack._ax_image
    if ax is None:
        return
    if state is None or state.get("kind") != "mpl":
        try:
            ax.relim()
            ax.autoscale_view()
        except Exception:  # noqa: BLE001
            pass
        return
    try:
        ax.set_xlim(state["xlim"])
        ax.set_ylim(state["ylim"])
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------
# Trace-axes viewport
# ---------------------------------------------------------------------


def get_trace_view_state(dustrack) -> dict:
    """Snapshot the trace axes' current xlim / ylim so a swap-out
    can preserve the user's pan/zoom on the trace pane the same
    way :func:`get_image_view_state` does for the image pane.

    Captures both trace axes (x and y) -- the marker, FOI ticks,
    and the per-label trace lines all share these two axes, and
    a returning swap should land back on the exact view the user
    left.
    """
    return {
        "trace_x_xlim": tuple(dustrack._ax_trace_x.get_xlim()),
        "trace_x_ylim": tuple(dustrack._ax_trace_x.get_ylim()),
        "trace_y_ylim": tuple(dustrack._ax_trace_y.get_ylim()),
    }


def set_trace_view_state(dustrack, state) -> None:
    """Restore a previously-snapshotted trace axes view. ``None``
    means "first visit to this bundle" -- caller applies the
    default fit (xlim 0..n_frames, autoscale-y on) instead.
    """
    if state is None:
        return
    try:
        dustrack._ax_trace_x.set_xlim(state["trace_x_xlim"])
        dustrack._ax_trace_x.set_ylim(state["trace_x_ylim"])
        dustrack._ax_trace_y.set_ylim(state["trace_y_ylim"])
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------
# Enhance sliders (CLAHE clip + gamma + brightness)
# ---------------------------------------------------------------------


def get_enhance_state(dustrack) -> dict:
    """Snapshot the shell's current CLAHE / gamma / brightness
    values so a swap-out can preserve them per-bundle. The
    EnhanceWidget sliders bind to these shell attributes; on
    swap-in :func:`set_enhance_state` pushes the restored values
    back into the widget so the sliders move to match.
    """
    return {
        "clahe_clip": float(dustrack._clahe_clip),
        "gamma": float(dustrack._gamma),
        "brightness": float(dustrack._brightness),
    }


def set_enhance_state(dustrack, state) -> None:
    """Restore a previously-snapshotted enhance state, or reset to
    construction-time defaults on a first-visit (``state is None``).

    Pre-fix this returned early on first-visit, which meant the new
    bundle inherited the leaving bundle's slider positions. Restoring
    to ``_initial_enhance_state`` on first-visit gives each bundle
    a clean baseline; user changes still persist via the per-bundle
    snapshot taken on swap-out.

    Pushes new slider positions into the EnhanceWidget if it's
    mounted (Tier 2 / Qt path) so the visible slider knobs match.
    """
    if state is None:
        state = getattr(dustrack, "_initial_enhance_state", None)
    if state is None:
        return
    dustrack._clahe_clip = float(state["clahe_clip"])
    dustrack._gamma = float(state["gamma"])
    dustrack._brightness = float(state.get("brightness", 0))
    widget = getattr(dustrack, "_enhance_widget", None)
    if widget is None:
        return
    # EnhanceWidget exposes a sync helper that updates slider knobs +
    # numeric labels in one go without triggering the per-slider
    # on-change cascade.
    sync = getattr(widget, "sync_from_shell", None)
    if sync is None:
        return
    try:
        sync()
    except Exception:  # noqa: BLE001
        pass
