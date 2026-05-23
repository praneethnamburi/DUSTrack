"""Window-close guard: save-on-close modal + recent-session history write.

The DUSTrack window can close for many reasons (X button, Alt+F4,
``plt.close()``); the close-guard hooks the QMainWindow's
``closeEvent`` to:

1. Sweep every ready bundle for in-memory-vs-disk diffs
   (:func:`scan_unsaved_layers_all_bundles`).
2. If any layer is dirty, show the Save / Discard / Cancel modal
   (:func:`prompt_save_on_close`). Cancel aborts the close.
3. On Save, write every dirty layer
   (:func:`save_unsaved_layers`).
4. Append the session to the unified recent-sessions store
   (:func:`record_session_in_history`).

Seed-modal sessions (``_is_seed_session=True``) skip every step --
the synthetic seed asset has no user edits worth prompting about
and shouldn't pollute the recent list.

Both logic (diff sweep, history write) and the single modal live
here because the file would otherwise be ~50 lines of orchestration
split across two modules. The diff primitives themselves are in
:mod:`._preflight`.

Extracted from ``gui.DUSTrack`` in the 1.2.0rc1 follow-up.
"""

from __future__ import annotations

from pathlib import Path

from . import _config
from ._overlays import _make_confirm_overlay_class
from ._preflight import format_unsaved_summary, scan_unsaved_layers


def scan_unsaved_layers_all_bundles(bundles) -> dict:
    """Sweep every ``ready`` bundle for in-memory-vs-disk diffs.

    Returns ``{video_index: {"fname": Path, "layers":
    {layer_name: diff}}}`` for bundles with at least one unsaved
    layer; empty when nothing is dirty. Pending / hydrating /
    failed bundles are skipped (their data isn't in memory, so the
    on-disk state IS the only state -- nothing to lose).
    """
    result: dict = {}
    for bundle in bundles:
        if not bundle.is_ready or bundle.annotations is None:
            continue
        layers = scan_unsaved_layers(bundle.annotations, str(bundle.fname))
        if layers:
            result[bundle.video_index] = {
                "fname": bundle.fname,
                "layers": layers,
            }
    return result


def save_unsaved_layers(unsaved, bundles) -> None:
    """Persist every layer with diffs across the given bundles.

    Accepts the multi-bundle shape ``{video_index: {"fname": Path,
    "layers": {layer_name: diff}}}`` -- callers always normalise to
    this shape before calling.
    """
    for video_index, info in unsaved.items():
        bundle = bundles[video_index]
        if bundle.annotations is None:
            continue
        for layer_name in info["layers"]:
            if layer_name in bundle.annotations.names:
                bundle.annotations[layer_name].save()


def prompt_save_on_close(qt_window, unsaved) -> str:
    """Modal triggered by the save-on-close guard. Returns the user's
    choice as one of ``"save"`` / ``"discard"`` / ``"cancel"``.

    ``unsaved`` is either the legacy single-bundle shape
    (``{layer_name: diff}``) or the 1.2.0a3+ per-bundle shape
    ``{video_index: {"fname": Path, "layers": {layer_name: diff}}}``.
    The modal renders each bundle's layers in a separate block so
    users can see which video each diff belongs to.

    *Save* writes every layer with diffs and lets the window close;
    *Discard* lets the window close without writing; *Cancel* keeps
    the window open. ``Cancel`` is the default button so that
    accidental Enter / Esc do not silently lose data.
    """
    ConfirmOverlay = _make_confirm_overlay_class()
    if unsaved and "fname" in next(iter(unsaved.values()), {}):
        blocks = []
        total_layers = 0
        for video_index, info in unsaved.items():
            layers = info["layers"]
            if not layers:
                continue
            total_layers += len(layers)
            blocks.append(
                f"  {Path(info['fname']).name} (video {video_index + 1}):\n"
                f"{format_unsaved_summary(layers)}"
            )
        breakdown = "\n\n".join(blocks)
        n_videos = sum(1 for info in unsaved.values() if info.get("layers"))
        header = (
            f"{total_layers} annotation layer"
            f"{'s' if total_layers != 1 else ''} across "
            f"{n_videos} video{'s' if n_videos != 1 else ''} "
            f"{'have' if total_layers != 1 else 'has'} unsaved changes."
        )
    else:
        n = len(unsaved)
        header = (
            f"{n} annotation layer{'s' if n != 1 else ''} "
            f"{'have' if n != 1 else 'has'} unsaved changes."
        )
        breakdown = format_unsaved_summary(unsaved)
    body = (
        f"{header}\n\n"
        f"{breakdown}\n\n"
        "Save all writes the in-memory edits to disk before closing.\n"
        "Discard closes without writing -- changes are lost.\n"
        "Cancel keeps the window open."
    )
    result = ConfirmOverlay(
        qt_window,
        title="Unsaved annotations",
        message=body,
        buttons=[
            ("Save all", "primary"),
            ("Discard", "destructive"),
            ("Cancel", "neutral"),
        ],
        default="Cancel",
        severity="destructive",
    ).exec_()
    if result == "Save all":
        return "save"
    if result == "Discard":
        return "discard"
    return "cancel"


def record_session_in_history(dustrack) -> None:
    """Write the current session's full bundle list to the unified
    ``recent_sessions`` store.

    Single-video sessions write a 1-element entry; multi-video
    sessions write the full bundle list in queue order (bundle 0
    first, then the tail). The active video is always the first
    element so a click-to-reopen lands on the same video the user
    was on.

    Skipped for seed sessions (``_is_seed_session=True`` on the
    tracker); recording the synthetic asset path would pollute the
    recent list with an entry that's never useful to reopen.

    Best-effort -- if the JSON store is unwritable, drops the entry
    but doesn't raise.
    """
    if getattr(dustrack, "_is_seed_session", False):
        return
    fname = getattr(dustrack, "fname", None)
    if not fname:
        return
    bundles = getattr(dustrack, "_bundles", None) or []
    if bundles:
        paths = [b.fname for b in bundles]
    else:
        paths = [fname]
    try:
        _config.record_recent_session(paths)
    except Exception:
        pass


def install_close_guard(dustrack) -> None:
    """Patch the QMainWindow ``closeEvent`` so window close triggers
    the unsaved-diffs scan + modal + history write.

    Monkey-patch rather than subclass because the QMainWindow is
    constructed inside matplotlib's Qt backend; intercepting it
    without owning the type means patching the instance. The
    original ``closeEvent`` is chained at the end so any
    backend-internal cleanup still runs.

    No-op on the mpl fallback path (no Qt window to hook).
    Idempotent: a second call on the same window is a no-op so
    subclass re-init doesn't stack handlers.
    """
    qt_window = dustrack._find_qt_window()
    if qt_window is None:
        return
    if getattr(qt_window, "_dustrack_close_guard_installed", False):
        return

    original_close_event = qt_window.closeEvent

    def closeEvent(event):
        # Seed-session short-circuit: synthetic seed asset, no user
        # edits, do not write to history.
        if getattr(dustrack, "_is_seed_session", False):
            original_close_event(event)
            return
        try:
            # Route through the dustrack methods (which delegate back
            # here) rather than calling the module functions directly,
            # so test harnesses that stub the methods on a mock
            # dustrack see the expected call graph.
            unsaved = dustrack._scan_unsaved_layers_all_bundles()
        except Exception:
            # Guard is a safety net, not a hard gate -- never block
            # the close on a scan failure.
            unsaved = {}
        if unsaved:
            choice = dustrack._prompt_save_on_close(qt_window, unsaved)
            if choice == "cancel":
                event.ignore()
                return
            if choice == "save":
                dustrack._save_unsaved_layers(unsaved)
        # History write after the unsaved-diff gate so a cancelled
        # close does NOT pollute the recent list.
        try:
            dustrack._record_session_in_history()
        except Exception:
            pass
        original_close_event(event)

    qt_window.closeEvent = closeEvent
    qt_window._dustrack_close_guard_installed = True
