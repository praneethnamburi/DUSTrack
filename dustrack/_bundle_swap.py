"""Multi-video swap state machine + bundle list management.

The swap contract from ``specs/dustrack.md`` Roadmap *Next 1.2.0*
item 3, factored out of ``gui.DUSTrack``:

1. :func:`init_bundles` -- populate ``_bundles`` from the
   just-constructed shell + a list of queued paths; spawn the
   background hydration worker for the tail.
2. :func:`swap_to` -- snapshot leaving bundle, park its artists,
   rebind shell, show arriving bundle's artists, restore selections
   + viewports + enhance, repaint.
3. :func:`add_video` / :func:`remove_video` / :func:`replace_active_with`
   -- bundle list mutations that survive swap-back.
4. :func:`install_broadcast_statevar_hooks` /
   :func:`broadcast_statevar` -- per-statevar cross-bundle
   propagation for UI-mode flavoured statevars (number_keys,
   annotation_label, label_range).
5. :func:`capture_statevar_selections` /
   :func:`restore_statevar_selections` -- the silent snapshot /
   restore pair used by swap_to and bundle init.
6. :func:`await_hydration` -- block (pumping Qt) until a pending
   bundle reaches a terminal state.

All functions take ``dustrack`` (the shell tracker) as the first
arg and read / write shell state through it; this makes the
swap-state machinery testable with a mock dustrack that just has
the right attribute shape.

Pairs with :mod:`._bundle` (the per-bundle hydration half).

Extracted from ``gui.DUSTrack`` in the 1.2.0rc1 follow-up.
"""

from __future__ import annotations

import time as _time
from pathlib import Path

from ._bundle import (
    HYDRATION_FAILED,
    HYDRATION_PENDING,
    HYDRATION_READY,
    _BgHydrationWorker,
    _BundleState,
)


# ---------------------------------------------------------------------
# Bundle init + cross-bundle broadcasts
# ---------------------------------------------------------------------


def init_bundles(dustrack, project, video_paths: list) -> None:
    """Populate ``dustrack._bundles`` from the just-constructed shell +
    a list of queued video paths.

    Called by :func:`dustrack.open` (and friends) after the
    active-bundle ``DUSTrack`` is constructed. The just-built
    annotations / VideoReader become bundle 0 (``ready`` from the
    start); pending bundles are scaffolded for each path in
    ``video_paths[1:]`` and hydrated by the background worker.
    """
    if len(video_paths) == 0:
        raise ValueError("init_bundles: video_paths cannot be empty")

    # Bundle 0 -- snapshot the just-constructed shell.
    active_bundle = _BundleState(
        fname=Path(video_paths[0]),
        video_index=0,
        project=project,
        reader=dustrack.data,
        annotations=dustrack.annotations,
        current_idx=dustrack._current_idx,
        ax_lims=dict(dustrack._ax_lims),
        image_view_state=dustrack._get_image_view_state(),
        frames_of_interest=list(dustrack.frames_of_interest),
        hydration_state=HYDRATION_READY,
    )
    active_bundle.selections = capture_statevar_selections(dustrack)
    dustrack._bundles = [active_bundle]

    # Pending bundles for the tail. All share the same project as
    # bundle 0 (multi-video contract: same-project only).
    for i, vp in enumerate(video_paths[1:], start=1):
        dustrack._bundles.append(
            _BundleState(
                fname=Path(vp),
                video_index=i,
                project=project,
                hydration_state=HYDRATION_PENDING,
            )
        )

    dustrack._active_index = 0
    # Back-compat: ``_video_queue`` (set since 1.2.0a2 by
    # :func:`dustrack.open`) remains as the tail-of-paths
    # observability attribute. Kept in sync with the bundle list.
    dustrack._video_queue = [b.fname for b in dustrack._bundles[1:]]

    dustrack._hydration_worker = None
    if project is not None and len(dustrack._bundles) > 1:
        dustrack._hydration_worker = _BgHydrationWorker(
            dustrack,
            project,
            dustrack._bundles[1:],
        )
        dustrack._hydration_worker.start()

    # Wire broadcast statevars (annotation_label / label_range /
    # number_keys carry across same-project videos).
    install_broadcast_statevar_hooks(dustrack)

    dustrack._refresh_nav_buttons()

    # Schedule a synchronous paint that fires AFTER the Qt event loop
    # starts. plt.show(block=True) hasn't been called yet; any inline
    # canvas.draw() renders into a buffer the window can't display.
    # QTimer.singleShot(0, ...) defers to the first idle after the
    # event loop is running. Same root cause + fix shape as the tail
    # of swap_to.
    try:
        from qtpy.QtCore import QTimer

        QTimer.singleShot(0, dustrack.figure.canvas.draw)
    except Exception:  # noqa: BLE001
        try:
            dustrack.figure.canvas.draw()
        except Exception:  # noqa: BLE001
            pass


def install_broadcast_statevar_hooks(dustrack) -> None:
    """Wire ``add_on_change`` callbacks on every broadcast statevar
    so user-driven mutations propagate to every bundle's ``selections``
    dict (including pending bundles).

    Idempotent guard via ``_broadcast_hooks_installed`` so a subclass
    re-entering ``__init__`` doesn't stack callbacks.
    """
    if getattr(dustrack, "_broadcast_hooks_installed", False):
        return
    for sv_name in dustrack._BROADCAST_STATEVARS:
        if sv_name not in dustrack.statevariables.names:
            continue
        sv = dustrack.statevariables[sv_name]
        sv.add_on_change(
            # Late-binding gotcha: bind by default-arg so each closure
            # captures its own name.
            lambda _name=sv_name: broadcast_statevar(dustrack, _name),
        )
    dustrack._broadcast_hooks_installed = True


def broadcast_statevar(dustrack, sv_name: str) -> None:
    """Write the shell's current value for ``sv_name`` into every
    bundle's ``selections`` dict.

    Silent-restore in :func:`restore_statevar_selections` bypasses the
    on_change callback chain, so this fires only on genuine user
    mutations (combo box pick, key cycle) -- not on swap-in restores.
    """
    if sv_name not in dustrack.statevariables.names:
        return
    new_value = dustrack.statevariables[sv_name].current_state
    for bundle in dustrack._bundles:
        bundle.selections[sv_name] = new_value


# ---------------------------------------------------------------------
# Statevar snapshot / restore (used by swap_to)
# ---------------------------------------------------------------------


def capture_statevar_selections(dustrack) -> dict:
    """Snapshot the shell's current statevar selections for the
    active bundle. Names absent from the container are omitted
    (mpl-fallback path may skip some Qt-only statevars).
    """
    out: dict = {}
    for sv in dustrack._ALL_TRACKED_STATEVARS:
        if sv in dustrack.statevariables.names:
            out[sv] = dustrack.statevariables[sv].current_state
    return out


def restore_statevar_selections(
    dustrack,
    selections: dict,
    layer_names: list,
) -> None:
    """Rewrite each statevar's ``states`` list to the new bundle's
    rotation, restore the snapshotted selection silently (bypass
    on_change callbacks so the per-statevar cascade doesn't fire
    during the restore), then refresh the Qt sidebar widgets in one
    ``_text.update()`` call.

    Silent restore matters because on_change callbacks include
    :func:`._workflow_gates.refresh_workflow_button_state` and
    ``_on_active_label_change`` -- firing them mid-restore would
    re-read partially-rebuilt state. The pattern (direct
    ``_current_state_idx = ...`` + manual ``_text.update()``) mirrors
    ``select_label_with_mouse``'s existing bypass.
    """
    # 1. Rewrite the rotations from the new bundle's layer list.
    if "annotation_layer" in dustrack.statevariables.names:
        sv = dustrack.statevariables["annotation_layer"]
        sv.states = list(layer_names)
    if "annotation_overlay" in dustrack.statevariables.names:
        sv = dustrack.statevariables["annotation_overlay"]
        sv.states = [None] + list(layer_names)

    # 2. Restore each snapshotted selection. annotation_layer /
    # annotation_overlay must come BEFORE annotation_label so the
    # active layer is set when we re-derive the label rotation.
    for sv_name in dustrack._ALL_TRACKED_STATEVARS:
        if sv_name not in dustrack.statevariables.names:
            continue
        if sv_name not in selections:
            continue
        sv = dustrack.statevariables[sv_name]
        value = selections[sv_name]
        try:
            idx = sv.states.index(value)
        except ValueError:
            # Snapshot value isn't valid for this bundle. Fall back
            # to the first state.
            idx = 0
        sv._current_state_idx = idx
        # For annotation_label specifically, also refresh the label
        # rotation against the new active layer before locking in
        # the selection.
        if sv_name == "annotation_layer":
            dustrack.update_annotation_label_states()

    # 3. Sync the Qt sidebar widgets to the new states + selections.
    try:
        if dustrack.statevariables._text is not None:
            dustrack.statevariables._text.update()
    except Exception:  # noqa: BLE001 - mpl-fallback / pre-teardown
        pass


# ---------------------------------------------------------------------
# Swap entry points
# ---------------------------------------------------------------------


def await_hydration(bundle: _BundleState) -> bool:
    """Block (pumping the Qt event loop) until ``bundle`` reaches
    a terminal state.

    Returns ``True`` if the bundle is ready, ``False`` if it failed.
    Used by :func:`swap_to` when the user clicks ahead of the
    background hydration worker.
    """
    if bundle.is_terminal:
        return bundle.is_ready
    try:
        from qtpy.QtCore import QCoreApplication

        qt_pump = QCoreApplication.processEvents
    except Exception:  # noqa: BLE001
        qt_pump = None
    deadline_per_tick = 0.02  # 50 Hz poll
    while not bundle.is_terminal:
        if qt_pump is not None:
            qt_pump()
        _time.sleep(deadline_per_tick)
    return bundle.is_ready


def swap_to(dustrack, index: int) -> bool:
    """Switch the active video to ``dustrack._bundles[index]``.

    Implements the swap contract:

    1. Snapshot the active bundle's per-video state.
    2. Park the leaving bundle's artists.
    3. Rebind shell attributes onto the arriving bundle.
    4. Show the arriving bundle's artists.
    5. Restore the arriving bundle's statevar selections + image /
       trace / enhance state.
    6. Repaint once.

    Returns ``True`` on a successful swap (or no-op when ``index``
    is already active), ``False`` when the swap was rejected
    (out-of-bounds, non-ready bundle).
    """
    if not (0 <= index < len(dustrack._bundles)):
        return False
    if index == dustrack._active_index:
        return True
    target = dustrack._bundles[index]
    if target.hydration_state == HYDRATION_FAILED:
        dustrack._notify_bundle_failure(target)
        return False
    if not target.is_ready:
        ready = dustrack._await_hydration(target)
        if not ready:
            dustrack._notify_bundle_failure(target)
            return False

    # 1. Snapshot leaving bundle.
    snapshot_active_bundle(dustrack)

    # 2. Park leaving bundle's artists.
    leaving = dustrack._bundles[dustrack._active_index]
    dustrack._park_bundle_artists(leaving)

    # 3. Rebind shell onto arriving bundle.
    attach_bundle(dustrack, target)

    # 4. Show arriving bundle's artists.
    dustrack._show_bundle_artists(target)

    # 4b. Restore (or first-time-fit) the trace axes view. If the
    # arriving bundle has a captured trace_view_state, restore it.
    # If not (first visit), apply a default fit (xlim = (0,
    # n_frames), autoscale-y on) so the trace pane shows the new
    # video's data at the right scale.
    if not dustrack._ax_lims["state"]:
        if target.trace_view_state is not None:
            dustrack._set_trace_view_state(target.trace_view_state)
            # Marker cache keys on (current_label, per-ann revisions,
            # FOI). After restore, the revisions don't match the
            # leaving bundle's cache, so the next paint recomputes
            # anyway -- but clear defensively.
            dustrack._frame_marker_cache = None
        else:
            # First visit -- setup_display_trace only claims set_xlim
            # while autoscalex_on is True, and the leaving bundle's
            # setup turned that off. Force-fit + re-enable autoscale.
            n_frames = len(target.reader)
            dustrack._ax_trace_x.set_xlim(0, n_frames)
            dustrack._ax_trace_x.set_autoscalex_on(True)
            dustrack._ax_trace_x.set_autoscaley_on(True)
            dustrack._ax_trace_y.set_autoscaley_on(True)
            dustrack._frame_marker_cache = None

    # 5. Restore statevars + image viewport + enhance state.
    restore_statevar_selections(
        dustrack,
        target.selections,
        target.annotations.names,
    )
    dustrack._set_image_view_state(target.image_view_state)
    dustrack._set_enhance_state(target.enhance_state)

    # 6. Repaint.
    dustrack._active_index = index
    dustrack._refresh_nav_buttons()
    try:
        dustrack._refresh_workflow_button_state()
    except Exception:  # noqa: BLE001
        pass
    dustrack.update()
    return True


def snapshot_active_bundle(dustrack) -> None:
    """Write the shell's current per-video UI state back to the
    active bundle. Called at the top of every swap so the next swap
    back lands the user where they were.
    """
    if not dustrack._bundles:
        return
    active = dustrack._bundles[dustrack._active_index]
    active.current_idx = dustrack._current_idx
    active.ax_lims = dict(dustrack._ax_lims)
    # Deep copies on the inner lists so subsequent shell mutations
    # don't leak back into the snapshot.
    for k in ("x", "y_trace_x", "y_trace_y"):
        if k in active.ax_lims and isinstance(active.ax_lims[k], list):
            active.ax_lims[k] = list(active.ax_lims[k])
    active.image_view_state = dustrack._get_image_view_state()
    active.trace_view_state = dustrack._get_trace_view_state()
    active.enhance_state = dustrack._get_enhance_state()
    active.frames_of_interest = list(dustrack.frames_of_interest)
    active.selections = capture_statevar_selections(dustrack)


def attach_bundle(dustrack, bundle: _BundleState) -> None:
    """Rebind shell attributes onto ``bundle``'s heavy state.

    Does not touch artists -- :func:`._bundle.park_bundle_artists`
    and :func:`._bundle.show_bundle_artists` handle visibility; this
    function just swaps the data pointers (fname, VideoReader,
    annotations container, DLC project) and the lightweight UI
    snapshot (frame, axis limits, frames of interest). Statevar /
    image-pane restore happens after this in :func:`swap_to`.

    Cross-Phase contract (1.2.0a3 seed-modal cut): bundles can
    carry different ``project`` values (``None`` for Phase 1
    bare-video bundles, a ``DLCProject`` for Phase 2). The rebind
    below pushes the arriving bundle's project onto the shell so
    any Workflow-button gating or other project-aware code reads
    the right value on the next paint.
    """
    dustrack.fname = str(bundle.fname)
    dustrack.data = bundle.reader
    dustrack.annotations = bundle.annotations
    dustrack._dlcproject = bundle.project
    dustrack._current_idx = bundle.current_idx
    # Force a fresh dict so the shell's later mutations don't alias
    # the bundle snapshot.
    dustrack._ax_lims = dict(bundle.ax_lims)
    for k in ("x", "y_trace_x", "y_trace_y"):
        if k in dustrack._ax_lims and isinstance(dustrack._ax_lims[k], list):
            dustrack._ax_lims[k] = list(dustrack._ax_lims[k])
    dustrack.frames_of_interest = list(bundle.frames_of_interest)


# ---------------------------------------------------------------------
# Bundle list management (add / remove / replace-active)
# ---------------------------------------------------------------------


def add_video(
    dustrack,
    path_or_paths,
    *,
    layer_name=None,
    set_active=False,
    **dustrack_kwargs,
) -> list:
    """Append one or more videos to the tracker's bundle list.

    See :meth:`DUSTrack.add_video` for the full contract; this is
    the implementation. Validates + hydrates the picked path(s),
    appends the new bundle(s) to ``_bundles``, optionally swaps to
    the first new bundle.
    """
    project, video_paths = validate_bundle_paths(path_or_paths)

    # Hydrate the first new bundle synchronously so a swap-to
    # immediately after add_video is a no-wait. The tail goes
    # PENDING and the bg worker takes over.
    base_index = len(dustrack._bundles)
    new_bundles: list = []
    first = _BundleState(
        fname=Path(video_paths[0]),
        video_index=base_index,
        project=project,
        hydration_state=HYDRATION_PENDING,
    )
    dustrack._hydrate_bundle_sync(first)
    if first.hydration_state == HYDRATION_FAILED:
        raise RuntimeError(
            f"add_video: hydration failed for {first.fname}: "
            f"{first.hydration_error}"
        )
    new_bundles.append(first)
    for i, vp in enumerate(video_paths[1:], start=1):
        new_bundles.append(
            _BundleState(
                fname=Path(vp),
                video_index=base_index + i,
                project=project,
                hydration_state=HYDRATION_PENDING,
            )
        )
    dustrack._bundles.extend(new_bundles)
    dustrack._video_queue = [b.fname for b in dustrack._bundles[1:]]
    # Kick off the bg worker for any PENDING tail bundles.
    pending_tail = [
        b for b in new_bundles[1:] if b.hydration_state == HYDRATION_PENDING
    ]
    if pending_tail and project is not None:
        worker = _BgHydrationWorker(dustrack, project, pending_tail)
        worker.start()
        # Track most-recent worker for diagnostics; tests can poke it.
        dustrack._hydration_worker = worker
    dustrack._refresh_nav_buttons()
    new_indices = [b.video_index for b in new_bundles]
    if set_active:
        dustrack.swap_to(new_indices[0])
    return new_indices


def remove_video(dustrack, index: int) -> bool:
    """Drop a bundle from the tracker's bundle list. See
    :meth:`DUSTrack.remove_video` for the full contract.
    """
    if not (0 <= index < len(dustrack._bundles)):
        return False
    if len(dustrack._bundles) <= 1:
        return False
    if index == dustrack._active_index:
        # Swap-first: prefer next; fall back to previous when at the
        # tail. The fallback index uses the pre-removal layout, so a
        # swap to (index - 1) lands on the bundle that will end up at
        # (index - 1) post-removal.
        if index + 1 < len(dustrack._bundles):
            target = index + 1
        else:
            target = index - 1
        if not dustrack.swap_to(target):
            return False
    # After the swap, dustrack._active_index no longer equals index.
    leaving = dustrack._bundles[index]
    dustrack._park_bundle_artists(leaving)
    del dustrack._bundles[index]
    if index < dustrack._active_index:
        dustrack._active_index -= 1
    # Renumber surviving bundles so video_index matches the new
    # list position.
    for new_idx, bundle in enumerate(dustrack._bundles):
        bundle.video_index = new_idx
    dustrack._video_queue = [b.fname for b in dustrack._bundles[1:]]
    dustrack._refresh_nav_buttons()
    return True


def replace_active_with(
    dustrack,
    path_or_paths,
    *,
    layer_name=None,
    **dustrack_kwargs,
) -> list:
    """Swap the active bundle for one (or more) newly-picked
    video(s); drop the previously-active bundle. See
    :meth:`DUSTrack.replace_active_with` for the full contract.
    """
    old_active_bundle = dustrack._bundles[dustrack._active_index]
    try:
        new_indices = dustrack.add_video(
            path_or_paths,
            layer_name=layer_name,
            set_active=True,
            **dustrack_kwargs,
        )
    except Exception:
        dustrack._video_queue = [b.fname for b in dustrack._bundles[1:]]
        dustrack._refresh_nav_buttons()
        raise
    # Find old_active_bundle's *current* index post-add. Identity
    # match by object identity, not __eq__.
    old_idx = next(
        (i for i, b in enumerate(dustrack._bundles) if b is old_active_bundle),
        None,
    )
    if old_idx is None:
        return new_indices
    n_new = len(new_indices)
    dustrack.remove_video(old_idx)
    total = len(dustrack._bundles)
    return list(range(total - n_new, total))


def validate_bundle_paths(path_or_paths) -> tuple:
    """Resolve ``path_or_paths`` into ``(project_or_None, [Path...])``.

    Mirrors the validation logic at the top of :func:`dustrack.open`:

    - Single path -> Phase 1 (project=None) if no ``config.yaml``
      is found beside it, else Phase 2 (project resolved).
    - List of paths -> Phase 2 multi (must all belong to one shared
      DLC project). Bare-video entries raise.
    - Project folder -> Phase 2 multi (queue every video in the
      project).
    - ``config.yaml`` -> Phase 2 single on the first project video.
    """
    from .dlcinterface import DLCProject
    from ._dlc_paths import (
        _find_dlc_config,
        _is_dlc_config_yaml,
        _is_dlc_project_root,
        _resolve_multi_video_from_list,
    )
    from .dlcloader import HAS_DLC

    if isinstance(path_or_paths, (list, tuple)):
        if len(path_or_paths) == 0:
            raise ValueError("add_video: empty path sequence")
        paths = [Path(p) for p in path_or_paths]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"add_video: path does not exist: {p}")
        if len(paths) == 1:
            return validate_bundle_paths(paths[0])
        return _resolve_multi_video_from_list(paths)

    p = Path(path_or_paths)
    if not p.exists():
        raise FileNotFoundError(f"add_video: path does not exist: {p}")
    if _is_dlc_config_yaml(p):
        # Mirror dustrack.open's config.yaml dispatch: queue every
        # video in the project, in config['video_sets'] order.
        if not HAS_DLC:
            raise ImportError(
                f"add_video: {p} is a DLC config.yaml but "
                "deeplabcut is not installed."
            )
        project = DLCProject(str(p))
        video_paths = [Path(v) for v in project.video_list]
        if not video_paths:
            raise ValueError(f"add_video: DLC project at {p.parent} has no videos.")
        return project, video_paths
    if p.is_dir():
        if not _is_dlc_project_root(p):
            raise ValueError(
                f"add_video: {p!s} is a directory but doesn't look "
                "like a DLC project."
            )
        if not HAS_DLC:
            raise ImportError(
                f"add_video: detected a DLC project at {p}, but "
                "deeplabcut is not installed."
            )
        project = DLCProject(str(p / "config.yaml"))
        video_paths = [Path(v) for v in project.video_list]
        if not video_paths:
            raise ValueError(f"add_video: DLC project at {p} has no videos.")
        return project, video_paths

    # File. Phase 1 vs Phase 2 split by config.yaml discovery.
    config_path = _find_dlc_config(p)
    if config_path is None:
        return None, [p]
    if not HAS_DLC:
        raise ImportError(
            f"add_video: detected a DLC project at {config_path.parent}, "
            "but deeplabcut is not installed."
        )
    return DLCProject(str(config_path)), [p]
