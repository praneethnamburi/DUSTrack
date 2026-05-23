"""``dustrack.open`` -- unified entry point and dispatch helpers.

:func:`open` is the user-facing function the ``dustrack`` shell command
and ``import dustrack`` callers use to start a tracking session. It
auto-dispatches between:

* Zero-arg: pop a welcome modal seeded against the packaged synthetic
  ``seed_video.mp4`` (:func:`_open_seed_session`).
* Phase 1: a bare video, no DLC project context. Equivalent to
  ``DUSTrack(path, layer_name, ...)``.
* Phase 2 single: one video inside a DLC project. Resolves the
  :class:`DLCProject` and dispatches to :meth:`DLCProject.annotate`.
* Phase 2 multi: every video in a project (or a subset list).
  Constructs the active bundle synchronously and queues the rest for
  background hydration via :class:`_BgHydrationWorker`.

Path-classification helpers (:func:`_is_dlc_config_yaml` /
:func:`_is_dlc_project_root` / :func:`_find_dlc_config` /
:func:`_find_video_index`) live here because they're only consumed by
``open()``'s dispatch logic.

Extracted from ``dlcinterface.py`` in dustrack 1.2.0rc1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from . import _config
from ._overlays import (
    _make_open_video_overlay_class,
    _prompt_for_videos,
    _show_first_paint_notice,
)
from .dlcinterface import DLCProject
from ._dlc_paths import (
    _find_dlc_config,
    _find_video_index,
    _is_dlc_config_yaml,
    _is_dlc_project_root,
    _resolve_multi_video_from_list,
)
from .dlcloader import HAS_DLC, _ensure_dlc_loaded_async
from .gui import DUSTrack


# Packaged synthetic seed video used by the no-arg
# :func:`dustrack.open` welcome modal. 8 frames of 64x64 mid-gray
# h264 (~1.7 KB); co-shipped ``.mp4.dnav-toc`` skips the first-launch
# TOC build. Regeneratable via ``tests/_assets/build_seed_video.py``.
_SEED_VIDEO_PATH = Path(__file__).resolve().parent / "_data" / "seed_video.mp4"


def _open_seed_session(**dustrack_kwargs):
    """Construct a DUSTrack instance against the packaged seed video.

    Used by :func:`dustrack.open` when called with no path -- the
    seed-tracker hosts the welcome modal, then gets its active bundle
    swapped for the user's pick via
    :meth:`DUSTrack.replace_active_with` (and the seed bundle dropped).

    Marks the tracker with ``_is_seed_session = True`` so the close-
    guard skips the save-on-close prompt and the history writer
    skips writing the seed asset to ``recent_sessions``.

    Raises:
        FileNotFoundError: Seed video asset missing from the install
            (build failure / corrupted wheel).
        Exception: Anything raised by the underlying DUSTrack
            construction -- caller treats as "seed-modal path
            unavailable" and falls back to the legacy direct picker.
    """
    if not _SEED_VIDEO_PATH.is_file():
        raise FileNotFoundError(f"seed video asset missing: {_SEED_VIDEO_PATH}")
    # Phase 1 layer naming: ``_seed`` keeps the seed annotations
    # distinct from any real session's ``iteration-0`` layer (in
    # particular, the seed's ``buffer`` layer won't collide with a
    # future ``add_video`` of the same video).
    tracker = DUSTrack(str(_SEED_VIDEO_PATH), "_seed", **dustrack_kwargs)
    tracker._is_seed_session = True
    # ``_init_bundles`` is normally called from
    # :func:`_attach_bundles_or_fallback` post-construction. Run it
    # here so the seed tracker has its single bundle wired up before
    # the modal opens (``replace_active_with`` expects an active
    # bundle to swap from).
    init = getattr(tracker, "_init_bundles", None)
    if init is not None:
        init(project=None, video_paths=[_SEED_VIDEO_PATH])
    return tracker


def open(path=None, layer_name=None, **dustrack_kwargs):
    """Open a DUSTrack annotation session; auto-resolves single- vs multi-video.

    The unified entry point for the DUSTrack workflow. Users hand it a
    path (or a list of paths) and DUSTrack figures out whether they're
    starting fresh on a standalone video (Phase 1), resuming inside a
    DLC project on one video (Phase 2 single), or queueing every video
    in a DLC project for in-session swap navigation (Phase 2 multi,
    1.2.0a3).

    **Phase 1 -- bare video, no DLC project context.**
        Equivalent to ``DUSTrack(path, layer_name, **kwargs)``. Works
        without ``deeplabcut`` installed -- the GUI plus the LK-RSTC
        post-processing run standalone, which is the "Option 1"
        install path from the paper. Single-video only; multi-video
        requires a DLC project (see Phase 2 multi).

    **Phase 2 single -- one video inside a DLC project.**
        Accepts a video inside a project's ``videos/`` folder.
        Resolves the :class:`DLCProject` and dispatches to
        :meth:`DLCProject.annotate` so a fresh DUSTrack opens with
        all existing annotation layers, DLC trace overlays, and a
        new iteration layer wired up.

    **Phase 2 multi -- the whole project (or a subset).** (1.2.0a3)
        Three entry shapes:

        - **Project folder**: ``dustrack.open('S:/path/to/project/')``
          queues every video in ``project.config['video_sets']`` (in
          project order). Behavior change vs <=1.2.0a2 (which opened
          only the first video).
        - **DLC config.yaml**: ``dustrack.open('.../config.yaml')``
          queues every video in ``project.config['video_sets']``,
          same as the folder form. Behavior change vs <=1.2.0a3-pre
          (which opened video 0 only). ``DLCProject.__init__`` runs
          :func:`rebase_to_config` on each video_sets key, so stale
          paths after a project-folder rename self-heal here.
        - **List of videos**: ``dustrack.open([v0, v1, ...])`` queues
          exactly those videos, in the given order. Every entry must
          resolve to the same DLC project; mismatches and bare-video
          entries raise. The list-form is the way to override the
          YAML-stored order (e.g. Ctrl+click in the file dialog).

        The active session opens on the first queued video; the rest
        are background-hydrated and reachable via the sidebar's arrow
        nav row (or ``Alt+Left`` / ``Alt+Right``). Per-video state
        (active layer, overlay, frame, frozen axes, image-pane zoom,
        unsaved edits) persists across swaps -- swap-back returns to
        the exact visual state the user left.

    The two-phase split mirrors DUSTrack's deliberate copy-on-project-
    creation design: once a DLC project exists, the project folder is
    the workspace and the original video becomes a frozen "rewind point"
    (delete the folder to start over). ``open()`` honors that boundary
    -- pointing at the original video gives you Phase 1, pointing at
    the in-project copy gives you Phase 2.

    Args:
        path: Video file, ``config.yaml``, DLC project folder, a
            sequence of videos inside one DLC project, or ``None`` --
            ``None`` pops a Qt file picker and lets the user pick one
            or more videos.
        layer_name: Annotation layer name. Optional in both phases:
            Phase 1 defaults to ``'iteration-0'`` (the canonical seed
            name for the rest of the DLC pipeline -- the next DLC
            training iteration lands as ``iteration-1``); Phase 2
            defaults to ``iteration-{N+1}`` (the next-iteration suffix
            derived from the project's training history). Callers can
            still pass an explicit name to override. Ignored for the
            project-folder multi-video form.
        **dustrack_kwargs: Forwarded to the underlying :class:`DUSTrack`
            constructor (``dark_mode``, ``fast_render``, ``clahe_clip``,
            ``gamma``, ``brightness``, etc.).

    Returns:
        DUSTrack: Live annotation UI, ready to use. ``None`` if the
        no-arg form's file picker was cancelled.

    Raises:
        FileNotFoundError: If ``path`` doesn't exist (or any entry in
            a list form is missing).
        ValueError: Path is a directory that isn't a DLC project, an
            empty sequence was supplied, a multi-video list mixes
            DLC projects, or a multi-video list includes any
            Phase 1 (bare-video) entry.
        ImportError: Phase 2 entry on a system without ``deeplabcut``
            installed.

    Examples:
        Zero-argument launch (pops a video picker)::

            import dustrack
            tracker = dustrack.open()

        Fresh annotation (default layer name ``'iteration-0'``)::

            tracker = dustrack.open('video.mp4')

        Multi-video launch from a DLC project folder (queues every
        video in the project)::

            tracker = dustrack.open('S:/path/to/project/')

        Multi-video launch from a subset of project videos::

            tracker = dustrack.open(['S:/proj/videos/v0.mp4',
                                     'S:/proj/videos/v3.mp4'])

        Resume after closing the UI mid-workflow (any of these work
        as single-video entries)::

            tracker = dustrack.open('S:/path/to/project/videos/video.mp4')
            tracker = dustrack.open('S:/path/to/project/config.yaml')

        With UI options::

            tracker = dustrack.open('video.mp4', 'manual', dark_mode=True)
    """
    if path is None:
        # No-arg form (1.2.0a3 seed-modal flow): construct DUSTrack
        # against a packaged synthetic seed video, mount the welcome
        # modal on top, and -- on a successful pick -- swap the
        # active bundle in-place via ``replace_active_with``. The
        # seed bundle is dropped during the replace; the returned
        # tracker is identity-equivalent to a tracker built directly
        # from the picked path. Defensive fallback: if the seed
        # video fails to load (corrupt asset, codec mismatch) or no
        # Qt window is available (headless / mpl-only), fall through
        # to the pre-1.2.0a3 direct-picker flow.
        seed_tracker = None
        try:
            seed_tracker = _open_seed_session(**dustrack_kwargs)
        except Exception:  # noqa: BLE001
            seed_tracker = None
        if seed_tracker is not None and seed_tracker._find_qt_window() is not None:
            # Mount the welcome overlay. exec_() blocks until the user
            # picks or closes the window. We fire the DLC preload here
            # (post-seed-construction, pre-modal-exec) so the ~7 s
            # DLC import overlaps with the user's time-in-modal.
            _ensure_dlc_loaded_async()
            qt_window = seed_tracker._find_qt_window()
            OpenVideoOverlay = _make_open_video_overlay_class()
            recent = _config.get_recent_sessions()
            picked = OpenVideoOverlay(
                qt_window,
                recent_sessions=recent,
            ).exec_()
            if picked is None:
                # Window X / dismiss -> close the seed tracker and
                # exit cleanly. The close-guard short-circuits the
                # save-prompt + history-write on
                # ``_is_seed_session = True``.
                try:
                    plt.close(seed_tracker.figure)
                except Exception:  # noqa: BLE001
                    pass
                return None
            # User picked. Swap the seed for the picked path(s)
            # in-place; the seed bundle is removed during the swap.
            seed_tracker._is_seed_session = False
            try:
                seed_tracker.replace_active_with(
                    picked,
                    layer_name=layer_name,
                )
            except Exception:
                # If hydration of the picked path failed, the
                # tracker is left holding only the seed bundle. The
                # cleanest exit is to surface the exception; the
                # CLI's traceback rendering handles it.
                seed_tracker._is_seed_session = True
                raise
            _show_first_paint_notice(seed_tracker)
            return seed_tracker
        # Fallback: seed construction failed OR no Qt window.
        if seed_tracker is not None:
            try:
                plt.close(seed_tracker.figure)
            except Exception:  # noqa: BLE001
                pass
        picked = _prompt_for_videos()
        if picked is None:
            return None
        path = picked

    # Kick off the DLC preload now (either right after the picker
    # returned, or immediately if a path was supplied). Idempotent
    # and cheap when DLC is missing (``find_spec`` short-circuits
    # without spawning a thread). Workflow-button gating in
    # ``DUSTrack.__init__`` reads the loader state to keep Create
    # DLC Project disabled until the import resolves.
    _ensure_dlc_loaded_async()

    # Normalise the input into either a single Path (single-video
    # dispatch below) or a (project, [video_paths]) pair (multi-video
    # dispatch). Validation lives in the two helpers so the dispatch
    # block below can focus on construction.
    multi: Optional[tuple] = None  # (DLCProject, list[Path]) when multi-video
    single_path: Optional[Path] = None

    if isinstance(path, (list, tuple)):
        if len(path) == 0:
            raise ValueError("dustrack.open: empty path sequence")
        path_list = [Path(p) for p in path]
        for p in path_list:
            if not p.exists():
                raise FileNotFoundError(f"dustrack.open: path does not exist: {p}")
        if len(path_list) == 1:
            # Single-element list dispatches identically to a scalar
            # path -- preserves the pre-1.2.0a3 list-form parity.
            single_path = path_list[0]
        else:
            multi = _resolve_multi_video_from_list(path_list)
    else:
        single_path = Path(path)
        if not single_path.exists():
            raise FileNotFoundError(
                f"dustrack.open: path does not exist: {single_path}"
            )

    if single_path is not None and _is_dlc_config_yaml(single_path):
        # 1.2.0a3 follow-up (2026-05-22): picking a ``config.yaml``
        # now dispatches to multi-video, queueing every video in
        # ``config['video_sets']`` in the YAML's stored order. Pre-
        # fix this opened video 0 only -- that was a holdover from
        # the pre-multi-video era. ``DLCProject.__init__`` runs
        # :func:`rebase_to_config` on every video_sets key BEFORE we
        # enumerate via ``project.video_list``, so a renamed
        # project folder self-heals: stale path prefixes get
        # rewritten to match the current ``config_path`` parent and
        # persisted back to disk. No backwards-compat shim.
        if not HAS_DLC:
            raise ImportError(
                f"dustrack.open: {single_path} is a DLC config.yaml but "
                "deeplabcut is not installed. Install deeplabcut to resume "
                "the project, or point at a video outside the project to use "
                "DUSTrack standalone."
            )
        project = DLCProject(str(single_path))
        video_paths = [Path(v) for v in project.video_list]
        if len(video_paths) == 0:
            raise ValueError(
                f"dustrack.open: DLC project at {single_path.parent} has no "
                "videos in its config['video_sets']. Add a video first or "
                "point at a specific file."
            )
        multi = (project, video_paths)
        single_path = None  # multi-video supersedes single dispatch
    elif single_path is not None and single_path.is_dir():
        # Directory single-form: a DLC project root means "queue every
        # video in the project" (1.2.0a3 multi-video).
        if not _is_dlc_project_root(single_path):
            raise ValueError(
                f"dustrack.open: {single_path!s} is a directory but doesn't "
                "look like a DLC project (no config.yaml + videos/ + "
                "labeled-data/). Pass a video file or a DLC project folder."
            )
        if not HAS_DLC:
            raise ImportError(
                f"dustrack.open: detected a DLC project at {single_path}, "
                "but deeplabcut is not installed. Install deeplabcut to resume "
                "the project, or point at a video outside the project to use "
                "DUSTrack standalone."
            )
        project = DLCProject(str(single_path / "config.yaml"))
        video_paths = [Path(v) for v in project.video_list]
        if len(video_paths) == 0:
            raise ValueError(
                f"dustrack.open: DLC project at {single_path} has no videos "
                "in its config['video_sets']. Add a video first or point at a "
                "specific file."
            )
        multi = (project, video_paths)
        single_path = None  # multi-video supersedes single dispatch

    if multi is not None:
        # Multi-video dispatch: build the active session on video 0,
        # construct pending bundles for the rest, hand off to
        # ``DUSTrack._init_bundles`` to wire the swap-state machinery.
        project, video_paths = multi
        active_path = video_paths[0]
        active_index_in_project = _find_video_index(project, active_path)
        if active_index_in_project is None:
            # Defensive: _resolve_multi_video_from_list already verified
            # every path resolves to this project, but a path could in
            # principle differ from project.video_list by drive-letter
            # or case in a way the stem-based lookup misses. In that
            # case fall back to index 0.
            active_index_in_project = 0
        tracker = project.annotate(
            video_index=active_index_in_project,
            new_annotation_suffix=layer_name,
            **dustrack_kwargs,
        )
        _attach_bundles_or_fallback(tracker, project, video_paths)
        _show_first_paint_notice(tracker)
        return tracker

    # Single-video dispatch (Phase 1 bare video, or Phase 2 explicit
    # single inside a project / config.yaml).
    p = single_path
    config_path = _find_dlc_config(p)

    if config_path is None:
        # Phase 1: no DLC project context. ``layer_name`` defaults to
        # ``iteration-0`` so a bare-video session seeds the canonical
        # DLC iteration-N naming -- the next DLC training round lands
        # as ``iteration-1`` rather than colliding with whatever ad-hoc
        # name the user picked.
        if not p.is_file():
            raise ValueError(
                f"dustrack.open: {p!s} is a directory but doesn't look like "
                "a DLC project (no config.yaml + videos/ + labeled-data/). "
                "Pass a video file or a DLC project folder."
            )
        if layer_name is None:
            layer_name = "iteration-0"
        tracker = DUSTrack(str(p), layer_name, **dustrack_kwargs)
        _attach_bundles_or_fallback(tracker, None, [p])
    else:
        # Phase 2 single: project found, single video.
        if not HAS_DLC:
            raise ImportError(
                f"dustrack.open: detected a DLC project at {config_path.parent}, "
                "but deeplabcut is not installed. Install deeplabcut to resume "
                "the project, or point at a video outside the project to use "
                "DUSTrack standalone."
            )
        project = DLCProject(str(config_path))

        # If the caller pointed at a specific video inside the project,
        # respect that; otherwise default to the first video (matches
        # DLCProject.annotate's own default).
        video_index = 0
        if p.is_file():
            match = _find_video_index(project, p)
            if match is not None:
                video_index = match

        tracker = project.annotate(
            video_index=video_index,
            new_annotation_suffix=layer_name,
            **dustrack_kwargs,
        )
        _attach_bundles_or_fallback(
            tracker,
            project,
            [Path(project.video_list[video_index])],
        )

    return tracker


def _attach_bundles_or_fallback(tracker, project, video_paths) -> None:
    """Call :meth:`DUSTrack._init_bundles` when available, otherwise
    fall back to setting the bare ``_video_queue`` attribute for
    test-fixture compatibility (the existing pre-1.2.0a3 contract).

    The fallback exists because :func:`dustrack.open` is invoked from
    test fixtures that monkeypatch :class:`DUSTrack` with a plain
    stub class -- the stub has no ``_init_bundles`` method but the
    tests assert against ``tracker._video_queue``. Production code
    always hits the real method.
    """
    init = getattr(tracker, "_init_bundles", None)
    if init is not None:
        init(project=project, video_paths=video_paths)
        return
    tracker._video_queue = [Path(v) for v in video_paths[1:]]
