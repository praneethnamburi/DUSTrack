"""Per-video bundle state for the 1.2.0a3 multi-video swap.

A :class:`_BundleState` represents one video's worth of backend +
frontend state inside a multi-video DUSTrack session. The shell
(``DUSTrack``) holds a list of bundles (``self._bundles``) and rebinds
itself onto the active bundle on every swap. Bundles are populated
either synchronously inside :func:`dustrack.open` (the active bundle)
or off-thread by the background hydration worker (every other
queue entry).

The split between *heavy* fields (``reader``, ``annotations``, set
during hydration) and *snapshot* fields (always present, mutated on
every swap-out / swap-in) is the load-bearing distinction: a swap is
``shell snapshot -> bundle snapshot`` followed by
``bundle snapshot -> shell snapshot`` on the arriving side; nothing on
disk is read, nothing in memory is freed.

See ``specs/dustrack.md`` (Roadmap *Next 1.2.0* item 3) for the
end-to-end contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


def _default_ax_lims() -> dict:
    return {
        "state": False,
        "x": [None, None],
        "y_trace_x": [None, None],
        "y_trace_y": [None, None],
    }


# Hydration state machine. ``pending`` -> ``hydrating`` -> ``ready``
# is the happy path; ``failed`` is a terminal absorbing state with
# ``hydration_error`` populated. Bundles in ``pending``, ``hydrating``,
# or ``failed`` have ``reader is None`` and ``annotations is None``.
HYDRATION_PENDING = "pending"
HYDRATION_HYDRATING = "hydrating"
HYDRATION_READY = "ready"
HYDRATION_FAILED = "failed"

_HYDRATION_STATES = (
    HYDRATION_PENDING,
    HYDRATION_HYDRATING,
    HYDRATION_READY,
    HYDRATION_FAILED,
)


@dataclass
class _BundleState:
    """Per-video backend + frontend snapshot.

    Heavy fields (``reader``, ``annotations``) are populated during
    hydration; they stay ``None`` for ``pending`` / ``hydrating`` /
    ``failed`` bundles. Lightweight snapshot fields are always present
    and survive across swap-out / swap-in cycles, even before the
    first hydration -- their defaults define the bundle's "first
    visit" view (frame 0, no frozen axes, fit-to-frame image pane).

    The shell never reaches into a non-ready bundle's heavy fields;
    :meth:`DUSTrack.swap_to` blocks on hydration via a loading overlay
    before rebinding.
    """

    # Identity (always populated).
    fname: Path
    video_index: int  # 0-based position inside ``DUSTrack._bundles``

    # Project context (1.2.0a3 seed-modal cut). ``None`` = Phase 1
    # (bare video, no DLC project); a ``DLCProject`` = Phase 2. Stored
    # per-bundle so a tracker can hold a mix of Phase 1 + Phase 2
    # bundles (seed-modal flow: seed is Phase 1, picked may be either;
    # future ``add_video`` callers can mix freely). Rebound onto the
    # shell's ``_dlcproject`` in :meth:`DUSTrack._attach_bundle` on
    # every swap, so Workflow-button gates and project-aware code
    # paths see the active bundle's project.
    project: Any = None  # dustrack.dlcinterface.DLCProject | None

    # Heavy state, populated during hydration.
    reader: Any = None  # datanavigator.VideoReader
    annotations: Any = None  # dustrack.annotations.VideoAnnotations

    # Lightweight per-bundle UI snapshot.
    current_idx: int = 0
    selections: dict = field(default_factory=dict)
    ax_lims: dict = field(default_factory=_default_ax_lims)
    image_view_state: Any = None  # opaque blob, set/get via shell dispatch
    trace_view_state: Any = None  # {trace_x_xlim, trace_x_ylim, trace_y_ylim}
    enhance_state: Any = None  # {clahe_clip, gamma, brightness}
    frames_of_interest: list = field(default_factory=list)

    # Lifecycle.
    hydration_state: str = HYDRATION_PENDING
    hydration_error: Optional[str] = None

    def __post_init__(self) -> None:
        self.fname = Path(self.fname)
        if self.hydration_state not in _HYDRATION_STATES:
            raise ValueError(
                f"unknown hydration_state {self.hydration_state!r}; "
                f"expected one of {_HYDRATION_STATES!r}"
            )

    @property
    def is_ready(self) -> bool:
        """True iff this bundle's heavy state is populated and parked
        ready for a swap-in."""
        return self.hydration_state == HYDRATION_READY

    @property
    def is_terminal(self) -> bool:
        """True iff the bundle has reached an absorbing state
        (``ready`` or ``failed``) -- the hydration worker won't touch
        it again."""
        return self.hydration_state in (HYDRATION_READY, HYDRATION_FAILED)



# ---------------------------------------------------------------------
# Background hydration worker
# ---------------------------------------------------------------------
#
# Moved from dlcinterface.py in dustrack 1.2.0rc1 -- the worker is
# tightly coupled to _BundleState's lifecycle so it lives in the same
# module. dlcinterface.py imports both _HDF5_LOCK and _BgHydrationWorker
# from here.

import queue
import sys
import threading
import traceback


# PyTables (the HDF5 backend pandas.read_hdf uses for DLC ``.h5``
# trace files) is NOT thread-safe; concurrent reads from background
# threads hit "Table object has no attribute 'colpathnames'" and
# similar mysteries. The 1.2.0a3 bg hydration worker reads dozens of
# h5 files per bundle off-thread; without serialisation it races with
# itself across bundles and with main-thread reads (e.g. Train DLC's
# post-success ``_refresh_dlc_layers``). One RLock covering every
# HDF5-touching code path is the standard fix for PyTables in
# multi-threaded apps.
_HDF5_LOCK = threading.RLock()


class _BgHydrationWorker:
    """Daemon-thread worker that hydrates pending bundles in queue
    order, with a paired Qt-thread poller that finalises artists.

    Two halves, three threads (counting the Qt main thread):

    1. **Daemon worker thread** runs the off-thread data half:
       ``dustrack._hydrate_bundle_data_only(bundle, project)`` opens
       the VideoReader, reads annotation sidecars + DLC h5 traces
       (serialised on ``_HDF5_LOCK`` -- PyTables isn't thread-safe),
       constructs VideoAnnotation objects with empty axis lists.
       On success the ``(bundle, payload)`` pair is pushed onto
       ``self._finalisation_queue``; on failure ``bundle`` is marked
       ``FAILED`` and pushed alone.
    2. **Qt-thread poller** (a QTimer installed on the main window)
       drains the finalisation queue every 50 ms, running
       ``dustrack._finalise_bundle_artists`` for each ready payload.
       This is where artists wire into the shell's axes -- it MUST
       run on the Qt thread because ``_image_pane.add_marker_group``
       modifies the QGraphicsScene.

    The QTimer.singleShot cross-thread fallback we tried first
    silently dropped events on Windows-Qt6 -- timers created from a
    non-Qt thread with no event loop never fire. The queue + poller
    pattern matches :meth:`_install_dlc_load_gate_refresh` and keeps
    every Qt touch on the main thread.

    Daemon thread: interpreter shutdown reaps it. ``stop()`` is a
    cooperative flag for forward-compat (bundle eviction).
    """

    def __init__(self, dustrack, project, bundles: list) -> None:
        self.dustrack = dustrack
        self.project = project
        self.bundles = list(bundles)
        self._thread: Optional[threading.Thread] = None
        self._stop = False
        # Worker thread pushes (bundle, payload_or_None) tuples; the
        # Qt poller pops them. ``payload_or_None is None`` signals
        # FAILED (worker thread already set the bundle's state +
        # error fields). ``queue.Queue`` is thread-safe out of box.
        self._finalisation_queue: "queue.Queue" = queue.Queue()
        self._poll_timer = None  # set in start()

    def start(self) -> None:
        """Spawn the daemon worker + install the Qt-thread poller.

        Idempotent: a second start call is ignored if a worker
        thread is already running.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="dustrack-hydration",
        )
        self._thread.start()
        self._install_qt_poller()

    def _install_qt_poller(self) -> None:
        """Install a 50 ms QTimer on the dustrack's QMainWindow that
        drains :attr:`_finalisation_queue`.

        Falls back silently when no Qt window is available (mpl-
        fallback path; tests). In that mode, callers can drain the
        queue manually via :meth:`drain_finalisation_queue`.
        """
        try:
            from qtpy.QtCore import QTimer
        except Exception:  # noqa: BLE001
            return
        find_qt_window = getattr(self.dustrack, "_find_qt_window", None)
        if find_qt_window is None:
            return
        qt_window = find_qt_window()
        if qt_window is None:
            return
        timer = QTimer(qt_window)
        timer.setInterval(50)
        timer.timeout.connect(self.drain_finalisation_queue)
        timer.start()
        self._poll_timer = timer  # keep a reference so Qt doesn't GC it

    def drain_finalisation_queue(self) -> None:
        """Pop every ready (bundle, payload) tuple from the queue and
        run :meth:`_finalise_bundle_artists` on the Qt thread.

        Called from the Qt poller's timeout signal AND from tests via
        the bypass path (when no Qt window is available). Each call
        drains every queued item (not just one per tick) so a burst
        of completions doesn't stretch over multiple ticks.

        Auto-stops the poll timer once every bundle in this worker's
        queue has reached a terminal state -- otherwise the timer
        fires every 50 ms forever, competing with paint events for
        the Qt event loop and contributing to ``draw_idle`` no-fire
        issues elsewhere in the app.
        """
        while True:
            try:
                bundle, payload = self._finalisation_queue.get_nowait()
            except queue.Empty:
                break
            if payload is None:
                # Failure path -- worker thread already wrote the
                # bundle's state + error; just nudge the nav button
                # row so the user sees the (n ready) count update.
                try:
                    self.dustrack._refresh_nav_buttons()
                except Exception:  # noqa: BLE001
                    pass
                continue
            try:
                self.dustrack._finalise_bundle_artists(
                    bundle, payload, self.project,
                )
            except BaseException as exc:  # noqa: BLE001
                bundle.hydration_state = HYDRATION_FAILED
                bundle.hydration_error = f"{type(exc).__name__}: {exc}"
                tb = traceback.format_exc()
                sys.__stderr__.write(
                    f"[dustrack] bundle {bundle.video_index} "
                    f"({bundle.fname}) hydration (artists) failed:\n{tb}\n"
                )
                try:
                    self.dustrack._refresh_nav_buttons()
                except Exception:  # noqa: BLE001
                    pass
        # Queue is drained for this tick. If every bundle has reached
        # a terminal state, stop the poller so it doesn't keep
        # firing forever on an empty queue.
        if self._poll_timer is not None and all(
            b.is_terminal for b in self.bundles
        ):
            try:
                self._poll_timer.stop()
            except Exception:  # noqa: BLE001
                pass
            self._poll_timer = None

    def stop(self) -> None:
        """Signal the worker to exit after the current bundle (if
        any). Slice 2's ``_await_hydration`` doesn't need this --
        kept for forward compat with bundle eviction (post-1.2.0a3)."""
        self._stop = True

    def _run(self) -> None:
        for bundle in self.bundles:
            if self._stop:
                return
            if bundle.hydration_state != HYDRATION_PENDING:
                # Skip bundles that another path (sync hydration in
                # tests, or a swap-driven hydrate) already finished.
                continue
            self._hydrate_one(bundle)

    def _hydrate_one(self, bundle) -> None:
        try:
            payload = self.dustrack._hydrate_bundle_data_only(
                bundle, self.project,
            )
        except BaseException as exc:  # noqa: BLE001
            bundle.hydration_state = HYDRATION_FAILED
            bundle.hydration_error = f"{type(exc).__name__}: {exc}"
            tb = traceback.format_exc()
            sys.__stderr__.write(
                f"[dustrack] bundle {bundle.video_index} "
                f"({bundle.fname}) hydration (data) failed:\n{tb}\n"
            )
            # Push a (bundle, None) sentinel so the Qt poller knows
            # to refresh nav buttons (and so tests that drain the
            # queue see the failure event).
            self._finalisation_queue.put((bundle, None))
            return
        # Data half done; queue the payload for Qt-thread finalisation.
        self._finalisation_queue.put((bundle, payload))


# ---------------------------------------------------------------------
# Per-bundle hydration helpers
# ---------------------------------------------------------------------
#
# Extracted from gui.DUSTrack in the 1.2.0rc1 follow-up. Two halves
# per bundle:
#
# 1. **Data half** (off-thread safe) -- :func:`hydrate_bundle_data_only`
#    for Phase 2 (DLC project) and :func:`hydrate_phase1_bundle_data`
#    for Phase 1 (bare video). Opens the VideoReader, reads annotation
#    sidecars + DLC h5 traces (under :data:`_HDF5_LOCK`), constructs
#    VideoAnnotation objects with EMPTY axis lists, returns a payload
#    dict.
#
# 2. **Qt-thread half** -- :func:`finalise_bundle_artists`. Wires
#    the per-annotation artists into the shell's axes (per-layer
#    marker group on the image pane, shared trace axes), then hides
#    every artist so the bundle is parked invisible until the user
#    swaps to it.
#
# :func:`hydrate_bundle_sync` runs both halves on the calling thread
# (used by single-video entry paths + tests). The multi-video happy
# path goes through :class:`_BgHydrationWorker`.

import os
from typing import Any  # noqa: F401 -- re-exported in some signatures


def _ann_path_alongside_video(bundle: "_BundleState", layer_name: str) -> str:
    """Phase 1 per-layer fname: ``<stem>_annotations_<layer>.json``
    in the same folder as the video. Mirrors
    :func:`._layer_names.get_fname_annotations` but specialised for
    a bundle object.
    """
    stem = bundle.fname.stem
    suffix_part = f"_{layer_name}" if layer_name else ""
    return str(bundle.fname.parent / f"{stem}_annotations{suffix_part}.json")


def hydrate_bundle_data_only(dustrack, bundle: "_BundleState", project) -> dict:
    """Off-thread half of Phase 2 bundle hydration.

    Touches filesystem (VideoReader open, JSON / h5 reads), numpy /
    pandas (vectorised DLC trace decoding), and VideoAnnotation
    construction with EMPTY axis lists so the artist setup downstream
    is a no-op. Does NOT touch Qt or matplotlib -- safe to call from
    a daemon thread.

    Sets ``bundle.hydration_state`` to ``HYDRATING`` on entry; the
    caller flips it to ``READY`` / ``FAILED`` based on the rest of
    the pipeline. Returns a payload dict for :func:`finalise_bundle_artists`.
    """
    from .annotations import VideoAnnotation, VideoAnnotations
    from ._dlc_paths import _find_video_index
    from ._file_management import VideoFileManager
    from datanavigator import VideoReader

    bundle.hydration_state = HYDRATION_HYDRATING

    video_index = _find_video_index(project, bundle.fname)
    if video_index is None:
        raise ValueError(
            f"bundle video {bundle.fname} is not in DLC project "
            f"{project.config_path}"
        )
    in_project_path = Path(project.video_list[video_index])

    # Compute the next-iteration suffix the way DLCProject.annotate
    # does, so a Train DLC run cuts the same fresh layer regardless
    # of which bundle was active when the user clicked Train.
    if project.latest_iteration_is_trained():
        new_iteration_num = project.latest_iteration + 1
    else:
        new_iteration_num = project.latest_iteration
    new_annotation_suffix = f"iteration-{new_iteration_num}"

    fm = VideoFileManager(project, video_index)
    ann_name_to_fname = fm.get_all_annotation_layers(new_annotation_suffix)
    ann_name_to_fname["buffer"] = fm.get_new_json("buffer")

    # Each bundle owns its own VideoReader (one open file per video).
    with open(str(in_project_path), "rb") as f:
        reader = VideoReader(f)

    # VideoAnnotation.__init__ -> load() reads DLC .h5 via PyTables,
    # which is NOT thread-safe. Serialise behind _HDF5_LOCK so
    # concurrent bundle hydrations + main-thread reads can't race.
    container = VideoAnnotations(parent=dustrack)
    with _HDF5_LOCK:
        for name, fname in ann_name_to_fname.items():
            # EMPTY ax lists -- finalise_bundle_artists attaches real
            # axes + builds artists on the Qt thread.
            ann = container.add(
                name=name,
                fname=fname,
                vname=str(in_project_path),
                video=reader,
                ax_list_scatter=[],
                ax_list_trace_x=[],
                ax_list_trace_y=[],
                palette_name="Set2",
                n_labels=1,
            )
            ann.__class__ = VideoAnnotation

    _union_labels_across_layers(container)

    return {
        "reader": reader,
        "container": container,
        "in_project_path": in_project_path,
    }


def hydrate_phase1_bundle_data(
    dustrack, bundle: "_BundleState", *, layer_name: str = "iteration-0",
) -> dict:
    """Off-thread half of Phase 1 (bare-video, no DLC project) hydration.

    Mirrors :func:`hydrate_bundle_data_only`'s contract, except the
    layer set is the canonical Phase 1 pair (``layer_name`` +
    ``buffer``) with paths derived from the bundle's video stem --
    no VideoFileManager / project lookup. Used by ``add_video``
    when appending a bare-video bundle (notably the seed-modal flow).
    """
    from .annotations import VideoAnnotation, VideoAnnotations
    from datanavigator import VideoReader

    bundle.hydration_state = HYDRATION_HYDRATING

    vname = str(bundle.fname)
    ann_name_to_fname = {
        layer_name: _ann_path_alongside_video(bundle, layer_name),
        "buffer": _ann_path_alongside_video(bundle, "buffer"),
    }

    with open(vname, "rb") as f:
        reader = VideoReader(f)

    # Same _HDF5_LOCK pattern as the Phase 2 path; Phase 1 .json reads
    # aren't HDF5-backed but the lock is cheap when uncontended.
    container = VideoAnnotations(parent=dustrack)
    with _HDF5_LOCK:
        for name, fname in ann_name_to_fname.items():
            ann = container.add(
                name=name,
                fname=fname,
                vname=vname,
                video=reader,
                ax_list_scatter=[],
                ax_list_trace_x=[],
                ax_list_trace_y=[],
                palette_name="Set2",
                n_labels=1,
            )
            ann.__class__ = VideoAnnotation

    _union_labels_across_layers(container)

    return {
        "reader": reader,
        "container": container,
        "in_project_path": bundle.fname,
    }


def _union_labels_across_layers(container) -> None:
    """Union of declared labels across every layer in the container so
    each layer presents the same label rotation. Mirrors
    ``_DUSTrackBase.add_annotation_layers``.
    """
    all_labels = sorted(
        {label for ann in container._list for label in ann.labels}
    )
    if not all_labels:
        all_labels = ["0"]
    for ann in container._list:
        for label in all_labels:
            if label not in ann.labels:
                ann.add_label(label)
        ann.sort_labels()
        ann.re_setup_display()


def finalise_bundle_artists(
    dustrack, bundle: "_BundleState", payload: dict, project,
) -> None:
    """Qt-thread half of bundle hydration.

    Wires each annotation's artists into the shell's axes (per-layer
    marker group on the image pane, shared trace axes), applies the
    plot-type convention (dense layers + buffer render as lines),
    then hides every artist so the bundle is parked invisible until
    the user swaps to it.

    MUST run on the Qt thread: ``_image_pane.add_marker_group()``
    modifies the QGraphicsScene, which is not thread-safe.

    On success: bundle ``hydration_state`` flips to ``READY``,
    ``selections`` seeded to the canonical fresh-load state, and
    ``_refresh_nav_buttons`` is called so the position indicator +
    arrow enable states update.
    """
    from ._layer_names import _is_dense_layer_name

    container = payload["container"]
    reader = payload["reader"]
    # Wire artists. Tier 2 builds a per-layer marker group on the
    # image pane; Tier 1 reuses the shell's image axis directly.
    for ann in container._list:
        if dustrack._fast_render:
            ax_list_scatter = [dustrack._image_pane.add_marker_group()]
        else:
            ax_list_scatter = [dustrack._ax_image]
        ann.setup_display(
            ax_list_scatter=ax_list_scatter,
            ax_list_trace_x=[dustrack._ax_trace_x],
            ax_list_trace_y=[dustrack._ax_trace_y],
        )
        # Apply per-annotation plot-type convention that the active
        # bundle gets via __init__'s buffer.plot_type = "line" line +
        # _normalize_dlc_layer_display. Without this every bundle-k+1
        # dense layer defaults to "dot" and the trace pane shows dots
        # instead of lines after swap.
        if ann.name == "buffer" or _is_dense_layer_name(ann.name):
            try:
                ann.set_plot_type("line", draw=False)
            except Exception:  # noqa: BLE001
                pass
        # Invalidate the trace cache so the first update_display_trace
        # against the freshly-bound handles repopulates ydata.
        ann.invalidate_caches()
        ann.hide(draw=False)

    bundle.reader = reader
    bundle.annotations = container
    # Derive the canonical fresh-load selections for this bundle.
    derived = derive_initial_bundle_selections(
        dustrack, container, project=project,
    )
    # If the user toggled a broadcast statevar (annotation_label /
    # label_range / number_keys) while this bundle was pending, the
    # broadcast wrote into bundle.selections BEFORE hydration
    # completed. Preserve those; only per-video statevars
    # (annotation_layer / annotation_overlay) come from the canonical
    # defaults.
    existing = bundle.selections or {}
    for sv_name in dustrack._BROADCAST_STATEVARS:
        if sv_name in existing:
            derived[sv_name] = existing[sv_name]
    bundle.selections = derived
    bundle.hydration_state = HYDRATION_READY
    bundle.hydration_error = None
    try:
        dustrack._refresh_nav_buttons()
    except Exception:  # noqa: BLE001
        pass


def derive_initial_bundle_selections(dustrack, container, project=None) -> dict:
    """First-time statevar selections for a freshly-hydrated bundle.

    Picks the canonical fresh-load state: latest manual layer as
    active, latest ``dlc_*`` layer as overlay (or None), first label
    / its label_range as the active bodypart, current shell's
    ``number_keys`` mode (so the cross-bundle UI mode carries from
    the start).
    """
    from ._layer_names import _is_dense_layer_name

    names = container.names
    # Latest manual layer = active. Manual layers are everything
    # except buffer / dense (dlc_* / dlccorr* / lkmovavg).
    manuals = [
        n for n in names
        if n != "buffer" and not _is_dense_layer_name(n)
    ]
    # The new iteration-{N+1} layer (just created by
    # get_all_annotation_layers) lands at the tail of the manuals
    # block -- match DLCProject.annotate's convention by picking the
    # LAST manual as active.
    active_layer = manuals[-1] if manuals else (names[0] if names else None)
    dlc_layers = [n for n in names if n.startswith("dlc_")]
    overlay = dlc_layers[-1] if dlc_layers else None
    # Active label / label_range -- derive from the active layer.
    ann = container[active_layer] if active_layer else None
    if ann and ann.labels:
        first_label = ann.labels[0]
        try:
            label_range_idx = int(first_label) // 10
            label_range_value = f"{label_range_idx*10}-{label_range_idx*10+9}"
        except (TypeError, ValueError):
            label_range_value = None
    else:
        first_label = None
        label_range_value = None
    # number_keys carries the shell's current mode (broadcast default).
    nk = None
    if "number_keys" in dustrack.statevariables.names:
        nk = dustrack.statevariables["number_keys"].current_state
    return {
        "annotation_layer": active_layer,
        "annotation_overlay": overlay,
        "annotation_label": first_label,
        "label_range": label_range_value,
        "number_keys": nk,
    }


def hydrate_bundle_sync(dustrack, bundle: "_BundleState", project=None) -> None:
    """Populate ``bundle``'s heavy state synchronously, dispatching
    on ``bundle.project`` (Phase 1 vs Phase 2).

    Used by single-video / single-bundle entry paths and by the
    worker's failure-path tests; the multi-video happy path goes
    through :class:`_BgHydrationWorker`.
    """
    # Per-bundle project takes precedence over the legacy arg so
    # cross-Phase batches stay coherent. ``project`` is kept for
    # back-compat with the pre-1.2.0a3 call-site.
    eff_project = bundle.project if bundle.project is not None else project
    try:
        if eff_project is None:
            payload = hydrate_phase1_bundle_data(dustrack, bundle)
        else:
            payload = hydrate_bundle_data_only(dustrack, bundle, eff_project)
    except Exception as exc:  # noqa: BLE001
        bundle.hydration_state = HYDRATION_FAILED
        bundle.hydration_error = f"{type(exc).__name__}: {exc}"
        sys.__stderr__.write(
            f"[dustrack] bundle {bundle.video_index} "
            f"({bundle.fname}) hydration failed:\n{traceback.format_exc()}\n"
        )
        return
    try:
        finalise_bundle_artists(dustrack, bundle, payload, eff_project)
    except Exception as exc:  # noqa: BLE001
        bundle.hydration_state = HYDRATION_FAILED
        bundle.hydration_error = f"{type(exc).__name__}: {exc}"
        sys.__stderr__.write(
            f"[dustrack] bundle {bundle.video_index} "
            f"({bundle.fname}) artist setup failed:\n{traceback.format_exc()}\n"
        )


def park_bundle_artists(bundle: "_BundleState") -> None:
    """Hide every annotation artist owned by ``bundle``."""
    if bundle.annotations is None:
        return
    for ann in bundle.annotations._list:
        try:
            ann.hide(draw=False)
        except Exception:  # noqa: BLE001
            # Defensive: never strand the user mid-swap if one
            # artist's hide() raises.
            traceback.print_exc()


def show_bundle_artists(bundle: "_BundleState") -> None:
    """Show every annotation artist owned by ``bundle``."""
    if bundle.annotations is None:
        return
    for ann in bundle.annotations._list:
        try:
            ann.show(draw=False)
        except Exception:  # noqa: BLE001
            traceback.print_exc()


def notify_bundle_failure(bundle: "_BundleState") -> None:
    """Surface a hydration failure to the user via stderr.

    Slice 2 will route through a proper error overlay; for now we
    just log so the swap-failure case is observable in the terminal.
    """
    sys.__stderr__.write(
        f"[dustrack] cannot swap to bundle {bundle.video_index} "
        f"({bundle.fname}): {bundle.hydration_error}\n"
    )
