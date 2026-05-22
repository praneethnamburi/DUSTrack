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
    annotations: Any = None  # dustrack.pointtracking.VideoAnnotations

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

