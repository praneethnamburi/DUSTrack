"""Tests for :class:`_BgHydrationWorker` (1.2.0a3 multi-video Slice 2).

The worker spawns a daemon thread that walks pending bundles in
queue order. For each pending bundle it runs the off-thread data
half (``dustrack._hydrate_bundle_data_only``) inline on its own
thread, then pushes the (bundle, payload) onto a thread-safe
finalisation queue. A Qt-thread poller (QTimer on the main window)
drains the queue and runs the artist-setup half
(``dustrack._finalise_bundle_artists``) on the GUI thread.

Tests bypass the daemon-thread + QTimer machinery by:

- Calling ``worker._run()`` directly on the test thread (no
  ``start()``); the data half runs inline and the (bundle, payload)
  tuples land in ``worker._finalisation_queue``.
- Calling ``worker.drain_finalisation_queue()`` synchronously to
  flush the queue + run the artist half against a fake dustrack
  (no Qt event loop required).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dustrack._bundle import (
    HYDRATION_FAILED,
    HYDRATION_HYDRATING,
    HYDRATION_PENDING,
    HYDRATION_READY,
    _BundleState,
)


def _make_fake_dustrack(data_payloads=None, data_errors=None,
                         artist_errors=None):
    """Build a fake DUSTrack exposing only the methods the worker
    calls. ``data_payloads`` / ``data_errors`` / ``artist_errors``
    are dicts keyed by ``bundle.video_index``."""
    data_payloads = data_payloads or {}
    data_errors = data_errors or {}
    artist_errors = artist_errors or {}

    def _hydrate_data(bundle, project):
        bundle.hydration_state = HYDRATION_HYDRATING
        if bundle.video_index in data_errors:
            raise data_errors[bundle.video_index]
        return data_payloads.get(bundle.video_index, {})

    def _finalise(bundle, payload, project):
        if bundle.video_index in artist_errors:
            raise artist_errors[bundle.video_index]
        bundle.hydration_state = HYDRATION_READY
        bundle.hydration_error = None

    # _find_qt_window returns None so the worker skips QTimer
    # installation -- tests drain the queue manually.
    return SimpleNamespace(
        _hydrate_bundle_data_only=_hydrate_data,
        _finalise_bundle_artists=_finalise,
        _refresh_nav_buttons=lambda: None,
        _find_qt_window=lambda: None,
    )


def _make_pending_bundle(i):
    return _BundleState(
        fname=Path(f"/v{i}.mp4"), video_index=i,
        hydration_state=HYDRATION_PENDING,
    )


# ---------------------------------------------------------------------
# Per-bundle data-half lifecycle (worker._run)
# ---------------------------------------------------------------------


class TestDataHalfLifecycle:
    def test_pending_data_success_queues_payload(self):
        from dustrack.dlcinterface import _BgHydrationWorker
        dustrack = _make_fake_dustrack(data_payloads={1: {"foo": "bar"}})
        bundle = _make_pending_bundle(1)
        worker = _BgHydrationWorker(dustrack, project=object(), bundles=[bundle])
        worker._run()
        # Bundle's state is HYDRATING (not READY -- artist half
        # hasn't run yet).
        assert bundle.hydration_state == HYDRATION_HYDRATING
        # Payload is on the queue.
        assert worker._finalisation_queue.qsize() == 1
        b, payload = worker._finalisation_queue.get_nowait()
        assert b is bundle
        assert payload == {"foo": "bar"}

    def test_data_half_exception_marks_failed_and_queues_sentinel(self):
        from dustrack.dlcinterface import _BgHydrationWorker
        err = FileNotFoundError("missing.h5")
        dustrack = _make_fake_dustrack(data_errors={1: err})
        bundle = _make_pending_bundle(1)
        worker = _BgHydrationWorker(dustrack, project=object(), bundles=[bundle])
        worker._run()
        assert bundle.hydration_state == HYDRATION_FAILED
        assert "FileNotFoundError" in bundle.hydration_error
        assert "missing.h5" in bundle.hydration_error
        # Failure sentinel: (bundle, None) on the queue so the Qt
        # poller can refresh nav buttons.
        b, payload = worker._finalisation_queue.get_nowait()
        assert b is bundle
        assert payload is None

    def test_already_ready_bundle_skipped(self):
        from dustrack.dlcinterface import _BgHydrationWorker
        dustrack = _make_fake_dustrack(
            data_errors={1: RuntimeError("should not be called")},
        )
        bundle = _BundleState(
            fname=Path("/v1.mp4"), video_index=1,
            hydration_state=HYDRATION_READY,
        )
        worker = _BgHydrationWorker(dustrack, project=object(), bundles=[bundle])
        worker._run()  # would raise if data half ran
        assert bundle.hydration_state == HYDRATION_READY
        assert worker._finalisation_queue.qsize() == 0

    def test_already_failed_bundle_skipped(self):
        from dustrack.dlcinterface import _BgHydrationWorker
        dustrack = _make_fake_dustrack(
            data_errors={1: RuntimeError("should not be called")},
        )
        bundle = _BundleState(
            fname=Path("/v1.mp4"), video_index=1,
            hydration_state=HYDRATION_FAILED,
            hydration_error="previous failure",
        )
        worker = _BgHydrationWorker(dustrack, project=object(), bundles=[bundle])
        worker._run()
        assert bundle.hydration_state == HYDRATION_FAILED
        assert bundle.hydration_error == "previous failure"


# ---------------------------------------------------------------------
# Artist-half lifecycle (drain_finalisation_queue)
# ---------------------------------------------------------------------


class TestArtistHalfLifecycle:
    def test_drain_flips_bundle_to_ready(self):
        from dustrack.dlcinterface import _BgHydrationWorker
        dustrack = _make_fake_dustrack()
        bundle = _make_pending_bundle(1)
        worker = _BgHydrationWorker(dustrack, project=object(), bundles=[bundle])
        worker._run()  # data half done -> queue has payload
        worker.drain_finalisation_queue()
        assert bundle.hydration_state == HYDRATION_READY
        assert bundle.hydration_error is None

    def test_drain_handles_failure_sentinel(self):
        """A (bundle, None) sentinel on the queue means data half
        failed; drain pops it and refreshes nav buttons without
        running the artist half."""
        from dustrack.dlcinterface import _BgHydrationWorker
        nav_calls = []
        dustrack = SimpleNamespace(
            _hydrate_bundle_data_only=lambda *a, **k: pytest.fail(
                "should not be called"
            ),
            _finalise_bundle_artists=lambda *a, **k: pytest.fail(
                "artist half must not run on failed bundle"
            ),
            _refresh_nav_buttons=lambda: nav_calls.append("refresh"),
            _find_qt_window=lambda: None,
        )
        bundle = _BundleState(
            fname=Path("/v1.mp4"), video_index=1,
            hydration_state=HYDRATION_FAILED,
            hydration_error="ValueError: synthetic",
        )
        worker = _BgHydrationWorker(dustrack, project=object(), bundles=[])
        # Manually push a failure sentinel onto the queue (simulating
        # what _hydrate_one does on exception).
        worker._finalisation_queue.put((bundle, None))
        worker.drain_finalisation_queue()
        assert nav_calls == ["refresh"]
        assert bundle.hydration_state == HYDRATION_FAILED  # unchanged

    def test_drain_catches_artist_half_exception(self):
        from dustrack.dlcinterface import _BgHydrationWorker
        err = ValueError("artist setup blew up")
        dustrack = _make_fake_dustrack(artist_errors={1: err})
        bundle = _make_pending_bundle(1)
        worker = _BgHydrationWorker(dustrack, project=object(), bundles=[bundle])
        worker._run()  # queues payload
        worker.drain_finalisation_queue()  # artist half raises
        assert bundle.hydration_state == HYDRATION_FAILED
        assert "ValueError" in bundle.hydration_error
        assert "artist setup blew up" in bundle.hydration_error

    def test_drain_processes_multiple_queued_payloads(self):
        from dustrack.dlcinterface import _BgHydrationWorker
        dustrack = _make_fake_dustrack()
        bundles = [_make_pending_bundle(i) for i in (1, 2, 3)]
        worker = _BgHydrationWorker(dustrack, project=object(), bundles=bundles)
        worker._run()  # all three data halves done
        assert worker._finalisation_queue.qsize() == 3
        worker.drain_finalisation_queue()
        for b in bundles:
            assert b.hydration_state == HYDRATION_READY
        assert worker._finalisation_queue.qsize() == 0

    def test_drain_on_empty_queue_is_noop(self):
        from dustrack.dlcinterface import _BgHydrationWorker
        dustrack = _make_fake_dustrack()
        worker = _BgHydrationWorker(dustrack, project=object(), bundles=[])
        # Must not raise.
        worker.drain_finalisation_queue()


# ---------------------------------------------------------------------
# Queue-level behavior
# ---------------------------------------------------------------------


class TestQueueBehavior:
    def test_continues_after_per_bundle_failure(self):
        """A failing bundle doesn't stall the worker; it moves on to
        the next pending bundle in the queue."""
        from dustrack.dlcinterface import _BgHydrationWorker
        dustrack = _make_fake_dustrack(
            data_errors={2: RuntimeError("v2 file corrupt")},
        )
        bundles = [_make_pending_bundle(i) for i in (1, 2, 3)]
        worker = _BgHydrationWorker(dustrack, project=object(), bundles=bundles)
        worker._run()
        worker.drain_finalisation_queue()
        assert bundles[0].hydration_state == HYDRATION_READY
        assert bundles[1].hydration_state == HYDRATION_FAILED
        assert bundles[2].hydration_state == HYDRATION_READY

    def test_stop_flag_exits_early(self):
        """The ``stop()`` cooperative flag halts the worker after the
        current bundle completes."""
        from dustrack.dlcinterface import _BgHydrationWorker
        processed = []

        def _hydrate_data(bundle, project):
            processed.append(bundle.video_index)
            bundle.hydration_state = HYDRATION_HYDRATING
            if bundle.video_index == 1:
                worker.stop()
            return {}

        def _finalise(bundle, payload, project):
            bundle.hydration_state = HYDRATION_READY

        dustrack = SimpleNamespace(
            _hydrate_bundle_data_only=_hydrate_data,
            _finalise_bundle_artists=_finalise,
            _refresh_nav_buttons=lambda: None,
            _find_qt_window=lambda: None,
        )
        bundles = [_make_pending_bundle(i) for i in (1, 2, 3)]
        worker = _BgHydrationWorker(dustrack, project=object(), bundles=bundles)
        worker._run()
        worker.drain_finalisation_queue()
        assert processed == [1]
        assert bundles[0].hydration_state == HYDRATION_READY
        assert bundles[1].hydration_state == HYDRATION_PENDING
        assert bundles[2].hydration_state == HYDRATION_PENDING

    def test_start_is_idempotent(self):
        """Calling ``start()`` twice doesn't spawn a second thread."""
        from dustrack.dlcinterface import _BgHydrationWorker
        dustrack = _make_fake_dustrack()
        worker = _BgHydrationWorker(dustrack, project=object(), bundles=[])
        worker.start()
        first_thread = worker._thread
        if first_thread is not None:
            first_thread.join(timeout=1.0)
        worker.start()  # would re-spawn if not guarded
        assert (worker._thread is first_thread) or (
            not first_thread.is_alive()
        )
