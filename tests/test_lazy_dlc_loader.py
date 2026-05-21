"""Tests for the lazy ``import deeplabcut`` plumbing.

The DLC import is ~7 s on the dlc3rc14 env and used to run at
``import dustrack`` time. The lazy loader (in ``dustrack/dlcinterface.py``)
splits this into:

- :data:`HAS_DLC` -- cheap ``importlib.util.find_spec`` result.
- :func:`_ensure_dlc_loaded` -- synchronous import, idempotent,
  thread-safe.
- :func:`_ensure_dlc_loaded_async` -- daemon-thread variant, fired
  from :func:`dustrack.open` so the import races the picker / GUI
  construction / user annotation work.
- :func:`register_dlc_load_callback` -- callback fan-out for code
  that needs to react to the load finishing (e.g. the
  workflow-button gate refresh).

These tests don't *unimport* DeepLabCut between runs (which would be
fragile + slow); instead they manipulate the module-level loader
state directly and verify the state-machine behavior. The actual
``deeplabcut`` import is exercised once by the smoke-test below
(skipped if DLC isn't installed) so we know the real import path
still works.
"""
from __future__ import annotations

import threading
import time

import pytest

from dustrack import dlcinterface


@pytest.fixture
def _reset_loader():
    """Save + restore the loader state so test cases that flip
    ``_DLC_LOAD_STATE`` / ``_DLC_LOAD_THREAD`` / module globals don't
    bleed into the rest of the suite.
    """
    saved = {
        "state": dlcinterface._DLC_LOAD_STATE,
        "thread": dlcinterface._DLC_LOAD_THREAD,
        "callbacks": list(dlcinterface._DLC_LOAD_CALLBACKS),
        "deeplabcut": dlcinterface.deeplabcut,
        "VideoWriter": dlcinterface.VideoWriter,
        "ScannerError": dlcinterface.ScannerError,
        "DLC3": dlcinterface.DLC3,
        "HAS_DLC": dlcinterface.HAS_DLC,
    }
    try:
        yield
    finally:
        dlcinterface._DLC_LOAD_STATE = saved["state"]
        dlcinterface._DLC_LOAD_THREAD = saved["thread"]
        dlcinterface._DLC_LOAD_CALLBACKS[:] = saved["callbacks"]
        dlcinterface.deeplabcut = saved["deeplabcut"]
        dlcinterface.VideoWriter = saved["VideoWriter"]
        dlcinterface.ScannerError = saved["ScannerError"]
        dlcinterface.DLC3 = saved["DLC3"]
        dlcinterface.HAS_DLC = saved["HAS_DLC"]


class TestStateMachine:
    def test_done_short_circuits_ensure(self, _reset_loader):
        dlcinterface._DLC_LOAD_STATE = "done"
        assert dlcinterface._ensure_dlc_loaded() is True

    def test_missing_short_circuits_ensure(self, _reset_loader):
        dlcinterface._DLC_LOAD_STATE = "missing"
        assert dlcinterface._ensure_dlc_loaded() is False

    def test_has_dlc_false_returns_false_without_import(self, _reset_loader):
        dlcinterface.HAS_DLC = False
        dlcinterface._DLC_LOAD_STATE = "pending"
        assert dlcinterface._ensure_dlc_loaded() is False
        assert dlcinterface._DLC_LOAD_STATE == "missing"

    def test_dlc_load_state_helper(self, _reset_loader):
        dlcinterface._DLC_LOAD_STATE = "pending"
        dlcinterface._DLC_LOAD_THREAD = None
        assert dlcinterface._dlc_load_state() == "pending"

        dlcinterface._DLC_LOAD_THREAD = threading.Thread(target=lambda: None)
        assert dlcinterface._dlc_load_state() == "loading"

        dlcinterface._DLC_LOAD_STATE = "done"
        assert dlcinterface._dlc_load_state() == "done"

        dlcinterface._DLC_LOAD_STATE = "missing"
        assert dlcinterface._dlc_load_state() == "missing"


class TestAsyncLoader:
    def test_async_returns_quickly(self, _reset_loader):
        """``_ensure_dlc_loaded_async`` must return in << the
        synchronous ``import deeplabcut`` time so the bg load runs
        in parallel with the picker / GUI construction.
        """
        dlcinterface._DLC_LOAD_STATE = "done"  # short-circuit; no real work
        t0 = time.perf_counter()
        result = dlcinterface._ensure_dlc_loaded_async()
        elapsed = time.perf_counter() - t0
        assert result is None  # nothing to do
        # 50 ms is generous; the real cost is sub-millisecond.
        assert elapsed < 0.05

    def test_async_is_no_op_when_done(self, _reset_loader):
        dlcinterface._DLC_LOAD_STATE = "done"
        assert dlcinterface._ensure_dlc_loaded_async() is None

    def test_async_is_no_op_when_missing(self, _reset_loader):
        dlcinterface._DLC_LOAD_STATE = "missing"
        assert dlcinterface._ensure_dlc_loaded_async() is None

    def test_async_returns_none_when_has_dlc_false(self, _reset_loader):
        dlcinterface.HAS_DLC = False
        dlcinterface._DLC_LOAD_STATE = "pending"
        assert dlcinterface._ensure_dlc_loaded_async() is None
        # Side effect: state was settled to ``"missing"`` so subsequent
        # sync callers short-circuit without touching the lock.
        assert dlcinterface._DLC_LOAD_STATE == "missing"

    def test_async_idempotent_repeat_call_returns_running_thread(
        self, _reset_loader
    ):
        # Simulate a long-running loader by hand: spawn a thread that
        # waits on an Event so the test can join it on its own
        # schedule. The real loader thread (which runs
        # ``_ensure_dlc_loaded``) is bypassed by setting state to
        # ``pending`` and putting a placeholder thread in the slot.
        gate = threading.Event()

        def _block_until_gate():
            gate.wait(timeout=5.0)

        dlcinterface._DLC_LOAD_STATE = "pending"
        dlcinterface._DLC_LOAD_THREAD = threading.Thread(target=_block_until_gate)
        dlcinterface._DLC_LOAD_THREAD.start()
        try:
            # Repeat ``_ensure_dlc_loaded_async`` returns the running
            # placeholder thread rather than spawning a second one.
            second = dlcinterface._ensure_dlc_loaded_async()
            assert second is dlcinterface._DLC_LOAD_THREAD
        finally:
            gate.set()
            dlcinterface._DLC_LOAD_THREAD.join(timeout=5.0)


class TestCallbacks:
    def test_callback_fires_when_already_done(self, _reset_loader):
        dlcinterface._DLC_LOAD_STATE = "done"
        calls = []
        dlcinterface.register_dlc_load_callback(lambda: calls.append("fired"))
        assert calls == ["fired"]

    def test_callback_fires_when_already_missing(self, _reset_loader):
        dlcinterface._DLC_LOAD_STATE = "missing"
        calls = []
        dlcinterface.register_dlc_load_callback(lambda: calls.append("fired"))
        assert calls == ["fired"]

    def test_callback_queued_when_pending(self, _reset_loader):
        dlcinterface._DLC_LOAD_STATE = "pending"
        calls = []
        dlcinterface.register_dlc_load_callback(lambda: calls.append("fired"))
        assert calls == []  # not fired yet
        assert len(dlcinterface._DLC_LOAD_CALLBACKS) == 1

    def test_one_failing_callback_does_not_block_others(
        self, _reset_loader, capsys
    ):
        dlcinterface._DLC_LOAD_STATE = "pending"
        calls = []

        def _bad():
            raise RuntimeError("boom")

        dlcinterface.register_dlc_load_callback(_bad)
        dlcinterface.register_dlc_load_callback(lambda: calls.append("ok"))

        # Drive the fan-out by flipping to "done" then firing.
        with dlcinterface._DLC_LOAD_LOCK:
            dlcinterface._DLC_LOAD_STATE = "done"
        dlcinterface._fire_dlc_load_callbacks()

        assert calls == ["ok"]
        # The bad callback's traceback went to stderr; assert the
        # surface so a future refactor that swallows the exception
        # silently is caught.
        err = capsys.readouterr().err
        assert "boom" in err


class TestSyncLoadSmoke:
    """One end-to-end exercise of the real ``deeplabcut`` import to
    pin that the lazy plumbing actually loads DLC (not just stubs
    the state machine). Skipped when DLC isn't installed.
    """

    def test_real_import_when_available(self, _reset_loader):
        if not dlcinterface.HAS_DLC:
            pytest.skip("deeplabcut not installed")
        # Reset state but DON'T null out the already-bound module
        # references -- the real import may already have happened in
        # another test or by an earlier ``dustrack.open()`` call. The
        # idempotency check below makes that fine.
        dlcinterface._DLC_LOAD_STATE = "pending"
        assert dlcinterface._ensure_dlc_loaded() is True
        assert dlcinterface._DLC_LOAD_STATE == "done"
        assert dlcinterface.deeplabcut is not None
        assert dlcinterface.VideoWriter is not None
        assert dlcinterface.ScannerError is not None
        # ``DLC3`` is True on the dlc3rc14 production env.
        assert dlcinterface.DLC3 is True
        # Second call is a fast no-op (cached state).
        t0 = time.perf_counter()
        dlcinterface._ensure_dlc_loaded()
        assert (time.perf_counter() - t0) < 0.01
