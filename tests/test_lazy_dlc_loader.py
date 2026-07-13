"""Tests for the lazy ``import deeplabcut`` plumbing.

The DLC import is ~7 s on the dlc3rc14 env and used to run at
``import dustrack`` time. The lazy loader (in ``dustrack/dlcloader.py``)
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

from dustrack import dlcloader


@pytest.fixture
def _reset_loader():
    """Save + restore the loader state so test cases that flip
    ``_DLC_LOAD_STATE`` / ``_DLC_LOAD_THREAD`` / module globals don't
    bleed into the rest of the suite.
    """
    saved = {
        "state": dlcloader._DLC_LOAD_STATE,
        "thread": dlcloader._DLC_LOAD_THREAD,
        "callbacks": list(dlcloader._DLC_LOAD_CALLBACKS),
        "deeplabcut": dlcloader.deeplabcut,
        "VideoWriter": dlcloader.VideoWriter,
        "ScannerError": dlcloader.ScannerError,
        "DLC3": dlcloader.DLC3,
        "HAS_DLC": dlcloader.HAS_DLC,
    }
    try:
        yield
    finally:
        dlcloader._DLC_LOAD_STATE = saved["state"]
        dlcloader._DLC_LOAD_THREAD = saved["thread"]
        dlcloader._DLC_LOAD_CALLBACKS[:] = saved["callbacks"]
        dlcloader.deeplabcut = saved["deeplabcut"]
        dlcloader.VideoWriter = saved["VideoWriter"]
        dlcloader.ScannerError = saved["ScannerError"]
        dlcloader.DLC3 = saved["DLC3"]
        dlcloader.HAS_DLC = saved["HAS_DLC"]


class TestStateMachine:
    def test_done_short_circuits_ensure(self, _reset_loader):
        dlcloader._DLC_LOAD_STATE = "done"
        assert dlcloader._ensure_dlc_loaded() is True

    def test_missing_short_circuits_ensure(self, _reset_loader):
        dlcloader._DLC_LOAD_STATE = "missing"
        assert dlcloader._ensure_dlc_loaded() is False

    def test_has_dlc_false_returns_false_without_import(self, _reset_loader):
        dlcloader.HAS_DLC = False
        dlcloader._DLC_LOAD_STATE = "pending"
        assert dlcloader._ensure_dlc_loaded() is False
        assert dlcloader._DLC_LOAD_STATE == "missing"

    def test_dlc_load_state_helper(self, _reset_loader):
        dlcloader._DLC_LOAD_STATE = "pending"
        dlcloader._DLC_LOAD_THREAD = None
        assert dlcloader._dlc_load_state() == "pending"

        dlcloader._DLC_LOAD_THREAD = threading.Thread(target=lambda: None)
        assert dlcloader._dlc_load_state() == "loading"

        dlcloader._DLC_LOAD_STATE = "done"
        assert dlcloader._dlc_load_state() == "done"

        dlcloader._DLC_LOAD_STATE = "missing"
        assert dlcloader._dlc_load_state() == "missing"


class TestAsyncLoader:
    def test_async_returns_quickly(self, _reset_loader):
        """``_ensure_dlc_loaded_async`` must return in << the
        synchronous ``import deeplabcut`` time so the bg load runs
        in parallel with the picker / GUI construction.
        """
        dlcloader._DLC_LOAD_STATE = "done"  # short-circuit; no real work
        t0 = time.perf_counter()
        result = dlcloader._ensure_dlc_loaded_async()
        elapsed = time.perf_counter() - t0
        assert result is None  # nothing to do
        # 50 ms is generous; the real cost is sub-millisecond.
        assert elapsed < 0.05

    def test_async_is_no_op_when_done(self, _reset_loader):
        dlcloader._DLC_LOAD_STATE = "done"
        assert dlcloader._ensure_dlc_loaded_async() is None

    def test_async_is_no_op_when_missing(self, _reset_loader):
        dlcloader._DLC_LOAD_STATE = "missing"
        assert dlcloader._ensure_dlc_loaded_async() is None

    def test_async_returns_none_when_has_dlc_false(self, _reset_loader):
        dlcloader.HAS_DLC = False
        dlcloader._DLC_LOAD_STATE = "pending"
        assert dlcloader._ensure_dlc_loaded_async() is None
        # Side effect: state was settled to ``"missing"`` so subsequent
        # sync callers short-circuit without touching the lock.
        assert dlcloader._DLC_LOAD_STATE == "missing"

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

        # This "return the already-running loader thread" path only applies
        # when DLC is importable; a no-DLC env short-circuits to None. Force
        # HAS_DLC True so the test is env-independent (_reset_loader restores).
        dlcloader.HAS_DLC = True
        dlcloader._DLC_LOAD_STATE = "pending"
        dlcloader._DLC_LOAD_THREAD = threading.Thread(target=_block_until_gate)
        dlcloader._DLC_LOAD_THREAD.start()
        try:
            # Repeat ``_ensure_dlc_loaded_async`` returns the running
            # placeholder thread rather than spawning a second one.
            second = dlcloader._ensure_dlc_loaded_async()
            assert second is dlcloader._DLC_LOAD_THREAD
        finally:
            gate.set()
            dlcloader._DLC_LOAD_THREAD.join(timeout=5.0)


class TestCallbacks:
    def test_callback_fires_when_already_done(self, _reset_loader):
        dlcloader._DLC_LOAD_STATE = "done"
        calls = []
        dlcloader.register_dlc_load_callback(lambda: calls.append("fired"))
        assert calls == ["fired"]

    def test_callback_fires_when_already_missing(self, _reset_loader):
        dlcloader._DLC_LOAD_STATE = "missing"
        calls = []
        dlcloader.register_dlc_load_callback(lambda: calls.append("fired"))
        assert calls == ["fired"]

    def test_callback_queued_when_pending(self, _reset_loader):
        dlcloader._DLC_LOAD_STATE = "pending"
        calls = []
        dlcloader.register_dlc_load_callback(lambda: calls.append("fired"))
        assert calls == []  # not fired yet
        assert len(dlcloader._DLC_LOAD_CALLBACKS) == 1

    def test_one_failing_callback_does_not_block_others(
        self, _reset_loader, capsys
    ):
        dlcloader._DLC_LOAD_STATE = "pending"
        calls = []

        def _bad():
            raise RuntimeError("boom")

        dlcloader.register_dlc_load_callback(_bad)
        dlcloader.register_dlc_load_callback(lambda: calls.append("ok"))

        # Drive the fan-out by flipping to "done" then firing.
        with dlcloader._DLC_LOAD_LOCK:
            dlcloader._DLC_LOAD_STATE = "done"
        dlcloader._fire_dlc_load_callbacks()

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
        if not dlcloader.HAS_DLC:
            pytest.skip("deeplabcut not installed")
        # Reset state but DON'T null out the already-bound module
        # references -- the real import may already have happened in
        # another test or by an earlier ``dustrack.open()`` call. The
        # idempotency check below makes that fine.
        dlcloader._DLC_LOAD_STATE = "pending"
        assert dlcloader._ensure_dlc_loaded() is True
        assert dlcloader._DLC_LOAD_STATE == "done"
        assert dlcloader.deeplabcut is not None
        assert dlcloader.VideoWriter is not None
        assert dlcloader.ScannerError is not None
        # ``DLC3`` is True on the dlc3rc14 production env.
        assert dlcloader.DLC3 is True
        # Second call is a fast no-op (cached state).
        t0 = time.perf_counter()
        dlcloader._ensure_dlc_loaded()
        assert (time.perf_counter() - t0) < 0.01
