"""End-to-end smoke test for the 1.2.0a3 multi-video swap.

Opens the real pia02 s006 DLC project from the user's M: drive and
verifies the construction + swap_to mechanics without requiring a
live user. Pumps the Qt event loop briefly so the bg hydration
worker can flush before we exercise swaps.

Run via::

    python tests/qt_learning/26_smoke_multi_video.py

Prints a structured progress report; on any AssertionError the test
fails loudly.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(
    "M:/DLC_MODELS/participant_models_general/s006/RFA/"
    "interosseous_pn24-x-2025-10-24"
)


def _wait_for_hydration(tracker, timeout: float = 60.0) -> None:
    """Pump the Qt event loop until every bundle is terminal or
    timeout expires. Logs progress every 5s."""
    from qtpy.QtCore import QCoreApplication
    deadline = time.time() + timeout
    last_log = time.time()
    while time.time() < deadline:
        terminal = sum(1 for b in tracker._bundles if b.is_terminal)
        if terminal == len(tracker._bundles):
            return
        if time.time() - last_log > 5.0:
            ready = sum(1 for b in tracker._bundles if b.is_ready)
            print(f"  ... {terminal}/{len(tracker._bundles)} terminal "
                  f"({ready} ready) after {time.time() - (deadline - timeout):.0f}s")
            last_log = time.time()
        QCoreApplication.processEvents()
        time.sleep(0.05)
    pending = [b.video_index for b in tracker._bundles if not b.is_terminal]
    raise TimeoutError(
        f"hydration didn't finish within {timeout}s; still-pending: {pending}"
    )


def main() -> int:
    assert PROJECT_ROOT.exists(), f"missing project root: {PROJECT_ROOT}"
    import dustrack

    print(f"opening multi-video session against {PROJECT_ROOT}")
    t0 = time.time()
    tracker = dustrack.open(PROJECT_ROOT)
    t1 = time.time()
    assert tracker is not None, "dustrack.open returned None"
    print(f"  construction: {t1 - t0:.2f}s")
    print(f"  bundles      : {len(tracker._bundles)}")
    print(f"  active index : {tracker._active_index}")
    print(f"  active fname : {tracker.fname}")
    assert tracker._active_index == 0
    assert len(tracker._bundles) > 1, (
        "expected multiple bundles; project may be misconfigured"
    )

    print("waiting for bg hydration to flush ...")
    t0 = time.time()
    _wait_for_hydration(tracker, timeout=300.0)
    t1 = time.time()
    print(f"  hydration    : {t1 - t0:.2f}s")
    ready = [b.video_index for b in tracker._bundles if b.is_ready]
    failed = [(b.video_index, b.hydration_error)
              for b in tracker._bundles if b.hydration_state == "failed"]
    print(f"  ready        : {len(ready)} / {len(tracker._bundles)}")
    if failed:
        print(f"  FAILED       : {failed}")
        return 1

    # Snapshot some state before swap.
    initial_idx = tracker._current_idx
    initial_fname = tracker.fname
    initial_layers = tracker.annotations.names
    print(f"  initial layers : {initial_layers}")

    # Swap to bundle 1.
    print("swap_to(1)...")
    t0 = time.time()
    ok = tracker.swap_to(1)
    t1 = time.time()
    print(f"  swap_to(1) -> {ok} ({(t1 - t0) * 1000:.1f}ms)")
    assert ok, "swap_to(1) returned False"
    assert tracker._active_index == 1
    assert tracker.fname != initial_fname, (
        f"shell.fname unchanged after swap: {tracker.fname}"
    )
    print(f"  active fname : {tracker.fname}")
    print(f"  active layers: {tracker.annotations.names}")

    # Swap back to 0 -- should be near-instant (artists parked).
    print("swap_to(0)...")
    t0 = time.time()
    ok = tracker.swap_to(0)
    t1 = time.time()
    print(f"  swap_to(0) -> {ok} ({(t1 - t0) * 1000:.1f}ms)")
    assert ok
    assert tracker._active_index == 0
    assert tracker.fname == initial_fname
    assert tracker.annotations.names == initial_layers, (
        f"layer list changed after round-trip swap"
    )
    print(f"  round-trip preserved layer list: {tracker.annotations.names}")

    # Swap to the last bundle.
    last = len(tracker._bundles) - 1
    print(f"swap_to({last})...")
    t0 = time.time()
    ok = tracker.swap_to(last)
    t1 = time.time()
    print(f"  swap_to({last}) -> {ok} ({(t1 - t0) * 1000:.1f}ms)")
    assert ok
    assert tracker._active_index == last

    # Bounds checks.
    assert tracker.swap_to(99) is False, "out-of-bounds should return False"
    assert tracker.swap_to(-1) is False, "negative should return False"
    assert tracker.swap_to(last) is True, "no-op at current index should return True"

    print()
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
