"""Visual smoke test: open the M: drive multi-video project, show
the window, capture screenshots before and after a swap.

Run via::

    python tests/qt_learning/27_visual_smoke.py

Saves ``smoke_before.png`` and ``smoke_after.png`` to the cwd.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(
    "M:/DLC_MODELS/participant_models_general/s006/RFA/"
    "interosseous_pn24-x-2025-10-24"
)


def _wait_terminal(tracker, timeout=300.0):
    from qtpy.QtCore import QCoreApplication
    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(b.is_terminal for b in tracker._bundles):
            return
        QCoreApplication.processEvents()
        time.sleep(0.05)
    raise TimeoutError("hydration timed out")


def _grab(qt_window, path):
    from qtpy.QtGui import QPixmap
    pix = qt_window.grab()
    pix.save(path)
    print(f"  saved {path} ({pix.width()}x{pix.height()})")


def main() -> int:
    import dustrack
    tracker = dustrack.open(PROJECT_ROOT)
    print(f"opened: {len(tracker._bundles)} bundles, active={tracker._active_index}")
    _wait_terminal(tracker)
    print(f"all bundles ready: {[b.hydration_state for b in tracker._bundles]}")

    # Let the window paint a few frames.
    from qtpy.QtCore import QCoreApplication
    for _ in range(40):
        QCoreApplication.processEvents()
        time.sleep(0.02)

    qt_window = tracker._find_qt_window()
    if qt_window is None:
        print("no qt window -- mpl-fallback path?")
        return 1

    out_dir = Path(__file__).parent
    _grab(qt_window, str(out_dir / "smoke_before.png"))

    print(f"swap_to(3)...")
    tracker.swap_to(3)
    for _ in range(40):
        QCoreApplication.processEvents()
        time.sleep(0.02)
    _grab(qt_window, str(out_dir / "smoke_after.png"))

    print(f"swap_to(0)...")
    tracker.swap_to(0)
    for _ in range(40):
        QCoreApplication.processEvents()
        time.sleep(0.02)
    _grab(qt_window, str(out_dir / "smoke_back.png"))

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
