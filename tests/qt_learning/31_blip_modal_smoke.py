"""Manual Qt smoke harness for the Detect-blip-outliers workflow.

Run from the repo root in the dlc3rc14 env::

    C:/Users/praneeth/anaconda3/envs/dlc3rc14/python.exe tests/qt_learning/31_blip_modal_smoke.py

Opens DUSTrack on a pia02 video + its iteration-0 DLC prediction
trace. The user drives the smoke by hand:

1. Switch the active layer to the DLC trace
   (sidebar statevars: ``annotation_layer``).
2. Click **Detect blip outliers** (Workflow group, right after
   Reduce jitter). Modal opens.
3. Click **Detect** -- per-label results populate in <1 s.
4. Tune knobs (e.g. threshold factor 4.0) + Detect again to confirm
   the results pane refreshes.
5. Click **Interpolate** -- modal closes, ProgressOverlay shows the
   LK progress bar walking up to N/N.
6. Click **Done** -- sparse blip-corrections layer adopts as the
   active layer with the DLC trace pinned as overlay; trace pane
   shows the corrections at the flagged frames.
7. (Optional) Cancel-path: re-click the button, click Cancel in the
   modal -- no side effects, no new layer.
8. (Optional) Overwrite-path: re-click the button + Interpolate
   with the file from step 6 still on disk -- the confirm-overlay
   asks whether to overwrite. Cancel keeps the existing file;
   Overwrite re-runs.

Not added to pytest collection (Qt-learning scripts intentionally
sit outside the collected test root).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

import dustrack


VIDEO = Path(
    "M:/DLC_MODELS/general/interosseous_pn24-x-2025-10-24/videos/"
    "pia02_s001_007_RFA2.mp4"
)


def main() -> int:
    if not VIDEO.exists():
        print(f"FATAL: video not found at {VIDEO}")
        print("Edit the VIDEO constant in this script to point at any pia02 video.")
        return 1

    print(f"Opening DUSTrack on {VIDEO.name}...")
    print(
        "Follow the steps in the module docstring; "
        "close the window when you're done."
    )
    tracker = dustrack.open(str(VIDEO))
    if tracker is None:
        print("FATAL: dustrack.open returned None")
        return 1
    plt.show(block=True)
    print("Smoke harness exited cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
