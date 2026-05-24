"""Real-data smoke harness for ``dustrack.blip``.

Run from the repo root in the dlc3rc14 env::

    C:/Users/praneeth/anaconda3/envs/dlc3rc14/python.exe tests/qt_learning/30_blip_demo.py

Loads one DLC predicted-trace .h5 from the pia02 general-model project,
runs sparse-blip detection and LK-RSTC interpolation, prints per-label
stats and the first 5 before/after corrections, and writes the sparse
corrections to ``<stem>_blip_corrections.json`` next to the source.

**Cleanup convention (per [[benchmark-cleanup-collision]]):** refuses to
overwrite an existing corrections JSON. If a prior run left one in
place, delete or rename it first; do not lose user state through
silent overwrite.

This is not a pytest test (Qt-learning scripts intentionally sit
outside the collected test root).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import dustrack


# Pick a representative pia02 DLC h5 trace. The iteration-0 folder
# contains snapshot_300 predictions for every video in the corpus;
# pia02_s001_006_RFA2 is a 36715-frame ultrasound clip that appears
# repeatedly in the perf-bench corpus and has a known mix of clean
# and motion-rich segments.
DLC_H5 = Path(
    "M:/DLC_MODELS/general/interosseous_pn24-x-2025-10-24/videos/iteration-0/"
    "pia02_s001_007_RFA2DLC_Resnet50_interosseous_pn24Oct24shuffle1_snapshot_300.h5"
)
VIDEO = Path(
    "M:/DLC_MODELS/general/interosseous_pn24-x-2025-10-24/videos/"
    "pia02_s001_007_RFA2.mp4"
)


def main() -> int:
    if not DLC_H5.exists():
        print(f"FATAL: DLC trace not found at {DLC_H5}")
        return 1
    if not VIDEO.exists():
        print(f"FATAL: video not found at {VIDEO}")
        return 1

    # Pre-existing corrections file -- abort rather than overwrite.
    expected_out = DLC_H5.parent / f"{DLC_H5.stem}_blip_corrections.json"
    if expected_out.exists():
        print(f"FATAL: existing corrections file at {expected_out}")
        print("Delete or rename it first; demo refuses to overwrite user state.")
        return 1

    print(f"Loading {DLC_H5.name}...")
    t0 = time.perf_counter()
    ann = dustrack.VideoAnnotation(fname=str(DLC_H5), vname=str(VIDEO))
    print(f"  -> labels={ann.labels}  n_frames={ann.n_frames}  ({time.perf_counter()-t0:.2f}s)")

    print(f"\nDetecting blips (defaults: factor=5.0, max_len=5, return_factor=3.0)...")
    t0 = time.perf_counter()
    report = dustrack.detect_blips(ann)
    dt_detect = time.perf_counter() - t0
    print(f"  -> {len(report)} blips found across {len(ann.labels)} labels ({dt_detect:.3f}s)")

    print("\nPer-label stats:")
    for label, stats in report.per_label_stats.items():
        print(
            f"  label={label!r}:  "
            f"med={stats['median_d']:.3f}  "
            f"mad={stats['mad_d']:.3f}  "
            f"threshold={stats['threshold']:.3f}  "
            f"n_blips={stats['n_blips']}  "
            f"skipped_edge={stats['n_skipped_edge']}  "
            f"skipped_long={stats['n_skipped_long']}  "
            f"skipped_noreturn={stats['n_skipped_noreturn']}"
        )

    if len(report) > 0:
        print("\nBlip length histogram (run_length: count):")
        for length, count in report.length_histogram().items():
            print(f"  {length}: {count}")

    if len(report) == 0:
        print("\nNo blips detected; skipping interpolation.")
        return 0

    print(f"\nInterpolating {len(report)} blips via LK-RSTC...")
    t0 = time.perf_counter()
    out = dustrack.interpolate_blips(ann, report)
    dt_interp = time.perf_counter() - t0
    print(f"  -> sparse output built ({dt_interp:.3f}s)")

    print("\nFirst 5 corrections (label / frame / before -> after):")
    for blip in report.blips[:5]:
        for offset in range(blip.length):
            frame = blip.start + offset
            before_xy = ann.data[blip.label].get(frame, [None, None])
            after_xy = out.data[blip.label].get(frame, [None, None])
            print(
                f"  label={blip.label!r}  frame={frame}  "
                f"before=({before_xy[0]:.2f}, {before_xy[1]:.2f})  "
                f"after=({after_xy[0]:.2f}, {after_xy[1]:.2f})"
            )

    print(f"\nWriting sparse corrections to {expected_out.name}...")
    out.save()
    size = expected_out.stat().st_size
    print(f"  -> {size} bytes")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
