"""
Microbench: DUSTrack's enhance_ultrasound_image() per-frame cost.

Question being answered: how much of the real-DUSTrack ~128.6 ms / frame
budget is spent inside the CLAHE+gamma+brightness pipe that runs on the
imshow input every update()? Reference point is the ~13 ms / frame the
Qt widget swap shaved off (1.3.0: 141.6 ms -> 1.4.0-qt: 128.6 ms).

Methodology
-----------
- Synthetic 706x558 RGB uint8 frame (matches the `interosseous_pn24-x`
  crop reported in M:\\DLC_MODELS\\...\\config.yaml: `crop: 0, 706, 0, 558`).
- Calls enhance_ultrasound_image with DUSTrack's default kwargs
  (clahe_clip=2.0, clahe_grid=8, gamma=1.2, brightness=10).
- Discards first 10 calls (CLAHE warmup, BLAS lazy init).
- Reports median / p95 / mean over the next N calls.

Also times an inlined variant with a pre-baked gamma LUT (the current
function rebuilds a 256-entry table with a Python list comprehension on
every call -- a clear no-op fix worth quantifying separately).

Standalone -- no DUSTrack / datanavigator imports needed.

Findings (2026-05-17, b4 env, cv2 4.7.0, numpy 1.26.4)
-------------------------------------------------------
- **Total: ~1.44 ms / frame** -- about **1%** of the 128.6 ms
  real-DUSTrack budget, and about **1/9** of the ~13 ms the Qt widget
  swap shaved off in 1.4.0. **Image enhance is not where the wins are.**
- Per-step breakdown (medians):
    - RGB->GRAY + CLAHE         0.28 ms
    - brightness add+clip       0.54 ms   <-- 2x CLAHE; numpy int16 round-trip
    - GRAY->RGB                 0.20 ms
    - cv.LUT (gamma)            0.04 ms
    - gamma LUT rebuild (Python listcomp)   0.025 ms
- Realistic ceiling from cleanup: ~0.7 ms recoverable by caching the
  LUT + reusing the CLAHE object, swapping
  `np.clip(astype(int16) + bright, 0, 255).astype(uint8)` for
  `cv.add(gray, bright)` (saturated 8-bit), and dropping the trailing
  GRAY->RGB. Still negligible vs. the Qt-swap baseline.
- Caveat: random-noise input rather than real ultrasound. CLAHE cost is
  tile-traversal-dominated, so the gap is at most ~20%, not 5x.

Implication for 1.4.0 perf work: image enhance lives in the "smaller
wins" tier with the QLabel / update_display() items in TODO.md. The
real targets remain blit-mode rendering (tens of ms / frame) and
PyAV pre-decode / lookahead (~30-50 ms / frame).
"""

import statistics
import sys
import time

import cv2 as cv
import numpy as np


# --- The function under test, copied verbatim from
#     C:/dev/DUSTrack/dustrack/dlcinterface.py:48-83 so this script can
#     run without DUSTrack installed. -----------------------------------

def enhance_ultrasound_image(image, clahe_clip=2.0, clahe_grid=8, gamma=1.0, brightness=0):
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    else:
        gray = image

    clahe = cv.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    enhanced = clahe.apply(gray)

    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
        ).astype("uint8")
        enhanced = cv.LUT(enhanced, table)

    if brightness != 0:
        enhanced = np.clip(
            enhanced.astype(np.int16) + brightness, 0, 255
        ).astype(np.uint8)

    return cv.cvtColor(enhanced, cv.COLOR_GRAY2RGB)


# --- "Optimized" variants, just enough to attribute cost ---------------

def enhance_v2_cached_lut(image, clahe, lut, brightness):
    """CLAHE object + gamma LUT are pre-built once and passed in."""
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    else:
        gray = image
    enhanced = clahe.apply(gray)
    if lut is not None:
        enhanced = cv.LUT(enhanced, lut)
    if brightness != 0:
        enhanced = np.clip(
            enhanced.astype(np.int16) + brightness, 0, 255
        ).astype(np.uint8)
    return cv.cvtColor(enhanced, cv.COLOR_GRAY2RGB)


def _build_lut(gamma):
    inv_gamma = 1.0 / gamma
    return np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype("uint8")


# --- Microbench harness ------------------------------------------------

def _stats(times_ms):
    s = sorted(times_ms)
    n = len(s)
    return {
        "n": n,
        "min": min(times_ms),
        "median": statistics.median(times_ms),
        "mean": statistics.mean(times_ms),
        "p95": s[int(n * 0.95)],
        "max": max(times_ms),
    }


def _print_row(label, st):
    print(
        f"  {label:40s} "
        f"median={st['median']:6.3f} ms  "
        f"mean={st['mean']:6.3f}  "
        f"p95={st['p95']:6.3f}  "
        f"min={st['min']:6.3f}  "
        f"max={st['max']:6.3f}  "
        f"n={st['n']}"
    )


def _time(fn, n_warmup=10, n_iter=300):
    for _ in range(n_warmup):
        fn()
    out = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        out.append((t1 - t0) * 1000)
    return _stats(out)


def main():
    # Real DUSTrack frame shape from config.yaml: crop 0, 706, 0, 558.
    # The image_process_func receives the RGB frame from PyAV (3-channel).
    H, W = 558, 706
    rng = np.random.default_rng(42)
    frame_rgb = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)

    # DUSTrack's defaults (dlcinterface.py:128).
    kwargs = dict(clahe_clip=2.0, clahe_grid=8, gamma=1.2, brightness=10)

    print("=" * 78)
    print(f"enhance_ultrasound_image microbench  ({H}x{W} RGB uint8)")
    print(f"kwargs: {kwargs}")
    print(f"cv2: {cv.__version__}   numpy: {np.__version__}")
    print("=" * 78)

    # 1. Verbatim DUSTrack implementation.
    st_full = _time(lambda: enhance_ultrasound_image(frame_rgb, **kwargs))
    _print_row("current DUSTrack (LUT rebuilt/call)", st_full)

    # 2. Pre-baked LUT + reused CLAHE object.
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lut = _build_lut(1.2)
    st_v2 = _time(lambda: enhance_v2_cached_lut(frame_rgb, clahe, lut, 10))
    _print_row("cached LUT + reused CLAHE object", st_v2)

    # 3. Components, to attribute the remaining cost.
    st_clahe = _time(lambda: clahe.apply(
        cv.cvtColor(frame_rgb, cv.COLOR_RGB2GRAY)
    ))
    _print_row("  RGB->GRAY + CLAHE only", st_clahe)

    gray = cv.cvtColor(frame_rgb, cv.COLOR_RGB2GRAY)
    st_lut = _time(lambda: cv.LUT(gray, lut))
    _print_row("  cv.LUT only", st_lut)

    st_bright = _time(lambda: np.clip(
        gray.astype(np.int16) + 10, 0, 255
    ).astype(np.uint8))
    _print_row("  brightness add+clip only", st_bright)

    st_rgb = _time(lambda: cv.cvtColor(gray, cv.COLOR_GRAY2RGB))
    _print_row("  GRAY->RGB only", st_rgb)

    st_lut_build = _time(lambda: _build_lut(1.2))
    _print_row("  gamma LUT rebuild (Python listcomp)", st_lut_build)

    print("=" * 78)
    print(f"Reference: Qt widget swap saved ~13 ms / frame on real DUSTrack")
    print(f"           (1.3.0: 141.6 ms -> 1.4.0-qt: 128.6 ms median)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
