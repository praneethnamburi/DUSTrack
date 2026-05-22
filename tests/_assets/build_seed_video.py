"""One-time generator for the packaged seed video asset.

Produces:

- ``dustrack/_data/seed_video.mp4`` -- 8 frames, 64x64 mid-gray h264.
- ``dustrack/_data/seed_video.dnav-toc`` -- TOC sidecar matching the
  composite-suffix convention (no ``.json`` extension; sidecar
  consumers walk by suffix substring).

Run from the repo root once when the asset needs regenerating
(e.g. tuning encoder params, dimensions, frame count):

    python tests/_assets/build_seed_video.py

The script writes the .mp4 + .dnav-toc into ``dustrack/_data/`` so
they ship via ``pyproject.toml`` package-data on the next install /
build.

Featureless mid-gray (intensity 128) is intentional: the
``_VideoPickerOverlay`` modal sits on top with a rgba(0, 0, 0, 200)
backdrop, so the seed frame is barely visible. Keeping it
contentless avoids a wordmark flash on the moment of the seed -> real
swap.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

# Output paths relative to repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "dustrack" / "_data"
OUT_MP4 = OUT_DIR / "seed_video.mp4"
# dnav's TOC convention is ``<filename>.dnav-toc`` (preserves the
# source video's full filename including extension), NOT
# ``<stem>.dnav-toc``. See [[no-json-extension-for-sidecars]].
OUT_TOC = OUT_DIR / "seed_video.mp4.dnav-toc"

# Asset parameters. Conservative -- DUSTrack's pia02 production videos
# are 706x558; a 64x64 stub is comfortably smaller than any real video
# while still large enough for the matplotlib image-axis to render
# something. 8 frames is enough for the trace pane to draw a tiny axis
# range without a single-frame edge case.
WIDTH = 64
HEIGHT = 64
N_FRAMES = 8
FPS = 30
PIXEL_VALUE = 128  # mid-gray


def _ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg before running this "
            "asset-build script."
        )


def _encode_mp4(out_path: Path) -> None:
    """Encode the seed video via ffmpeg's rawvideo pipe.

    h264 with yuv420p so every downstream cv2 / PyAV / dnav backend
    reads it the same way. ``-pix_fmt yuv420p`` on input + output
    + ``-crf 23`` keeps the file in the ~1-2 KB range we want.
    """
    _ensure_ffmpeg()
    # Mid-gray frame: a single (H, W) array stamped per frame.
    frame = np.full((HEIGHT, WIDTH), PIXEL_VALUE, dtype=np.uint8)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-pix_fmt", "gray",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        "-preset", "veryfast",
        # Single-frame-per-packet so PyAV's per-frame seek is fast
        # for the unlikely case anyone navigates inside the seed.
        "-x264-params", "keyint=1:min-keyint=1",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for _ in range(N_FRAMES):
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        ret = proc.wait()
    except BrokenPipeError:
        proc.wait()
        ret = proc.returncode
    if ret != 0:
        raise RuntimeError(f"ffmpeg returned {ret}; seed video build failed")


def _build_toc_via_dnav(mp4_path: Path, toc_path: Path) -> None:
    """Trigger dnav's own TOC build by opening the freshly-encoded
    video with :class:`datanavigator.VideoReader`. dnav writes the
    sidecar at ``<filename>.dnav-toc`` in canonical format -- we
    commit the file dnav produced rather than hand-rolling a
    matching shape that could drift from the dnav contract.

    Removes any pre-existing TOC first so the rebuild is deterministic.
    """
    if toc_path.is_file():
        toc_path.unlink()
    from datanavigator import VideoReader
    with mp4_path.open("rb") as f:
        reader = VideoReader(f)
        # Force a length probe so the TOC is fully populated.
        _ = len(reader)
    if not toc_path.is_file():
        raise RuntimeError(
            f"dnav did not produce a TOC at {toc_path} after opening "
            f"{mp4_path}; check dnav's VideoReader contract."
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing {OUT_MP4}...")
    _encode_mp4(OUT_MP4)
    size = OUT_MP4.stat().st_size
    print(f"  -> {size} bytes")
    print(f"Building {OUT_TOC} via dnav.VideoReader...")
    _build_toc_via_dnav(OUT_MP4, OUT_TOC)
    print(f"  -> {OUT_TOC.stat().st_size} bytes")
    # Clean up any stale hand-rolled sidecar from earlier iterations
    # of this script (the pre-dnav-built version used
    # ``<stem>.dnav-toc`` which doesn't match dnav's convention).
    stale = OUT_DIR / "seed_video.dnav-toc"
    if stale.is_file():
        stale.unlink()
        print(f"  (removed stale {stale.name})")
    print("Done. Commit dustrack/_data/seed_video.mp4 + .mp4.dnav-toc.")


if __name__ == "__main__":
    main()
