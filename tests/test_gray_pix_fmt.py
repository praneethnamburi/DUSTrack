"""Smoke tests for the dnav 1.5.0a2 ``pix_fmt='gray'`` integration.

Dnav auto-detects monochrome-encoded sources (h265 ``pix_fmt=gray``)
and decodes them directly as (H, W) gray ndarrays. Dustrack's hot
gray-touching code paths (``postprocess.gray``, ``opticalflow._gray_rgb``,
``dlcinterface.enhance_ultrasound_image``) must transparently handle
both shapes:

* (H, W, 3) RGB -- the historical color-source path
* (H, W)    gray -- the new monochrome-source path

These tests pin that contract.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dustrack.lk_filter import gray
from dustrack.lk_opticalflow import _gray_rgb
from dustrack._image_enhance import enhance_ultrasound_image


FPS = 24
DURATION_S = 1
N_FRAMES = FPS * DURATION_S


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


pytestmark = pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not on PATH")


@pytest.fixture(scope="module")
def mono_clip(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("gray_pix_fmt") / "mono.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=duration={DURATION_S}:size=64x48:rate={FPS}",
            "-c:v", "libx265", "-pix_fmt", "gray", "-x265-params", "log-level=none",
            str(out),
        ],
        check=True,
    )
    return out


# ---------- postprocess.gray ----------

def test_postprocess_gray_passes_2d_through():
    """2D input is already grayscale -- short-circuit, no cvtColor."""
    src = np.random.default_rng(0).integers(0, 255, (48, 64), dtype=np.uint8)
    out = gray(src)
    assert out is src or np.array_equal(out, src)
    assert out.shape == (48, 64)


def test_postprocess_gray_converts_rgb_via_bt601():
    """3D RGB input goes through cv2.cvtColor(RGB2GRAY) -- BT.601 luminance."""
    import cv2 as cv
    rgb = np.random.default_rng(1).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    expected = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    out = gray(rgb)
    assert out.shape == (48, 64)
    assert np.array_equal(out, expected)


# ---------- opticalflow._gray_rgb ----------

class _FakeFrame:
    def __init__(self, arr):
        self._arr = arr

    def asnumpy(self):
        return self._arr


class _FakeVideo:
    """Stand-in for dnav VideoReader: indexing returns a frame with .asnumpy()."""

    def __init__(self, frames):
        self._frames = frames

    def __getitem__(self, i):
        return _FakeFrame(self._frames[i])


def test_gray_rgb_passes_2d_gray_through():
    """If video[i].asnumpy() returns 2D (mono source via dnav auto-detect),
    _gray_rgb returns it unchanged."""
    frames = [np.random.default_rng(2).integers(0, 255, (48, 64), dtype=np.uint8)]
    out = _gray_rgb(_FakeVideo(frames), 0)
    assert out.shape == (48, 64)
    assert np.array_equal(out, frames[0])


def test_gray_rgb_converts_rgb_input():
    """3D RGB input is converted via BT.601."""
    import cv2 as cv
    rgb = np.random.default_rng(3).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    expected = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    out = _gray_rgb(_FakeVideo([rgb]), 0)
    assert out.shape == (48, 64)
    assert np.array_equal(out, expected)


# ---------- enhance_ultrasound_image ----------

def test_enhance_returns_rgb_for_gray_input():
    """Existing branch already handles 2D input; the contract is that the
    output is always 3-channel RGB for downstream matplotlib display."""
    src = np.random.default_rng(4).integers(20, 200, (48, 64), dtype=np.uint8)
    out = enhance_ultrasound_image(src, clahe_clip=2.0, gamma=1.0)
    assert out.shape == (48, 64, 3)
    assert out.dtype == np.uint8


def test_enhance_returns_rgb_for_rgb_input():
    src = np.random.default_rng(5).integers(20, 200, (48, 64, 3), dtype=np.uint8)
    out = enhance_ultrasound_image(src, clahe_clip=2.0, gamma=1.0)
    assert out.shape == (48, 64, 3)


# ---------- End-to-end: open a mono clip via dnav, exercise gray helpers ----------

def test_end_to_end_mono_clip_through_gray_helpers(mono_clip):
    """Open a mono h265 clip via dnav -> auto-detect picks gray ->
    postprocess.gray and _gray_rgb both pass through; enhance returns RGB."""
    from datanavigator import VideoReader

    vr = VideoReader(str(mono_clip))
    assert vr.pix_fmt == "gray"

    # gray() short-circuits.
    frame0 = vr[0].asnumpy()
    assert frame0.ndim == 2
    g = gray(frame0)
    assert g.shape == frame0.shape
    assert np.array_equal(g, frame0)

    # _gray_rgb short-circuits.
    g2 = _gray_rgb(vr, 0)
    assert g2.shape == frame0.shape

    # enhance coerces back to RGB.
    enhanced = enhance_ultrasound_image(frame0, clahe_clip=2.0, gamma=1.2)
    assert enhanced.shape == (frame0.shape[0], frame0.shape[1], 3)
