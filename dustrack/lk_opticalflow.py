"""
Per-video Lucas-Kanade helpers used by
:py:meth:`dustrack.DUSTrack.predict_labels_with_lucas_kanade`
(absorbed from the pre-1.2.0rc1 ``_DUSTrackBase`` parent class when
that split collapsed into :class:`dustrack.DUSTrack`).

These are the *per-video* shapes (called with ``video, start_frame, end_frame, ...``).
The frame-list shapes used by the LK-filter pipeline
(``lucas_kanade_2`` / ``lucas_kanade_rstc_2``) live in :py:mod:`dustrack.lk_filter`
-- both shapes delegate to :func:`_lk_track_frames` here so the per-pair
LK loop has a single home.

Lucas-Kanade + reverse sigmoid tracking correction (RSTC) is described in:

Magana-Salgado, U., Namburi, P., Feigin-Almon, M., Pallares-Lopez, R., & Anthony, B (2023)
A comparison of point-tracking algorithms in ultrasound videos from the upper limb.
BioMedical Engineering OnLine, 22(1), 52.
https://doi-org.libproxy.mit.edu/10.1186/s12938-023-01105-y

Moved here from ``datanavigator.opticalflow`` in datanavigator 1.5.0a1 /
dustrack 1.2.0a1; renamed from ``dustrack.opticalflow`` to
``dustrack.lk_opticalflow`` in dustrack 1.2.0rc1. Full pre-relocation
history preserved via ``git log --follow dustrack/lk_opticalflow.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import cv2 as cv
import numpy as np
from datanavigator.video_reader import VideoReader

from datanavigator import utils


_DEFAULT_LK_CONFIG = dict(
    winSize=(45, 45),
    maxLevel=2,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
)


def _normalize_init_points(init_points: np.ndarray) -> np.ndarray:
    """Coerce to ``(n_points, 1, 2)`` float32 -- the shape OpenCV's LK wants."""
    init_points = np.asarray(init_points, dtype=np.float32)
    if init_points.ndim == 1:
        init_points = init_points[np.newaxis, :]
    assert init_points.shape[-1] == 2
    if init_points.ndim == 2:
        init_points = init_points.reshape((init_points.shape[0], 1, 2))
    return init_points


def _gray_rgb(video, frame_num: int) -> np.ndarray:
    """Decode a frame from a VideoReader / utils.Video and grayscale it.

    PyAV+TOC normally returns RGB; dnav 1.5.0a2 auto-detects
    monochrome-encoded sources (h265 ``pix_fmt=gray``) and returns
    (H, W) gray directly, in which case the cvtColor step short-circuits.
    Color-source path uses ``COLOR_RGB2GRAY`` (BT.601 luminance);
    unified with :py:func:`dustrack.lk_filter.gray` in the 1.2.0a2
    perf pass -- the LK-filter path previously used ``COLOR_BGR2GRAY``
    on the same RGB input, which swapped the R/B coefficients and
    produced a different (but still grayscale) value.
    """
    arr = video[frame_num].asnumpy()
    if arr.ndim == 2:
        return arr
    return cv.cvtColor(arr, cv.COLOR_RGB2GRAY)


def _lk_track_frames(
    frame_list: list[np.ndarray],
    init_points: np.ndarray,
    **lk_config,
) -> np.ndarray:
    """Canonical pyramidal-LK per-pair loop over pre-decoded grayscale frames.

    Returned shape ``(n_frames, n_points, 2)``. Row 0 is ``init_points``;
    row k is the location at ``frame_list[k]``. Both
    :func:`lucas_kanade` (per-video) and
    :func:`dustrack.lk_filter.lucas_kanade_2` (frame-list) delegate
    here so the LK call lives in exactly one place.

    NOTE: pyramid reuse across pairs (build once for ``ff``, pass into
    the next pair as ``fi``) is the textbook LK optimisation, but the
    opencv-python 4.11 binding of ``calcOpticalFlowPyrLK`` does not
    accept a pre-built pyramid list as ``prevImg`` / ``nextImg`` --
    the C++ ``vector<Mat>`` overload is not exposed to Python (the
    binding rejects tuples and lists of ndarrays alike, demanding
    ``Ptr<UMat>``). So each call rebuilds both pyramids internally;
    interior frames are pyramid-built twice. Re-evaluate if the
    binding surfaces this overload in a later release.
    """
    init_points = _normalize_init_points(init_points)
    cfg = {**_DEFAULT_LK_CONFIG, **lk_config}

    n_frames = len(frame_list)
    n_points = init_points.shape[0]
    tracked_points = np.empty((n_frames, n_points, 2))
    tracked_points[0] = init_points[:, 0, :]

    fi = frame_list[0]
    for frame_idx in range(1, n_frames):
        ff = frame_list[frame_idx]
        fp, _, _ = cv.calcOpticalFlowPyrLK(fi, ff, init_points, None, **cfg)
        tracked_points[frame_idx] = fp[:, 0, :]
        init_points = fp
        fi = ff
    return tracked_points


def lucas_kanade(
    video: Union[utils.Video, VideoReader, str, Path],
    start_frame: int,
    end_frame: int,
    init_points: np.ndarray,
    mode: str = "full",
    **lk_config,
) -> np.ndarray:
    """Track points in a video using Lucas-Kanade algorithm.

    Args:
        video (Union[utils.Video, VideoReader, str, Path]): Video object or path to video file.
        start_frame (int): Initial frame for tracking.
        end_frame (int): Final frame (inclusive).
        init_points (np.ndarray): n_points x 2. Locations to be tracked at start_frame.
        mode (str, optional): 'full' tracks the points at every frame in the entire segment.
            'direct' tracks the point at the last frame using the first frame.
            Defaults to 'full'.
        **lk_config: Additional configuration for Lucas-Kanade algorithm.

    Returns:
        np.ndarray: n_frames x n_points x 2, includes start and end frame for 'full',
            and 1 x n_points x 2 for 'direct'.
    """
    if isinstance(video, (str, Path)):
        assert os.path.exists(video)
        video = VideoReader(video)

    assert mode in ("direct", "full")
    init_points = _normalize_init_points(init_points)
    cfg = {**_DEFAULT_LK_CONFIG, **lk_config}

    if mode == "direct":
        fi = _gray_rgb(video, start_frame)
        ff = _gray_rgb(video, end_frame)
        fp, _, _ = cv.calcOpticalFlowPyrLK(fi, ff, init_points, None, **cfg)
        return fp[:, 0, :][np.newaxis, :, :]

    # mode == "full": prefetch frames in tracking order, delegate to the
    # canonical helper. Forward goes start -> end inclusive; reverse goes
    # start -> end inclusive but the step is -1 so the first frame in
    # the list is ``start_frame`` and the last is ``end_frame``. Matches
    # the historical contract that tracked_points[0] is ``init_points``.
    step = 1 if end_frame > start_frame else -1
    frame_numbers = range(start_frame, end_frame + step, step)
    frame_list = [_gray_rgb(video, int(fn)) for fn in frame_numbers]
    return _lk_track_frames(frame_list, init_points, **lk_config)


def lucas_kanade_rstc(
    video: Union[utils.Video, VideoReader, str, Path],
    start_frame: int,
    end_frame: int,
    start_points: np.ndarray,
    end_points: np.ndarray,
    target_frame: int = None,
    **lk_config,
) -> np.ndarray:
    """Track points in a video using Lucas-Kanade algorithm,
    and apply the reverse sigmoid tracking correction (RSTC)
    as described in Magana-Salgado et al., 2023.

    Args:
        video (Union[utils.Video, VideoReader, str, Path]): Video object or path to video file.
        start_frame (int): Initial frame for tracking.
        end_frame (int): Final frame (inclusive).
        start_points (np.ndarray): n_points x 2. Locations to be tracked at start_frame.
        end_points (np.ndarray): n_points x 2. Locations to be tracked at end_frame.
        target_frame (int, optional): Target frame for direct mode. Defaults to None.
        **lk_config: Additional configuration for Lucas-Kanade algorithm.

    Returns:
        np.ndarray: Corrected tracking path.
    """
    assert end_frame > start_frame

    if target_frame is None:
        mode = "full"
    else:
        assert isinstance(target_frame, int)
        mode = "direct"

    if isinstance(video, (str, Path)):
        assert os.path.exists(video)
        video = VideoReader(video)

    if mode == "full":
        # Decode each frame in [start_frame, end_frame] once and share
        # the grayscale frame list between the forward and reverse
        # passes. Pre-refactor this path called ``lucas_kanade`` twice,
        # which decoded every frame in both directions -- and the
        # reverse-direction decode is much more expensive than the
        # forward one on PyAV+TOC (sub-keyframe reverse seeks). The
        # forward-then-reverse-view shares decode work cleanly.
        frames_fwd = [_gray_rgb(video, fn) for fn in range(start_frame, end_frame + 1)]
        frames_rev = frames_fwd[::-1]
        forward_path = _lk_track_frames(frames_fwd, start_points, **lk_config)
        reverse_path = _lk_track_frames(frames_rev, end_points, **lk_config)
    else:
        # Direct mode: each call decodes only the two endpoints, so the
        # combined shape is 4 decodes total -- not worth refactoring.
        forward_path = lucas_kanade(
            video, start_frame, end_frame, start_points, mode, **lk_config
        )
        reverse_path = lucas_kanade(
            video, end_frame, start_frame, end_points, mode, **lk_config
        )
    assert forward_path.shape == reverse_path.shape
    n_frames, n_points = forward_path.shape[:2]

    epsilon = 0.01
    b = 2 * np.log(1 / epsilon - 1) / (end_frame - start_frame)
    c = (end_frame + start_frame) / 2
    x = np.r_[start_frame : end_frame + 1] if mode == "full" else target_frame
    sigmoid_forward = (1 / (1 + np.exp(b * (x - c))) - 0.5) / (1 - 2 * epsilon) + 0.5
    sigmoid_reverse = (1 / (1 + np.exp(-b * (x - c))) - 0.5) / (1 - 2 * epsilon) + 0.5

    s_f = sigmoid_forward[:, np.newaxis, np.newaxis]
    s_r = sigmoid_reverse[:, np.newaxis, np.newaxis]

    # Fuse the blend into two ufunc-with-out calls instead of three
    # implicit allocations (forward*sf, flip(reverse)*sr, sum). The
    # ``[::-1]`` view replaces ``np.flip(reverse_path, 0)`` (same
    # behaviour, no copy).
    rstc_path = np.empty_like(forward_path)
    tmp = np.empty_like(forward_path)
    np.multiply(forward_path, s_f, out=rstc_path)
    np.multiply(reverse_path[::-1], s_r, out=tmp)
    rstc_path += tmp
    return rstc_path
