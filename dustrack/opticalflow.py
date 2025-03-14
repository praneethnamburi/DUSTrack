"""
This module provides functions to track points in a video using the Lucas-Kanade
algorithm and apply the reverse sigmoid tracking correction (RSTC) as described
in Magana-Salgado, et al., 2023.

Magana-Salgado, U., Namburi, P., Feigin-Almon, M., Pallares-Lopez, R., & Anthony, B (2023)
A comparison of point-tracking algorithms in ultrasound videos from the upper limb. 
BioMedical Engineering OnLine, 22(1), 52.
https://doi-org.libproxy.mit.edu/10.1186/s12938-023-01105-y
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import cv2 as cv
import numpy as np
from decord import VideoReader

from . import utils


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

    def gray(video: VideoReader, frame_num: int) -> np.ndarray:
        """Convert a frame to grayscale.

        Args:
            video (VideoReader): Video object.
            frame_num (int): Frame number to convert.

        Returns:
            np.ndarray: Grayscale frame.
        """
        return cv.cvtColor(video[frame_num].asnumpy(), cv.COLOR_BGR2GRAY)

    # input validation
    if isinstance(video, (str, Path)):
        assert os.path.exists(video)
        video = VideoReader(video)

    direction = "forward" if end_frame > start_frame else "back"

    init_points = np.array(init_points).astype(np.float32)
    if init_points.ndim == 1:
        init_points = init_points[np.newaxis, :]
    assert init_points.shape[-1] == 2
    if init_points.ndim == 2:
        init_points = init_points.reshape((init_points.shape[0], 1, 2))

    assert mode in ("direct", "full")

    lk_config_default = dict(
        winSize=(45, 45),
        maxLevel=2,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    lk_config = {**lk_config_default, **lk_config}

    # compute locations at end_frame based on locations at start_frame
    fi = gray(video, start_frame)
    if mode == "direct":
        ff = gray(video, end_frame)
        fp, _, _ = cv.calcOpticalFlowPyrLK(fi, ff, init_points, None, **lk_config)
        tracked_points = fp[:, 0, :][np.newaxis, :, :]
        return tracked_points

    # compute locations at every frame iteratively from start_frame to end_frame
    assert mode == "full"  # for readability
    n_frames = np.abs(end_frame - start_frame) + 1
    if direction == "forward":
        frame_numbers = np.arange(start_frame + 1, end_frame + 1, 1)
    else:
        frame_numbers = np.arange(start_frame - 1, end_frame - 1, -1)

    n_points = init_points.shape[0]  # number of tracked points

    tracked_points = np.full((n_frames, n_points, 2), np.nan)
    frame_count = 0
    tracked_points[frame_count] = init_points[:, 0, :]
    for frame_num in frame_numbers:
        frame_count = frame_count + 1
        ff = gray(video, frame_num)
        fp, _, _ = cv.calcOpticalFlowPyrLK(fi, ff, init_points, None, **lk_config)
        tracked_points[frame_count] = fp[:, 0, :]
        fi = ff
        init_points = fp
    return tracked_points


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

    s_f = np.broadcast_to(
        sigmoid_forward[:, np.newaxis, np.newaxis], (n_frames, n_points, 2)
    )
    s_r = np.broadcast_to(
        sigmoid_reverse[:, np.newaxis, np.newaxis], (n_frames, n_points, 2)
    )

    rstc_path = forward_path * s_f + np.flip(reverse_path, 0) * s_r
    return rstc_path
