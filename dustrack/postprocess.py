"""
This module provides functionality for post-processing video annotations using the Lucas-Kanade 
optical flow algorithm with moving average and reverse sigmoid tracking correction (RSTC). 

The main function, `lk_moving_average_filter`, processes video annotations to smooth tracking data over a 
specified window size. It supports both manual and automatic annotations and saves the processed 
results for further use.

The Lucas-Kanade RSTC algorithm is re-implemented here to improve the performance of the postprocesing algorithm.
This implementation assumes that the caller implements a video "buffer" to avoid re-loading frames from disk.

Functions:
- lk_moving_average_filter: Main function to apply Lucas-Kanade moving average post-processing.
"""

from __future__ import annotations

import os
from typing import Union
from pathlib import Path

import dill
import cv2 as cv
import numpy as np
from tqdm import tqdm

from datanavigator import VideoAnnotation


def lk_moving_average_filter(tracked_points: Union[str, VideoAnnotation], video_name: str = None, window_size: float = 0.5) -> VideoAnnotation:
    """
    Post-process video annotations using the Lucas-Kanade optical flow algorithm with a moving average.

    Args:
        tracked_points (Union[str, VideoAnnotation]): Path to a tracked points file or a VideoAnnotation object.
        video_name (str, optional): Name of the video file. Required if `tracked_points` is a file path.
        window_size (float, optional): Time window (in seconds) for applying the moving average. Defaults to 0.5.

    Returns:
        VideoAnnotation: Processed video annotation with smoothed tracking data.
    """
    def gray(video_frame: int):
        """Convert a video frame to grayscale."""
        return cv.cvtColor(video_frame, cv.COLOR_BGR2GRAY)
    
    def lucas_kanade_2(
            frame_list: list,
            init_points: np.ndarray,
            **lk_config,
            ) -> np.ndarray:
        """
        Track points in a video using the Lucas-Kanade optical flow algorithm.

        Args:
            frame_list (list): List of video frames.
            init_points (np.ndarray): Initial points to track, with shape (n_points, 2).
            **lk_config: Additional configuration for the Lucas-Kanade algorithm.

        Returns:
            np.ndarray: Tracked points with shape (n_frames, n_points, 2).
        """
        init_points = np.array(init_points).astype(np.float32)
        if init_points.ndim == 1:
            init_points = init_points[np.newaxis, :]
        assert init_points.shape[-1] == 2
        if init_points.ndim == 2:
            init_points = init_points.reshape((init_points.shape[0], 1, 2))

        lk_config_default = dict(
                winSize=(45, 45), 
                maxLevel=2, 
                criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
            )
        lk_config = {**lk_config_default, **lk_config}

        n_frames = len(frame_list)
        n_points = init_points.shape[0]
        tracked_points = np.full((n_frames, n_points, 2), np.nan)
        tracked_points[0] = init_points[:, 0, :]
        for cnt, (fi, ff) in enumerate(zip(frame_list, frame_list[1:])):
            fp, _, _ = cv.calcOpticalFlowPyrLK(fi, ff, init_points, None, **lk_config)
            tracked_points[cnt+1] = fp[:, 0, :]
            init_points = fp
        return tracked_points

    def lucas_kanade_rstc_2(
            frame_list: list,
            start_points: np.ndarray, 
            end_points: np.ndarray,
            **lk_config
            ) -> np.ndarray:
        """
        Apply reverse sigmoid tracking correction (RSTC) to improve tracking accuracy.

        Args:
            frame_list (list): List of video frames.
            start_points (np.ndarray): Starting points for tracking, with shape (n_points, 2).
            end_points (np.ndarray): Ending points for tracking, with shape (n_points, 2).
            **lk_config: Additional configuration for the Lucas-Kanade algorithm.

        Returns:
            np.ndarray: Corrected tracking paths with shape (n_frames, n_points, 2).
        """
        forward_path = lucas_kanade_2(frame_list, start_points, **lk_config)
        reverse_path = lucas_kanade_2(frame_list[::-1], end_points, **lk_config)
        assert forward_path.shape == reverse_path.shape
        n_frames, n_points = forward_path.shape[:2]
        
        start_frame = 0
        end_frame = n_frames-1
        epsilon = 0.01
        b = 2*np.log(1/epsilon - 1)/(end_frame-start_frame)
        c = (end_frame + start_frame)/2
        x = np.r_[start_frame:end_frame+1]
        sigmoid_forward = ( 1/(1+np.exp(b*(x-c))) - 0.5)/(1-2*epsilon) + 0.5
        sigmoid_reverse = ( 1/(1+np.exp(-b*(x-c))) - 0.5)/(1-2*epsilon) + 0.5

        s_f = np.broadcast_to(sigmoid_forward[:, np.newaxis, np.newaxis], (n_frames, n_points, 2))
        s_r = np.broadcast_to(sigmoid_reverse[:, np.newaxis, np.newaxis], (n_frames, n_points, 2))
        
        rstc_path = forward_path*s_f + np.flip(reverse_path, 0)*s_r
        return rstc_path

    if isinstance(tracked_points, str):
        assert video_name is not None
        ann = VideoAnnotation(video_name, tracked_points)
    else:
        ann = tracked_points
    
    assert isinstance(ann, VideoAnnotation), "tracked_points must be a VideoAnnotation object or a path to a json or h5 file."

    postprocess_path = Path(ann.fname).parent

    source_ann = ann
    suffix = f"lkmovavg_{window_size:.3f}"
    fname_rawlk = str(postprocess_path / f"{ann.fstem}_{suffix}.pkl")

    label_list = source_ann.labels
    frame_list = list(range(source_ann.n_frames))
    if not os.path.exists(fname_rawlk):
        video = source_ann.video
        n_window_frames = round(window_size*source_ann.video.get_avg_fps())
        video_frame_buffer = [gray(f) for f in video[:n_window_frames-1].asnumpy()]

        rstc_paths = np.full((n_window_frames, source_ann.n_frames, len(label_list), 2), np.nan)
        for cnt, (start_frame, end_frame) in tqdm(enumerate(zip(frame_list, frame_list[n_window_frames-1:]))):
            video_frame_buffer.append(gray(video[end_frame].asnumpy()))
            start_points = [source_ann.data[label][start_frame] for label in label_list]
            end_points = [source_ann.data[label][end_frame] for label in label_list]
            rstc_path = lucas_kanade_rstc_2(video_frame_buffer, start_points, end_points)
            rstc_paths[cnt%n_window_frames, start_frame:end_frame+1, :, :] = rstc_path
            video_frame_buffer.pop(0)
        # save the paths
        with open(fname_rawlk, "wb") as f:
            dill.dump(rstc_paths, f)

    fname_processed = str(postprocess_path / f"{ann.fstem}_{suffix}.json")
    if not os.path.exists(fname_processed):
        with open(fname_rawlk, "rb") as f:
            rstc_paths = dill.load(f)
        rstc_paths_avg = np.nanmean(rstc_paths, axis=0)
        ann_processed = VideoAnnotation(fname_processed, source_ann.video.fname)
        ann_processed.data = {label: {} for label in label_list}
        for label_cnt, label in enumerate(label_list):
            for frame_num in frame_list:
                ann_processed.data[label][frame_num] = rstc_paths_avg[frame_num, label_cnt, :]
        ann_processed.save()
    
    ann_processed = VideoAnnotation(fname_processed, source_ann.video.fname)
    return ann_processed

# Add a function to combine manual and automatic annotations
