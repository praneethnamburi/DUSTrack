"""
Post-processing module for video annotations using optical flow algorithms,
typically applied on the output of deep learing models when the results have
frame to frame jitter.

The primary algorithm is Lucas-Kanade optical flow with Reverse Sigmoid Tracking
Correction (RSTC) applied in a "transposed" moving average window.

Key Features:
    - Lucas-Kanade optical flow tracking for point trajectories
    - Reverse Sigmoid Tracking Correction (RSTC) guarantees the locations of the start and end points
    - Moving average filter over time windows
    - Parallel processing support for faster computation
    - Video frame buffering to minimize disk I/O

Typical Usage:
    >>> from dustrack.postprocess import lk_moving_average_filter
    >>> from dustrack import VideoAnnotation
    >>>
    >>> # Load annotations
    >>> ann = VideoAnnotation('video_annotations.json', 'video.mp4')
    >>>
    >>> # Apply post-processing with 0.5 second window
    >>> ann_smooth = lk_moving_average_filter(ann, window_size=0.5)

In practice, access the lk_moving_average_filter method via :class:`~dustrack.VideoAnnotation`:
    >>> from dustrack import VideoAnnotation
    >>> ann = VideoAnnotation('video_dlc_predictions.h5', 'video.mp4')
    >>> ann_rstc = ann.postprocess()


Algorithm Overview:
    The RSTC algorithm tracks points both forward and backward in time, then
    combines the two trajectories using sigmoid weights that emphasize the
    forward path at the start and the backward path at the end. This reduces
    drift accumulation compared to pure forward tracking.
    
    The moving average applies RSTC over overlapping time windows and averages
    the results, further reducing noise while maintaining temporal coherence.

Performance:
    Processing speed depends on:
    - Video resolution and frame rate
    - Number of tracked points
    - Window size (larger = more computation)
    - Parallel processing (use_parallel=True recommended)
    
    Typical performance: ~5-10 fps for 1080p video with 10 points and 0.5s window.

Reference for the LK-RSTC Algorithm:
    Magana-Salgado, U., Namburi, P., Feigin-Almon, M., Pallares-Lopez, R., &
    Anthony, B. (2023). A comparison of point-tracking algorithms in ultrasound
    videos from the upper limb. BioMedical Engineering OnLine, 22(1), 52.
"""

from __future__ import annotations

import os
from typing import Union
from pathlib import Path
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import dill
import cv2 as cv
import numpy as np
from tqdm import tqdm

from .opticalflow import _lk_track_frames
from .pointtracking import VideoAnnotation


def gray(video_frame: np.ndarray) -> np.ndarray:
    """
    Convert a color video frame to grayscale.

    The Lucas-Kanade algorithm operates on grayscale images. Input
    frames come from dnav ``VideoReader`` (PyAV+TOC) which returns
    RGB-ordered ndarrays, so we use ``COLOR_RGB2GRAY`` for the
    mathematically correct ITU-R BT.601 luminance:
    ``Y = 0.299 R + 0.587 G + 0.114 B``.

    Args:
        video_frame (np.ndarray): Input frame in RGB format (H, W, 3).

    Returns:
        np.ndarray: Grayscale frame (H, W).

    Pre-2026-05-21 this used ``COLOR_BGR2GRAY`` on the same RGB input
    -- functionally still grayscale, but it swapped the R and B
    coefficients, computing ``Y = 0.114 R + 0.587 G + 0.299 B``.
    Self-consistent within ``lk_moving_average_filter`` (both
    template and search frames went through the same wrong
    conversion, so LK gradient + iteration math worked fine), but it
    diverged from :py:mod:`dustrack.opticalflow` which always used
    ``COLOR_RGB2GRAY``. Unified to RGB2GRAY in the 1.2.0a2 perf pass.
    """
    return cv.cvtColor(video_frame, cv.COLOR_RGB2GRAY)


def lucas_kanade_2(frame_list: list, init_points: np.ndarray, **lk_config) -> np.ndarray:
    """
    Track points through a sequence of frames using Lucas-Kanade optical flow.
    
    This implementation uses OpenCV's pyramidal Lucas-Kanade method to track
    points from the initial position through all subsequent frames. Tracking
    is performed in a single forward pass.
    
    Args:
        frame_list (list): List of grayscale video frames (each frame is np.ndarray).
        init_points (np.ndarray): Initial point positions, shape (n_points, 2) where
            each row is [x, y] in pixel coordinates.
        **lk_config: Configuration parameters for cv.calcOpticalFlowPyrLK():
            - winSize (tuple): Window size for search. Default: (45, 45)
            - maxLevel (int): Pyramid levels. Default: 2
            - criteria (tuple): Termination criteria. Default: (EPS|COUNT, 10, 0.03)

    Returns:
        np.ndarray: Tracked point trajectories, shape (n_frames, n_points, 2).
            Missing/lost points are filled with NaN.
    
    Note:
        Larger winSize provides more robustness but slower computation.
        Higher maxLevel helps track larger motions but may lose precision.
    
    Example:
        >>> frames = [gray(f) for f in video_frames]
        >>> initial_pts = np.array([[100, 200], [150, 250]])
        >>> trajectories = lucas_kanade_2(frames, initial_pts)
        >>> # trajectories.shape = (len(frames), 2, 2)
    """
    return _lk_track_frames(frame_list, init_points, **lk_config)


def compute_sigmoid_weights(n_frames: int, epsilon: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute forward and reverse sigmoid weights for RSTC blending.
    
    The weights control how forward and backward tracking paths are combined:
    - At the start: forward weight ≈ 1, reverse weight ≈ 0
    - At the end: forward weight ≈ 0, reverse weight ≈ 1
    - In the middle: both weights ≈ 0.5
    
    Args:
        n_frames (int): Number of frames.
        epsilon (float, optional): Small value to control sigmoid scaling. Defaults to 0.01.

    Returns:
        tuple[np.ndarray, np.ndarray]: Two arrays of shape (n_frames,):
            - sigmoid_forward: Decreasing weights from ~1 to ~0
            - sigmoid_reverse: Increasing weights from ~0 to ~1
            
            Forward and reverse weights sum to 1.0 at each frame.
    
    Mathematical Form:
        sigmoid(x) = 1 / (1 + exp(b * (x - c)))
        where b controls steepness and c is the center point.
    """
    start_frame = 0
    end_frame = n_frames - 1
    b = 2 * np.log(1 / epsilon - 1) / (end_frame - start_frame)
    c = (end_frame + start_frame) / 2
    x = np.arange(start_frame, end_frame + 1)
    sigmoid_forward = (1 / (1 + np.exp(b * (x - c))) - 0.5) / (1 - 2 * epsilon) + 0.5
    sigmoid_reverse = (1 / (1 + np.exp(-b * (x - c))) - 0.5) / (1 - 2 * epsilon) + 0.5
    return sigmoid_forward, sigmoid_reverse


def lucas_kanade_rstc_2(
    frame_list: list,
    start_points: np.ndarray,
    end_points: np.ndarray,
    sigmoid_forward: np.ndarray = None,
    sigmoid_reverse: np.ndarray = None,
    **lk_config,
) -> np.ndarray:
    """
    Apply Reverse Sigmoid Tracking Correction (RSTC) for robust point tracking.
    
    RSTC combines forward tracking (from start_points) and backward tracking
    (from end_points) using sigmoid weights. This reduces error accumulation
    compared to pure forward tracking, especially over long sequences.
    
    Algorithm:
        1. Track forward from start_points through all frames
        2. Track backward from end_points through all frames (in reverse)
        3. Blend the two paths using sigmoid weights:
           result = forward_path * sigmoid_forward + reverse_path * sigmoid_reverse
    
    Args:
        frame_list (list): List of grayscale video frames.
        start_points (np.ndarray): Initial positions at first frame, shape (n_points, 2).
        end_points (np.ndarray): Final positions at last frame, shape (n_points, 2).
            Should match start_points for consistency.
        sigmoid_forward (np.ndarray, optional): Forward weights. If None, computed automatically.
        sigmoid_reverse (np.ndarray, optional): Reverse weights. If None, computed automatically.
        **lk_config: Lucas-Kanade configuration (passed to lucas_kanade_2).

    Returns:
        np.ndarray: Blended tracking paths, shape (n_frames, n_points, 2).
    
    Note:
        Precomputing sigmoid weights and reusing them across windows significantly
        improves performance when processing multiple sequences of the same length.
    
    Example:
        >>> frames = [gray(f) for f in video_frames[10:20]]
        >>> start = np.array([[100, 200]])
        >>> end = np.array([[105, 205]])  # Approximate end position
        >>> trajectory = lucas_kanade_rstc_2(frames, start, end)
    """
    forward_path = lucas_kanade_2(frame_list, start_points, **lk_config)
    reverse_path = lucas_kanade_2(frame_list[::-1], end_points, **lk_config)
    assert forward_path.shape == reverse_path.shape
    n_frames, n_points = forward_path.shape[:2]

    # Compute sigmoid weights if not provided
    if sigmoid_forward is None or sigmoid_reverse is None:
        sigmoid_forward, sigmoid_reverse = compute_sigmoid_weights(n_frames)

    s_f = sigmoid_forward[:, np.newaxis, np.newaxis]
    s_r = sigmoid_reverse[:, np.newaxis, np.newaxis]

    # Fuse the blend into two ufunc-with-out calls. ``[::-1]`` view
    # replaces ``np.flip(reverse_path, 0)`` (no copy).
    rstc_path = np.empty_like(forward_path)
    tmp = np.empty_like(forward_path)
    np.multiply(forward_path, s_f, out=rstc_path)
    np.multiply(reverse_path[::-1], s_r, out=tmp)
    rstc_path += tmp
    return rstc_path


def process_window(video_frame_buffer, start_points, end_points, sigmoid_forward, sigmoid_reverse):
    """
    Process a single time window using RSTC.
    
    This is a helper function designed for parallel execution by ThreadPoolExecutor.
    It wraps lucas_kanade_rstc_2 with precomputed sigmoid weights.
    
    Args:
        video_frame_buffer: Deque or list of grayscale frames for this window.
        start_points (np.ndarray): Point positions at window start.
        end_points (np.ndarray): Point positions at window end.
        sigmoid_forward (np.ndarray): Precomputed forward weights.
        sigmoid_reverse (np.ndarray): Precomputed reverse weights.
    
    Returns:
        np.ndarray: Tracked trajectories for this window, shape (window_frames, n_points, 2).
    """
    return lucas_kanade_rstc_2(
        list(video_frame_buffer),
        start_points,
        end_points,
        sigmoid_forward=sigmoid_forward,
        sigmoid_reverse=sigmoid_reverse,
    )


def lk_moving_average_filter(
    tracked_points: Union[str, VideoAnnotation],
    video_name: str = None,
    window_size: float = 0.5,
    use_parallel: bool = True,
    save_raw: bool = True,
) -> VideoAnnotation:
    """
    Apply Lucas-Kanade moving average filter to smooth tracking annotations.

    This is the main post-processing function. It applies RSTC over overlapping
    time windows and averages the results to produce smooth, jitter-free trajectories.

    Workflow:
        1. Load video and annotations
        2. Slide a time window through the video
        3. For each window:
           - Apply RSTC to get smoothed trajectory
           - Store results indexed by window position
        4. Average all trajectories that cover each frame
        5. Save smoothed annotations

    Args:
        tracked_points (Union[str, VideoAnnotation]): Either:
            - Path to annotation JSON file
            - VideoAnnotation object with tracking data
        video_name (str, optional): Video path. Defaults to None; required when
            ``tracked_points`` is a file path (asserted at call time). Ignored
            when ``tracked_points`` is already a ``VideoAnnotation``.
        window_size (float, optional): Time window in seconds for moving average.
            Larger windows provide more smoothing but increase computation.
            Typical range: 0.1 to 1.0 seconds. Defaults to 0.5.
        use_parallel (bool, optional): Use ThreadPoolExecutor for parallel processing.
            Significantly faster for longer videos. Defaults to True.
        save_raw (bool, optional): If True (default), allocate the full
            ``(n_window_frames, n_total_frames, n_labels, 2)`` per-window
            RSTC array and dill-pickle it to
            ``{stem}_lkmovavg_{window_size}.pkl`` alongside the averaged
            JSON. Per-window data is consumed downstream by
            ``pn-projects/wobble`` and ``gaitmusic`` (their ``.rawlk``
            property → ``lk_gradients`` velocity estimation), so keep
            True for those flows. If False, stream a running
            ``(n_total_frames, n_labels, 2)`` sum and per-cell count
            instead, divide at the end, and skip the .pkl write. Cuts
            peak memory from ``W*N*L*16`` bytes to ``N*L*12`` bytes
            (~10x on typical W=15, L=4 cases). On the example-video
            bench wall time is essentially unchanged (post-processing
            is <0.3% of total) but on real-world long videos
            (N=36715+) the .pkl round-trip alone is a couple of
            seconds. FP summation order changes between modes so
            ``save_raw=False`` output is numerically close
            (``np.allclose`` with ``atol~1e-10``) rather than
            bit-exact vs ``save_raw=True``.

    Returns:
        VideoAnnotation: New annotation object with smoothed trajectories.
            Saved to {original_stem}_lkmovavg_{window_size}.json

    Output Files:
        - {stem}_lkmovavg_{window_size}.pkl: Per-window RSTC results
          (only when ``save_raw=True``; consumed by ``.rawlk`` downstream).
        - {stem}_lkmovavg_{window_size}.json: Averaged smoothed annotations
          (always written).

    Performance Tips:
        - Use use_parallel=True for videos longer than a few seconds
        - Smaller window_size is faster but provides less smoothing
        - Results are cached: the .json is the primary cache, and when
          ``save_raw=True`` the .pkl is reused on subsequent calls
          regardless of the ``save_raw`` flag (a pre-existing .pkl is
          always read if the .json is missing). Delete either to
          recompute the corresponding stage.

    Example:
        >>> # Post-process manual annotations
        >>> ann = VideoAnnotation('video_annotations.json', 'video.mp4')
        >>> ann_smooth = lk_moving_average_filter(ann, window_size=0.5)
        >>>
        >>> # Post-process DLC predictions with larger window
        >>> ann_dlc = VideoAnnotation('video_dlc_predictions.h5', 'video.mp4')
        >>> ann_smooth = lk_moving_average_filter(ann_dlc, window_size=1.0)
        >>>
        >>> # Non-parallel processing for short clips
        >>> ann_smooth = lk_moving_average_filter(ann, window_size=0.3, use_parallel=False)
        >>>
        >>> # GUI-style: skip the .pkl sidecar (memory + I/O savings)
        >>> ann_smooth = lk_moving_average_filter(ann, save_raw=False)

        In practice, the ``postprocess`` shortcut on a :class:`~dustrack.VideoAnnotation`
        is equivalent:
        >>> from dustrack import VideoAnnotation
        >>> ann = VideoAnnotation('video_dlc_predictions.h5', 'video.mp4')
        >>> ann_rstc = ann.postprocess()

    Raises:
        AssertionError: If video_name is not provided when tracked_points is a file path.
        AssertionError: If tracked_points is not a VideoAnnotation or valid file path.

    Note:
        The algorithm uses a deque-based frame buffer to minimize memory usage
        and avoid repeated video decoding. Frames are read once and discarded
        after the window moves past them.
    """
    if isinstance(tracked_points, str):
        assert video_name is not None, "video_name must be provided if tracked_points is a file path."
        ann = VideoAnnotation(video_name, tracked_points)
    else:
        ann = tracked_points

    assert isinstance(ann, VideoAnnotation), "tracked_points must be a VideoAnnotation object or a path to a json or h5 file."

    postprocess_path = Path(ann.fname).parent
    suffix = f"lkmovavg_{window_size:.3f}"
    fname_rawlk = str(postprocess_path / f"{ann.fstem}_{suffix}.pkl")

    layer_name = getattr(ann, "name", ann.fstem)
    # Only smooth labels with data at every frame in the video. A
    # freshly-clicked refinement label only has one frame's worth of
    # data; iterating ``ann.data[label][start_frame]`` would KeyError
    # at start_frame=0 (or anywhere else the label hasn't been
    # populated). Skip + warn rather than crash; the user can still
    # see the manual point in the source layer.
    all_labels = ann.labels
    frame_list = list(range(ann.n_frames))
    label_list = [
        label for label in all_labels
        if all(frame in ann.data[label] for frame in frame_list)
    ]
    skipped = [label for label in all_labels if label not in label_list]
    if skipped:
        skip_summary = ", ".join(
            f"{label!r} ({len(ann.data[label])}/{ann.n_frames} frames)"
            for label in skipped
        )
        print(
            f"[reduce_jitter] layer={layer_name!r}: skipping "
            f"{len(skipped)}/{len(all_labels)} label(s) with incomplete "
            f"frame coverage: {skip_summary}"
        )
    if not label_list:
        raise ValueError(
            f"Layer {layer_name!r} has no labels with complete frame "
            f"coverage ({ann.n_frames} frames). Nothing to smooth. "
            f"Per-label coverage: " + ", ".join(
                f"{label!r}={len(ann.data[label])}" for label in all_labels
            )
        )
    fname_processed = str(postprocess_path / f"{ann.fstem}_{suffix}.json")

    # Cache short-circuits.
    # 1. If the averaged JSON already exists, we're done -- just reload it.
    # 2. Else if the per-window .pkl exists (from any prior run, regardless
    #    of this call's save_raw), re-use it to compute the average without
    #    re-running LK.
    # 3. Else: full computation. ``save_raw`` controls whether we accumulate
    #    into the (W, N, L, 2) per-window array (preserved + dill-pickled,
    #    nanmean'd at the end) or into running sum + count buffers (no .pkl,
    #    final mean = sum / count).
    rstc_paths_avg = None  # populated by one of the three branches below

    if os.path.exists(fname_processed):
        # Branch 1: nothing to do; the saved annotation is the truth.
        pass
    elif os.path.exists(fname_rawlk):
        # Branch 2: reuse cached per-window data even if save_raw=False
        # (a previous save_raw=True run left it on disk -- cheaper to
        # average than to recompute).
        with open(fname_rawlk, "rb") as f:
            rstc_paths = dill.load(f)
        rstc_paths_avg = np.nanmean(rstc_paths, axis=0)
    else:
        # Branch 3: full computation.
        video = ann.video
        n_window_frames = round(window_size * video.get_avg_fps())
        video_frame_buffer = deque([gray(f) for f in video[:n_window_frames - 1].asnumpy()], maxlen=n_window_frames)

        # Precompute sigmoid weights for the given window size
        sigmoid_forward, sigmoid_reverse = compute_sigmoid_weights(n_window_frames)

        # ``save_raw`` chooses the accumulator. Both branches expose a
        # ``_record(cnt, start_frame, rstc_path)`` callable so the
        # parallel + sequential loops below stay uniform.
        if save_raw:
            rstc_paths = np.full((n_window_frames, ann.n_frames, len(label_list), 2), np.nan)
            def _record(cnt, start_frame, rstc_path):
                n = rstc_path.shape[0]
                rstc_paths[cnt % n_window_frames, start_frame:start_frame + n, :, :] = rstc_path
        else:
            # Streaming sum + count, no per-window slab. ``count_paths``
            # is per (frame, label) -- the same window touches all
            # labels at each frame, so collapsing the trailing 2-axis
            # is safe.
            sum_paths = np.zeros((ann.n_frames, len(label_list), 2), dtype=np.float64)
            count_paths = np.zeros((ann.n_frames, len(label_list)), dtype=np.int32)
            def _record(cnt, start_frame, rstc_path):
                n = rstc_path.shape[0]
                sum_paths[start_frame:start_frame + n] += rstc_path
                count_paths[start_frame:start_frame + n] += 1

        if use_parallel:
            # Bounded-inflight ThreadPoolExecutor loop. Pre-2026-05-21
            # this submitted all N futures up front then collected
            # in submission order with two separate tqdm bars
            # ("Submitting jobs" / "Processing results"); the
            # "Submitting" bar filled almost instantly on long videos
            # then "Processing" sat at 0 for ages, making the Qt overlay
            # phase label uninformative. Now: a single tqdm bar
            # ("Processing tracking jobs") with at most ``max_inflight``
            # futures pending, decode + submit interleaved with
            # collection, completions written into the circular buffer
            # by the cnt index they own (slot = cnt % n_window_frames),
            # so out-of-order completion is safe.
            #
            # Pin OpenCV's internal thread count to 1 for the duration
            # of the parallel block. Without this, every worker's
            # ``cv.calcOpticalFlowPyrLK`` call spawns up to ``cpu_count``
            # internal threads of its own; with a ``cpu_count``-sized
            # Python worker pool that's ``cpu_count**2`` thread-slots
            # fighting for ``cpu_count`` cores -- pure oversubscription.
            # Measured (24-core, 300-frame bench, save_raw=False):
            # cv=24 + 28 workers = 6.49 s, cv=1 + 24 workers = 5.73 s
            # (~12% faster). Beyond that there's a hard plateau at
            # ~5.7 s independent of pool size (>= 8 workers) and pool
            # type (a ProcessPoolExecutor + shared-memory frames
            # variant hits the same ceiling -- see
            # ``tests/qt_learning/_movavg_gil_probe.py``), so the GIL
            # is NOT the bottleneck. The plateau is likely OpenCV's
            # heap allocator contending on pyramid memory:
            # ``calcOpticalFlowPyrLK`` internally builds two ~7 MB
            # pyramids per pair, ~117 GB of allocation traffic across
            # the 8008 LK calls of this bench. Going lower would need
            # either a different system allocator (mimalloc/jemalloc)
            # or a custom LK that pools pyramid memory across calls;
            # neither is in scope.
            pair_iter = enumerate(zip(frame_list, frame_list[n_window_frames - 1:]))
            n_windows = len(frame_list) - n_window_frames + 1
            saved_cv_threads = cv.getNumThreads()
            cv.setNumThreads(1)
            try:
                with ThreadPoolExecutor() as executor:
                    max_inflight = max(2 * (executor._max_workers or 4), 8)
                    inflight: dict = {}
                    n_done = 0
                    with tqdm(total=n_windows, desc="Processing tracking jobs") as pbar:
                        while n_done < n_windows:
                            while len(inflight) < max_inflight:
                                try:
                                    cnt, (start_frame, end_frame) = next(pair_iter)
                                except StopIteration:
                                    break
                                video_frame_buffer.append(gray(video[end_frame].asnumpy()))
                                start_points = [ann.data[label][start_frame] for label in label_list]
                                end_points = [ann.data[label][end_frame] for label in label_list]
                                fut = executor.submit(
                                    process_window,
                                    video_frame_buffer.copy(),
                                    start_points,
                                    end_points,
                                    sigmoid_forward,
                                    sigmoid_reverse,
                                )
                                inflight[fut] = cnt
                            if not inflight:
                                break
                            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                            for fut in done:
                                cnt = inflight.pop(fut)
                                rstc_path = fut.result()
                                _record(cnt, frame_list[cnt], rstc_path)
                                n_done += 1
                                pbar.update(1)
            finally:
                cv.setNumThreads(saved_cv_threads)
        else:
            # Sequential processing
            with tqdm(total=len(frame_list) - n_window_frames + 1, desc="Processing sequentially") as pbar:
                for cnt, (start_frame, end_frame) in enumerate(zip(frame_list, frame_list[n_window_frames - 1:])):
                    video_frame_buffer.append(gray(video[end_frame].asnumpy()))
                    start_points = [ann.data[label][start_frame] for label in label_list]
                    end_points = [ann.data[label][end_frame] for label in label_list]

                    # Process the window sequentially
                    rstc_path = process_window(
                        video_frame_buffer.copy(),
                        start_points,
                        end_points,
                        sigmoid_forward,
                        sigmoid_reverse,
                    )
                    _record(cnt, start_frame, rstc_path)
                    pbar.update(1)  # Update the progress bar as results are processed

        # Finalise + persist sidecars.
        if save_raw:
            with open(fname_rawlk, "wb") as f:
                dill.dump(rstc_paths, f)
            rstc_paths_avg = np.nanmean(rstc_paths, axis=0)
        else:
            # Mean = sum / count. ``count`` is 0 only at frames that no
            # window touched -- shouldn't happen since the window iter
            # covers [0, n_frames-1], but guard with errstate so a
            # surprise 0 produces NaN rather than a runtime warning.
            with np.errstate(invalid="ignore", divide="ignore"):
                rstc_paths_avg = sum_paths / count_paths[..., np.newaxis]

    if not os.path.exists(fname_processed):
        assert rstc_paths_avg is not None, (
            "rstc_paths_avg was not populated; the cache-branch logic above "
            "should ensure this is unreachable."
        )
        ann_processed = VideoAnnotation(fname_processed, ann.video.fname)
        # Build per-label dicts via comprehension instead of the
        # nested ``data[label][frame] = ...`` loop. Same end state,
        # one fewer name-lookup chain per write.
        ann_processed.data = {
            label: {frame_num: rstc_paths_avg[frame_num, label_cnt, :]
                    for frame_num in frame_list}
            for label_cnt, label in enumerate(label_list)
        }
        ann_processed.save()

    return VideoAnnotation(fname_processed, ann.video.fname)
