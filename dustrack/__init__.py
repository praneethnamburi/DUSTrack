"""
DUSTrack: Semi-automated point tracking in ultrasound videos.


There are three main components in the DUSTrack workflow, namely the DUSTrack
GUI for initial manual annotation, the module for deep learning based tracking
using DLC project management, and the annotation class with LK-RSTC
post-processing.

:func:`~dustrack.dlcinterface.open`
    Unified entry point. Pass a video to start fresh, or a DLC project
    (or a video inside one, or its config.yaml) to resume in place.

:class:`~dustrack.dlcinterface.DUSTrack`
    Interactive GUI for manual annotation and refinement of tracking points.

:class:`~dustrack.dlcinterface.DLCProject`
    Interface for training and managing DeepLabCut pose estimation models.
    Requires the ``deeplabcut`` package to be installed.

:class:`~dustrack.pointtracking.VideoAnnotation`
    Annotation-data container with DeepLabCut HDF5 interop. Use directly
    for programmatic loads -- ``dustrack.VideoAnnotation(json_path, video).to_signals()``.

:func:`~dustrack.postprocess.lk_moving_average_filter`
    Lucas-Kanade optical flow algorithm with Reverse Sigmoid Tracking Correction
    for reducing frame-to-frame jitter in tracked points. In practice, access this via
    :meth:`dustrack.VideoAnnotation.postprocess` method.

Typical workflow:
    1. Create annotations using DUSTrack GUI
    2. Create DLC project from annotations
    3. Train and refine models iteratively
    4. Apply optical flow post-processing to reduce jitter

Example:
    >>> import dustrack
    >>>
    >>> # Step 1: start fresh on a video
    >>> tracker = dustrack.open('video.mp4', 'manual')
    >>> # Use the GUI to annotate points and save (writes video_annotations_manual.json),
    >>> # then click "Create DLC Project" + "Train DLC model" from the sidebar.
    >>>
    >>> # If you close the UI mid-workflow, resume with the same call:
    >>> tracker = dustrack.open('video.mp4')             # auto-detects the project
    >>> tracker = dustrack.open('path/to/project/')      # equivalent
    >>> tracker = dustrack.open('path/to/config.yaml')   # equivalent
    >>>
    >>> # DUSTrack also works without deeplabcut installed -- the LK-RSTC
    >>> # post-processing alone gets you usable tracks:
    >>> tracker = dustrack.open('video.mp4', 'manual')
    >>> tracker.process_with_lk()
    >>>
    >>> # Programmatic load (no UI):
    >>> va = dustrack.VideoAnnotation('video_annotations_pn.json', 'video.mp4')
    >>> signals = va.to_signals()

"""

from .__version__ import __version__

# Order matters: opticalflow has no in-package deps; pointtracking pulls
# from opticalflow; postprocess pulls VideoAnnotation from pointtracking;
# dlcinterface pulls pointtracking + postprocess.
from .opticalflow import lucas_kanade, lucas_kanade_rstc
from .pointtracking import VideoAnnotation, VideoAnnotations
from .postprocess import lk_moving_average_filter
from .dlcinterface import DUSTrack, DLCProject, open
from .convert import convert_to_mono
from .seed import (
    extract_snapshot_for_seeding,
    import_seed_bundle_into_project,
    inspect_seed_bundle,
)

# ``_dlc_patch`` exposes two independent runtime patches for DLC:
#
#   * ``patch_dlc()`` -- multi-threaded preprocessing + force-on
#     autocast + non_blocking H2D. NOT applied automatically. The
#     2026-05-20 multirun benchmark
#     (``S:/_corpus/dustrack/dlc_inference_bench_2026-05-20/``) showed
#     this patch is a net regression on DLC 3.0.0rc13/rc14 because
#     DLC's own refactor (``_batch_list`` + ``queue_length=4`` default)
#     captured most of the available win. On rc10 the patch is
#     approximately neutral on peak fps with a small high-bs floor
#     improvement. Call manually if running on rc10 and inference is
#     preprocessing-bound.
#
#   * ``patch_dlc_decoder()`` -- replaces DLC's
#     ``cv2.VideoCapture``-backed ``VideoReader`` with a dnav PyAV+TOC
#     reader so annotation, training-frame extraction, and inference
#     all go through one decode path. NOT applied automatically. Three
#     parity tests on 2026-05-20 confirmed semantic transparency on
#     the pia02 production format (1-frame-per-packet h264, common
#     for medical/scientific video):
#       - decoder pixel parity      : 50/50 bit-exact
#       - inference output parity   : 499/500 bit-exact predictions
#         (1 frame at 1.5e-5 px from pytorch nondeterminism, not the
#         decoder)
#       - training-extraction parity: 50/50 bit-exact
#     ...and the end-to-end perf cost is ~42% (154 -> 88 fps at bs=4)
#     on this codec because PyAV's ``to_ndarray(rgb24)`` swscale is
#     ~3.6 ms/frame vs cv2's ~0.5 ms/frame. Since the outputs are
#     bit-exact, defaulting to cv2 (the faster path) is the right
#     call. Use ``patch_dlc_decoder()`` as a power-user / debugging
#     toggle when you want unified-decoder semantics — e.g. to test a
#     codec class we haven't validated, or to A/B-check whether a
#     suspicious result is decoder-dependent.
#
#     Artefacts at S:/_corpus/dustrack/dlc_inference_bench_2026-05-20/
#     (parity_*.{py,json}, decoder_patch_bench_rc14.py, README.md).
from . import _dlc_patch as _dlc_patch  # noqa: F401

# Attach lk_moving_average_filter as VideoAnnotation's default postprocess
# hook. Done here (not in pointtracking.py) so pointtracking stays free of
# DLC-aware post-processing concerns; dustrack owns its DLC story.
# Pre-1.2.0a1 this lived as a one-line ``VideoAnnotation(dnav.VideoAnnotation)``
# subclass in dlcinterface.py -- that subclass bit
# ``isinstance(..., dustrack.VideoAnnotation)`` when ``lk_moving_average_filter``
# returned a parent-class instance (see feedback_isinstance_subclass_narrowing).
VideoAnnotation.postprocess = lk_moving_average_filter
