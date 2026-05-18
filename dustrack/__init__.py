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

:func:`~dustrack.postprocess.lk_moving_average_filter`
    Lucas-Kanade optical flow algorithm with Reverse Sigmoid Tracking Correction
    for reducing frame-to-frame jitter in tracked points. In practice, access this via
    :meth:`dustrack.dlcinterface.VideoAnnotation.postprocess` method.

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

"""

from .__version__ import __version__

from .postprocess import lk_moving_average_filter
from .dlcinterface import DUSTrack, DLCProject, VideoAnnotation, open
