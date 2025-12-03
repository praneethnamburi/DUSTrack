"""
DUSTrack: Semi-automated point tracking in ultrasound videos.


There are three main components in the DUSTrack workflow, namely the DUSTrack
GUI for initial manual annotation, the module for deep learning based tracking
using DLC project management, and the annotation class with LK-RSTC
post-processing.

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
    >>> # Interactive annotation
    >>> tracker = DUSTrack('video.mp4', "pn") # pn is the name of the annotation layer"
    >>> # Use the GUI to annotate points and save the annotations, resulting in a .json file.
    >>> # Create and train DLC project
    >>> dlc_proj = tracker.create_dlc_project()
    >>> dlc_proj.process()
    >>> 
    >>> # Post-process to reduce jitter
    >>> tracker.process_with_lk() # postprocess the current annotation layer

"""

from .__version__ import __version__

from .postprocess import lk_moving_average_filter
from .dlcinterface import DUSTrack, DLCProject, VideoAnnotation
