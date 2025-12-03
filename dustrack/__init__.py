"""
DUSTrack: Semi-automated point tracking in ultrasound videos.

Main components:
- DUSTrack: Interactive annotation GUI
- DLCProject: DeepLabCut model training interface (requires deeplabcut)
- VideoAnnotation: Annotation class with LK-RSTC post-processing
- lk_moving_average_filter: Optical flow smoothing algorithm
"""

from .__version__ import __version__

from .postprocess import lk_moving_average_filter
from .dlcinterface import DUSTrack
from . import dlcinterface
if dlcinterface.HAS_DLC:
    from .dlcinterface import DLCProject
