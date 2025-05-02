from .__version__ import __version__

from .postprocess import lk_moving_average_filter

from datanavigator import VideoAnnotation

class VideoAnnotation(VideoAnnotation):
    """
    A subclass of VideoAnnotation that adds a method for applying a moving average filter to the annotations.
    """
    postprocess = lk_moving_average_filter
