import numpy as np
import pytest
from datanavigator import VideoReader, get_example_video
from dustrack import lucas_kanade, lucas_kanade_rstc


pytestmark = pytest.mark.slow


def test_lucas_kanade_rstc():
    vname = get_example_video()
    video = VideoReader(vname)
    start_frame = 35
    end_frame = 50

    start_points = [[153.81, 195.34], [231.90, 209.27]]
    end_points = [[166.24, 166.74], [246.63, 181.54]]

    forward_path = lucas_kanade(
        video, start_frame, end_frame, start_points, mode="full"
    )
    reverse_path = lucas_kanade(video, end_frame, start_frame, end_points, mode="full")

    rstc_path = lucas_kanade_rstc(
        video, start_frame, end_frame, start_points, end_points
    )

    n_frames = end_frame - start_frame + 1
    n_points = len(start_points)
    assert forward_path.shape == (n_frames, n_points, 2)

    direct_prediction = lucas_kanade(
        video, end_frame, start_frame, end_points, mode="direct"
    )
    assert direct_prediction.shape == (1, n_points, 2)
    assert reverse_path.shape == (n_frames, n_points, 2)
    assert rstc_path.shape == (n_frames, n_points, 2)
