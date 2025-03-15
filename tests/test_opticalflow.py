import numpy as np
from datanavigator import lucas_kanade, lucas_kanade_rstc
from decord import VideoReader
from matplotlib import pyplot as plt
from datanavigator import get_example_video


def test_lucas_kanade_rstc():
    vname = get_example_video()
    video = VideoReader(vname)
    start_frame = 35
    end_frame = 50

    start_points = [[153.81, 195.34], [231.90, 209.27]]
    end_points = [[166.24, 166.74], [246.63, 181.54]]

    forward_path = lucas_kanade(video, start_frame, end_frame, start_points, mode="full")
    reverse_path = lucas_kanade(video, end_frame, start_frame, end_points, mode="full")

    rstc_path = lucas_kanade_rstc(video, start_frame, end_frame, start_points, end_points)

    plt.figure()
    plt.plot(
        np.array(
            [
                forward_path[:, 0, 0],
                np.flip(reverse_path, 0)[:, 0, 0],
                rstc_path[:, 0, 0],
            ]
        ).T
    )
    plt.legend(["Forward", "Reverse", "RSTC"])
    plt.show(block=False)

    n_frames = end_frame - start_frame + 1
    n_points = len(start_points)
    assert forward_path.shape == (n_frames, n_points, 2)
    
    direct_prediction = lucas_kanade(video, end_frame, start_frame, end_points, mode="direct")
    assert direct_prediction.shape == (1, n_points, 2)

    return forward_path, reverse_path, rstc_path
