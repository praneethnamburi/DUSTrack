import numpy as np
from datanavigator import lucas_kanade, lucas_kanade_rstc
from decord import VideoReader
from matplotlib import pyplot as plt


def test_lucas_kanade_rstc():
    # test for lucas_kanade_rstc
    vname = r"C:\Dropbox (MIT)\File requests\_Praneeth\opr02_s004_us_b_006.mp4"
    video = VideoReader(vname)
    start_frame = 849
    end_frame = 904

    start_points = [[153.81, 195.34], [231.90, 209.27]]

    end_points = [[166.24, 166.74], [246.63, 181.54]]

    forward_path = lucas_kanade(
        video, start_frame, end_frame, start_points, mode="full"
    )

    reverse_path = lucas_kanade(video, end_frame, start_frame, end_points, mode="full")

    rstc_path = lucas_kanade_rstc(
        video, start_frame, end_frame, start_points, end_points
    )

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
