import os
import sys
from pathlib import Path

import cv2

video_path = r"\\192.168.1.104\home\piano\data\010\overhead camera\fx30_2_0003.MP4"

# check if the video exists
if not os.path.exists(video_path):
    print(f"Video file {video_path} does not exist")
    exit()


# extract the last frame of the video and visualize it
cap = cv2.VideoCapture(video_path)
ok, frame = cap.read()
if ok:
    cv2.imshow("Last frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"Failed to read frame from {video_path}")
    exit()




