

from utils import get_video_info, show_frame_with_bbox, crop_mp4_ffmpeg_python
import os



if __name__ == "__main__":
    video_folder = r"C:\Users\haowe\OneDrive\Desktop\MIT\PianoProject\Code\DUSTrack\pia02_workflow\fingor_torque"
    video_name = "s061.MP4"
    x = 1300
    y = 100
    w = 600
    h = 600

    video_path = os.path.join(video_folder, video_name)
    get_video_info(video_path)
    show_frame_with_bbox(video_path, frame_idx=100, x=x, y=y, w=w, h=h)