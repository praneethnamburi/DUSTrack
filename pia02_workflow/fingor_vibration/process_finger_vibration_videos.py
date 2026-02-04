import os
import json

from utils import get_video_info, show_frame_with_bbox, crop_mp4_ffmpeg_python

if __name__ == "__main__":

    original_video_folder = r"\\192.168.1.104\home\piano\data\finger_vibration\original_videos"
    cropped_video_folder = r"\\192.168.1.104\home\piano\data\finger_vibration\cropped_videos"

    video_cropping_parameters_path = r"C:\Users\mitim\Desktop\MITHIC\code\DUSTrack\pia02_workflow\fingor_vibration\video_cropping_parameters.json"
    video_cropping_parameters = json.load(open(video_cropping_parameters_path))
    video_cropping_parameters = video_cropping_parameters["video_cropping_parameters"]

    video_files = [f for f in os.listdir(original_video_folder) if f.endswith(".MP4")]
    video_files.sort()
    print(f"Number of video files: {len(video_files)}")

    # loop all the videos in the video_cropping_parameters
    for video_name, video_parameters in video_cropping_parameters.items():
        print(f"Processing video: {video_name}")
        # print meta data with get_video_info
        video_path = os.path.join(original_video_folder, video_name + ".MP4")
        get_video_info(video_path)
        cropped_video_path = os.path.join(cropped_video_folder, video_name + ".MP4")
        crop_mp4_ffmpeg_python(video_path, cropped_video_path, video_parameters["start_sec"], video_parameters["end_sec"], video_parameters["x"], video_parameters["y"], video_parameters["w"], video_parameters["h"])
        # print some white space
        print("\n" + "="*100 + "\n")





