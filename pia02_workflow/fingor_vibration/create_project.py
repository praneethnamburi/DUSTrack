import datanavigator
import os
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import deeplabcut
# Add the parent directory to Python path so we can import dustrack
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
import numpy as np



if __name__ == "__main__":

    video_root_path = r"\\192.168.1.104\home\piano\data\finger_vibration\cropped_videos"

    project_name = 'finger_vibration'
    annotator_name = 'x'

    # get all the video files from the video_root_path
    video_files = [f for f in os.listdir(video_root_path) if f.endswith(".MP4")]
    video_files.sort()
    video_paths = [os.path.join(video_root_path, f) for f in video_files]
    working_directory = r"\\192.168.1.104\home\piano\data\finger_vibration"


    project_config_path = deeplabcut.create_new_project(
            project_name,
            annotator_name,
            video_paths,
            working_directory=working_directory,
            copy_videos=True,
            multianimal=False
            )
    
    dlc_models_pytorch_folder = os.path.join(os.path.dirname(project_config_path), 'dlc-models-pytorch')
    if not os.path.exists(dlc_models_pytorch_folder):
        os.makedirs(dlc_models_pytorch_folder)
    
    # edit config file, make the bodypart is ['point0', 'point1'] and skeleton is None
    config_dict = {'bodyparts': ['point0', 'point1', 'point2'], 'skeleton': None}
    deeplabcut.auxiliaryfunctions.edit_config(project_config_path, config_dict)

