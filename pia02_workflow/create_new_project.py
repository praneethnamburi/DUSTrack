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

from dustrack import DUSTrack, DLCProject


if __name__ == "__main__":

    
    # participant_ids are a list of participant id with format 001, 002, 003, 004, etc
    participant_ids = np.arange(1, 11)
    participant_ids = [f'0{participant_id:02d}' for participant_id in participant_ids]
    print(f"Number of participant ids: {len(participant_ids)}")
    # print all participant ids
    print(f"Participant ids: {participant_ids}")


    hands = ['LFA', 'RFA']

    annotator_name = 'x' # modify the name to the annotator name
    
    video_root_path = r'\\192.168.1.104\home\piano\us_videos_for_tracking2'
    root_participant_working_directory = r'\\192.168.1.104\home\piano\DLC_MODELS\participant_models'

    # check if the video root path exists
    if not os.path.exists(video_root_path):
        print(f"Video root path {video_root_path} does not exist")
        exit()

    # check if the video root path is a directory
    if not os.path.isdir(video_root_path):
        print(f"Video root path {video_root_path} is not a directory")
        exit()

    for participant_id in participant_ids:

        for hand in hands:

            # get any .mp4 file with format: pia02_s{participant_id}_xxx_{hand}2.mp4
            # start with pia02_s{participant_id}_ and end with _{hand}2.mp4
            video_files = [f for f in os.listdir(video_root_path) if f.startswith(f'pia02_s{participant_id}_') and f.endswith(f'_{hand}2.mp4')]
            # sort the video file strings
            video_files.sort()
            
            if len(video_files) == 0:
                print(f"No video files found for participant {participant_id} and hand {hand}")
                exit()

            video_paths = [os.path.join(video_root_path, f) for f in video_files]
            project_name = f'{participant_id}_{hand}'

            # create folder name with participant id:
            participant_working_directory = os.path.join(root_participant_working_directory, f'{participant_id}')
            if not os.path.exists(participant_working_directory):
                os.makedirs(participant_working_directory)

            project_config_path = deeplabcut.create_new_project(
            project_name,
            annotator_name,
            video_paths,
            working_directory=participant_working_directory,
            copy_videos=True,
            multianimal=False
            )

            # create a folder with name dlc-models-pytorch in the parent of the config file
            dlc_models_pytorch_folder = os.path.join(os.path.dirname(project_config_path), 'dlc-models-pytorch')
            if not os.path.exists(dlc_models_pytorch_folder):
                os.makedirs(dlc_models_pytorch_folder)

            # edit config file, make the bodypart is ['point0', 'point1'] and skeleton is None
            config_dict = {'bodyparts': ['point0', 'point1'], 'skeleton': None}
            deeplabcut.auxiliaryfunctions.edit_config(project_config_path, config_dict)
        


