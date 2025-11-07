import datanavigator
import os
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import deeplabcut
# Add the parent directory to Python path so we can import dustrack
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


from dustrack import DUSTrack, DLCProject


if __name__ == "__main__":
    
    video_root_path = r'/home/hwjwei/scratch/hwjwei/pia02/data/us_videos_for_tracking2'
    root_participant_working_directory = r'/home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/participant_models'

    # check if the video root path exists
    if not os.path.exists(video_root_path):
        print(f"Video root path {video_root_path} does not exist")
        exit()

    # check if the video root path is a directory
    if not os.path.isdir(video_root_path):
        print(f"Video root path {video_root_path} is not a directory")
        exit()

    annotator_name = 'x'
    participant_id = '001'
    hand = 'LFA'

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


