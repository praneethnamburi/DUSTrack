import deeplabcut
import os
from utils import dlc_edit_config, copy_annotations
from pathlib import Path
import sys

# Add the parent directory to Python path so we can import dustrack
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

if __name__ == "__main__":

    #########################################################

    # this should be the root directory of that participant!
    root_dir = r"C:\Users\haowe\OneDrive\Desktop\MIT\PianoProject\Code\PianoProjectVenv\data\pia_02"
    # this would be the path to the DLC_MODELS folder for example: M:\DLC_MODELS

    participant_id = '021' # modify the number to the participant id, please note this doese not have the s prefix
    hand_side = 'LFA' # change this to the hand side you are annotating
    annotator_name = 'hw' # change this to your name initials

    # add the videos you have annotated
    videos = [
        'pia02_s021_013_LFA.mp4'
    ]

    #########################################################
    participant_dir = os.path.join(root_dir, participant_id)
    
    participant_name = 's' + participant_id

    # don't change this for pia02 project
    common_labels = ['point0', 'point1']


    # check if the root directory exists
    if not os.path.exists(participant_dir):
        raise FileNotFoundError(f"Root directory not found at {participant_dir}")
    
    project = f'pia02_{participant_name}_{hand_side}'




    # convert the videos to full paths
    videos = [os.path.join(participant_dir, video) for video in videos]

    project_config_path = deeplabcut.create_new_project(
        project,
        annotator_name,
        videos,
        working_directory=participant_dir,
        copy_videos=True,
        multianimal=False
    )


    # check if the annotator directory exists
    dlc_edit_config(project_config_path, bodyparts=common_labels, skeleton=None)

    # deeplabcut.add_new_videos(project_config_path, videos, copy_videos=True)
    # copy_annotations(videos, project_config_path)

    print(f"Project created at: {project_config_path}")
    

















