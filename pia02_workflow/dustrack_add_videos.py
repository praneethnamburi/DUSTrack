# add videos to the dlc project

import deeplabcut
import os
# from utils import dlc_edit_config, copy_annotations
from pia02_project_config import root_dir #, dlc_project_config_path

if __name__ == "__main__":

    #########################################################
    # this should be the root directory of that participant!
    root_dir = root_dir
    dlc_project_config_path = r"\\192.168.1.104\home\piano\DLC_MODELS\001\pia02_s001_RFA_hw-x-2025-09-03\config.yaml"
    
    participant_id = '001' # modify the number to the participant id
    # hand_side = 'LFA'
    # annotator_name = 'hw' # change this to your name initials

    # add the videos you have annotated
    videos = [
        'pia02_s001_003_RFA.mp4',
        'pia02_s001_004_RFA.mp4',
        'pia02_s001_005_RFA.mp4',
        'pia02_s001_006_RFA.mp4',
        'pia02_s001_007_RFA.mp4',
        'pia02_s001_008_RFA.mp4',   
        'pia02_s001_009_RFA.mp4',
        'pia02_s001_010_RFA.mp4',
        'pia02_s001_011_RFA.mp4',
        'pia02_s001_012_RFA.mp4',
        'pia02_s001_013_RFA.mp4',
        'pia02_s001_014_RFA.mp4',
        'pia02_s001_015_RFA.mp4',
    ]
    
    #########################################################

    participant_root_dir = os.path.join(root_dir, participant_id)

    videos = [os.path.join(participant_root_dir, video) for video in videos]
    # check if all the videos exist
    for video in videos:
        if not os.path.exists(video):
            print(f"Video {video} does not exist")
            exit()
    
    print(f"Adding videos to the project at: {dlc_project_config_path}")

    deeplabcut.add_new_videos(dlc_project_config_path, videos, copy_videos=True)

    # copy_annotations(videos, dlc_project_config_path)

    print(f"Videos added to the project at: {dlc_project_config_path}")


