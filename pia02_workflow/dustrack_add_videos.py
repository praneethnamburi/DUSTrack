# add videos to the dlc project

import deeplabcut
import os
from utils import dlc_edit_config, copy_annotations


if __name__ == "__main__":

    #########################################################
        # this should be the root directory of that participant!
    data_root_dir = r"\\192.168.1.104\home\piano\DLC_MODELS\021\pia02_s021_001_LFA_hw-x-2025-08-21"
    dlc_project_config_path = r'\\192.168.1.104\home\piano\DLC_MODELS\021\pia02_s021_001_LFA_hw-x-2025-08-21\config.yaml'
    
    participant_id = '021' # modify the number to the participant id
    hand_side = 'LFA'
    annotator_name = 'hw' # change this to your name initials

    # add the videos you have annotated
    videos = [
        'pia02_s021_002_LFA.mp4',
        'pia02_s021_003_LFA.mp4',
        'pia02_s021_004_LFA.mp4',
        'pia02_s021_005_LFA.mp4',
        'pia02_s021_006_LFA.mp4',
        'pia02_s021_007_LFA.mp4',
        'pia02_s021_008_LFA.mp4',
        'pia02_s021_009_LFA.mp4',
        'pia02_s021_010_LFA.mp4',
        'pia02_s021_011_LFA.mp4',
        'pia02_s021_012_LFA.mp4',
        'pia02_s021_013_LFA.mp4',
        'pia02_s021_014_LFA.mp4',
        'pia02_s021_015_LFA.mp4',
    ]
    
    #########################################################

    participant_root_dir = os.path.join(data_root_dir, participant_id)

    videos = [os.path.join(participant_root_dir, video) for video in videos]
    # check if all the videos exist
    for video in videos:
        if not os.path.exists(video):
            print(f"Video {video} does not exist")
            exit()

    deeplabcut.add_new_videos(dlc_project_config_path, videos, copy_videos=True)

    copy_annotations(videos, dlc_project_config_path)

    print(f"Videos added to the project at: {dlc_project_config_path}")


