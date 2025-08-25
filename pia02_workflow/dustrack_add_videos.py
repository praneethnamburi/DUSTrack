# add videos to the dlc project

import deeplabcut
import os
# from utils import dlc_edit_config, copy_annotations
from pia02_project_config import root_dir, dlc_project_config_path

if __name__ == "__main__":

    #########################################################
    # this should be the root directory of that participant!
    root_dir = root_dir
    dlc_project_config_path = dlc_project_config_path
    
    participant_id = '018' # modify the number to the participant id
    # hand_side = 'LFA'
    annotator_name = 'hw' # change this to your name initials

    # add the videos you have annotated
    videos = [
        'pia02_s018_002_LFA.mp4',
        'pia02_s018_003_LFA.mp4',
        'pia02_s018_004_LFA.mp4',
        'pia02_s018_005_LFA.mp4',
        'pia02_s018_006_LFA.mp4',
        'pia02_s018_007_LFA.mp4',
        'pia02_s018_008_LFA.mp4',
        'pia02_s018_009_LFA.mp4',
        'pia02_s018_010_LFA.mp4',
        'pia02_s018_011_LFA.mp4',
        'pia02_s018_012_LFA.mp4',
        'pia02_s018_013_LFA.mp4',
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


