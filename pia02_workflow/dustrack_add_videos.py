# add videos to the dlc project

import deeplabcut
import os
from utils import dlc_edit_config, copy_annotations


if __name__ == "__main__":

    #########################################################
        # this should be the root directory of that participant!
    root_dir = r"C:\Users\haowe\OneDrive\Desktop\MIT\PianoProject\Code\PianoProjectVenv\data\pia_02"
    config_path = r'C:\Users\haowe\OneDrive\Desktop\MIT\PianoProject\Code\PianoProjectVenv\data\pia_02\021\pia02_021_LFA-hw-2025-07-23\config.yaml'
    
    participant_id = '021' # modify the number to the participant id
    hand_side = 'LFA'
    annotator_name = 'hw' # change this to your name initials

    # add the videos you have annotated
    videos = [
        'pia02_s021_013_LFA.mp4'
    ]
    
    #########################################################

    participant_root_dir = os.path.join(root_dir, participant_id)

    videos = [os.path.join(participant_root_dir, video) for video in videos]

    deeplabcut.add_new_videos(config_path, videos, copy_videos=True)

    copy_annotations(videos, config_path)

    print(f"Videos added to the project at: {config_path}")


