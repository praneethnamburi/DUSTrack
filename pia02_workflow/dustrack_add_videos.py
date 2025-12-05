# add videos to the dlc project

import deeplabcut
import os
# from utils import dlc_edit_config, copy_annotations

if __name__ == "__main__":

    #########################################################
    # this should be the root directory of that participant!
    dlc_project_config_path = r"\\192.168.1.104\home\piano\DLC_MODELS\general\interosseous_pn24-x-2025-10-24\config.yaml"
    
    participant_id = '022' # modify the number to the participant id
    # hand_side = 'LFA'
    # annotator_name = 'hw' # change this to your name initials

    # add the videos you have annotated

    root_dir = r"\\192.168.1.104\home\piano\DLC_MODELS\general\interosseous_pn24-x-2025-10-24\videos"

    # get all the video file in the root_dir
    videos = [f for f in os.listdir(root_dir) if f.endswith('.mp4')]
    videos.sort()
    print(f"Number of videos: {len(videos)}")
    
    #########################################################


    videos = [os.path.join(root_dir, video) for video in videos]
    # check if all the videos exist
    for video in videos:
        if not os.path.exists(video):
            print(f"Video {video} does not exist")
            exit()
    
    print(f"Adding videos to the project at: {dlc_project_config_path}")

    deeplabcut.add_new_videos(dlc_project_config_path, videos, copy_videos=True)

    # copy_annotations(videos, dlc_project_config_path)
    print(f"Videos added to the project at: {dlc_project_config_path}")

