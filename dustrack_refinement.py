import datanavigator
import os
import matplotlib.pyplot as plt
# from dustrack import DUSTrack, DLCProject
from datanavigator.pointtracking import VideoPointAnnotator

if __name__ == "__main__":
    annotator_name = 'hw'
    
    root_dir = r"M:\DLC_MODELS"
    participant_id = '021'

        # Check if the root directory exists
    if not os.path.exists(root_dir):
        print(f"Root directory {root_dir} does not exist")
        exit()

    
    video_path = r'M:\DLC_MODELS\021\pia02_s021_001_LFA_hw-x-2025-07-17\videos\pia02_s021_001_LFA.mp4'

    # check if the video path is valid
    if not os.path.exists(video_path):
        print(f"Video path {video_path} does not exist")
        exit()
    
    original_label = r'M:\DLC_MODELS\021\pia02_s021_001_LFA_hw-x-2025-07-17\videos\pia02_s021_001_LFA_annotations_hw.json'

    # check if the original label path is valid
    if not os.path.exists(original_label):
        print(f"Original label path {original_label} does not exist")
        exit()

    iteration0_label = r'M:\DLC_MODELS\021\pia02_s021_001_LFA_hw-x-2025-07-17\videos\iteration-0\pia02_s021_001_LFADLC_resnet50_pia02_s021_001_LFA_hwJul17shuffle1_6500.h5'

    # check if the overlay label path is valid
    if not os.path.exists(iteration0_label):
        print(f"Overlay label path {iteration0_label} does not exist")
        exit()

    labels = {
        'original_label': original_label,
        'iteration0_label': iteration0_label,
        }


    # check if the video path is valid
    if not os.path.exists(video_path):
        print(f"Video path {video_path} does not exist")
        exit()

    d = VideoPointAnnotator(video_path, labels)

    # d.ann.set_plot_type('line')
    
    # Keep the GUI window open
    plt.show()


