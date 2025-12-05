import datanavigator
import os
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the parent directory to Python path so we can import dustrack
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from dustrack import DUSTrack, DLCProject

prediction_root_folder = r"\\192.168.1.104\home\piano\DLC_MODELS\general\interosseous_pn23-x-2025-09-11\videos\iteration-6"

video_root_path = r"\\192.168.1.104\home\piano\us_videos_for_tracking"

annotation_file_path = r"\\192.168.1.104\home\piano\DLC_MODELS\general\interosseous_pn23-x-2025-09-11\videos\pia02_s049_005_LFA_annotations_iteration-3.json"

video_name = "pia02_s049_005_RFA"


if __name__ == "__main__":

    print(f"Processing video: {video_name}")

    # get the video path
    video_path = os.path.join(video_root_path, f"{video_name}.mp4")

    # find the .h5 files start with the video name in the prediction root folder
    prediction_files = [f for f in os.listdir(prediction_root_folder) if f.startswith(video_name) and f.endswith('.h5')]

    # stop if there is more than one file or no file is found
    if len(prediction_files) != 1:
        raise ValueError(f"Expected 1 prediction file for {video_name}, but found {len(prediction_files)}")
    
    prediction_file = prediction_files[0]

    prediction_file_path = os.path.join(prediction_root_folder, prediction_file)

    annotation_layers = {
    'prediction': prediction_file_path,
    # 'annotation': annotation_file_path
    }

    d = DUSTrack(video_path, annotation_layers)

    plt.show()