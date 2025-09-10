

import datanavigator
import os
import matplotlib.pyplot as plt
from pathlib import Path
import sys



# Add the parent directory to Python path so we can import dustrack
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from dustrack import DUSTrack, DLCProject



if __name__ == "__main__":
    
    # root_dir = r"C:\Users\haowe\OneDrive\Desktop\MIT\PianoProject\Code\PianoProjectVenv\data\pia_02"
    # participant_id = '021'
    # video_name = 'pia02_s021_013_LFA.mp4'
    # annotation_dir = r"C:\Users\haowe\OneDrive\Desktop\MIT\PianoProject\Code\PianoProjectVenv\data\pia_02\021"

    # # Check if the root directory exists
    # if not os.path.exists(root_dir):
    #     print(f"Root directory {root_dir} does not exist")
    #     exit()

    # vpath = os.path.join(root_dir, participant_id, video_name)

    # # check if the video path is valid
    # if not os.path.exists(vpath):
    #     print(f"Video path {vpath} does not exist")
    #     exit()

    # Define the annotation layers to load
    vpath = r"\\192.168.1.104\home\piano\DLC_MODELS\022\pia02_s022_001_LFA.mp4"


    annotation_layers = {
        'prediction': r'\\192.168.1.104\home\piano\DLC_MODELS\general\pia02_s018_RFA-x-2025-09-04\videos\iteration-22\pia02_s022_001_LFADLC_Resnet50_pia02_s018_RFASep4shuffle1_snapshot_125.h5'    
        }    
    # Check if annotation files exist
    for layer_name, file_path in annotation_layers.items():
        if not os.path.exists(file_path):
            print(f"Annotation file {file_path} does not exist")
            exit()
        else:
            print(f"Found annotation layer '{layer_name}': {file_path}")

    # Create DUSTrack with multiple annotation layers        
    d = DUSTrack(vpath, annotation_layers)
    
    # Keep the GUI window open
    plt.show()


   
