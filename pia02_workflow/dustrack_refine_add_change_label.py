import sys
from pathlib import Path
import os

# Add the parent directory to Python path so we can import dustrack
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from dustrack import DUSTrack, DLCProject

if __name__ == "__main__":

    #########################################################

    dlc_project_config_path = r"C:\Users\haowe\OneDrive\Desktop\MIT\PianoProject\Code\PianoProjectVenv\data\pia_02\021\pia02_s021_001_LFA_hw-2025-07-22\config.yaml"
    video_index = 0
    new_annotation_suffix = 'iteration-1'

    #########################################################
    
    # check if the config file exists
    if not os.path.exists(dlc_project_config_path):
        print(f"Config file {dlc_project_config_path} does not exist")
        exit()

    dlcp = DLCProject(path = dlc_project_config_path)

    dlcp.annotate(video_index, new_annotation_suffix=new_annotation_suffix)

    
    



