import sys
from pathlib import Path
import os
import matplotlib.pyplot as plt

# Add the parent directory to Python path so we can import dustrack
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from dustrack import DUSTrack, DLCProject
from pia02_project_config import dlc_project_config_path

if __name__ == "__main__":

    #########################################################

    dlc_project_config_path = dlc_project_config_path # from pia02_project_config.py
    video_index = 0

    #########################################################
    
    # check if the config file exists
    if not os.path.exists(dlc_project_config_path):
        print(f"Config file {dlc_project_config_path} does not exist")
        exit()

    dlcp = DLCProject(path = dlc_project_config_path)

    dlcp.annotate(video_index)

    plt.show()

    
    



