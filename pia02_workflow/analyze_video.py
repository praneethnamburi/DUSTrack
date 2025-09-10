import sys
from pathlib import Path
import os
import time
from tqdm import tqdm


# Add the parent directory to Python path so we can import dustrack
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from dustrack import DUSTrack, DLCProject
# from pia02_project_config import dlc_project_config_path

if __name__ == "__main__":

    # # sleep for 4 hours use tqdm to show the progress
    # for i in tqdm(range(6 * 60 * 60)):
    #     time.sleep(1)


    dlc_project_config_path = r"\\192.168.1.104\home\piano\DLC_MODELS\022\pia02_s022_RFA_hw-x-2025-09-09\config.yaml"

    videos = [r"\\192.168.1.104\home\piano\DLC_MODELS\022\pia02_s022_001_LFA.mp4"]

        # check if the config file exists
    if not os.path.exists(dlc_project_config_path):
        print(f"Config file {dlc_project_config_path} does not exist")
        exit()

    dlcp = DLCProject(path = dlc_project_config_path)

    # snapshotindex = dlcp.get_best_snapshot_idx(dlcp.latest_iteration)

    # dlcp.add_videos()
    dlcp.analyze_videos(videos=videos)