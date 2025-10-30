import sys
from pathlib import Path
import os
import time
from tqdm import tqdm

import deeplabcut

# Add the parent directory to Python path so we can import dustrack
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from dustrack import DUSTrack, DLCProject

if __name__ == "__main__":

    dlc_project_config_path = r"\\192.168.1.104\home\piano\DLC_MODELS\general\interosseous_pn24-x-2025-10-24\\config.yaml"
    dlcp = DLCProject(path = dlc_project_config_path)
    # dlcp.extract_frames()
    # dlcp.extract_frames()

    # create training dataset

    net_type = 'resnet_50'
    maxiters=200
    max_snapshots_to_keep = 20

    # deeplabcut.create_training_dataset(dlc_project_config_path, net_type=net_type)

    # deeplabcut.train_network(dlc_project_config_path, maxiters=maxiters, max_snapshots_to_keep=max_snapshots_to_keep, shuffle=1)

    # a = {'snapshotindex': 'all'}
    # deeplabcut.auxiliaryfunctions.edit_config(configname=dlc_project_config_path, edits=a)
    # deeplabcut.evaluate_network(config=dlc_project_config_path)


    # dlcp.evaluate()

    dlcp.analyze_videos(iteration_num=0, create_video=False)









