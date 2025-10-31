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

    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    import torch
    # print the number of GPUs available
    print(f"Number of GPUs available: {torch.cuda.device_count()}")



    dlc_project_config_path = r"/home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/general/interosseous_pn24-x-2025-10-24/config.yaml"
    
    # check if the project exists
    if not os.path.exists(dlc_project_config_path):
        print(f"Project config file {dlc_project_config_path} does not exist")
        exit()

    dlcp = DLCProject(path = dlc_project_config_path)
    # dlcp.extract_frames()
    # dlcp.extract_frames()

    # create training dataset

    # net_type = 'resnet_50'
    # epochs=300
    # max_snapshots_to_keep = 20
    # batch_size = 16

    # deeplabcut.create_training_dataset(dlc_project_config_path, net_type=net_type)

    # deeplabcut.train_network(dlc_project_config_path, epochs=epochs, max_snapshots_to_keep=max_snapshots_to_keep, shuffle=1, batch_size=batch_size)

    # a = {'snapshotindex': 'all'}
    # deeplabcut.auxiliaryfunctions.edit_config(configname=dlc_project_config_path, edits=a)
    # deeplabcut.evaluate_network(config=dlc_project_config_path)


    # # dlcp.evaluate()

    dlcp.analyze_videos(iteration_num=0, create_video=False)









