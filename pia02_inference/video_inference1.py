import sys
from pathlib import Path
import os
import time
from tqdm import tqdm

import deeplabcut

# Add the parent directory to Python path so we can import dustrack
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from dustrack import DUSTrack, DLCProject

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    import torch
    # print the number of GPUs available
    print(f"Number of GPUs available: {torch.cuda.device_count()}")



    dlc_project_config_path = r"/home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/general/interosseous_pn24-x-2025-10-24/config.yaml"
    
    # check if the project exists
    if not os.path.exists(dlc_project_config_path):
        print(f"Project config file {dlc_project_config_path} does not exist")
        exit()

    dlcp = DLCProject(path = dlc_project_config_path)

    video_root_path = r"/home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/general/interosseous_pn24-x-2025-10-24/videos"
    video_list = [os.path.join(video_root_path, f) for f in os.listdir(video_root_path) if f.endswith('.mp4')]
    videos = [video_list[20]]
    dlcp.analyze_videos(iteration_num=0, create_video=False, batchsize=64, videos=videos)









