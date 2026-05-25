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

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    import torch
    # print the number of GPUs available
    print(f"Number of GPUs available: {torch.cuda.device_count()}")



    dlc_project_config_path = r"\\192.168.1.104\home\piano\DLC_MODELS\participant_models_general\s029\LFA\interosseous_pn24-x-2025-10-24\config.yaml"
    source_model_path = r"\\192.168.1.104\home\piano\DLC_MODELS\participant_models_general\snapshot-best-270.pt"
    import torch
    # print the number of GPUs available
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    # check if the project exists
    if not os.path.exists(dlc_project_config_path):
        print(f"Project config file {dlc_project_config_path} does not exist")
        exit()

    dlcp = DLCProject(path = dlc_project_config_path)

    dlcp.process(maxiters=100, analyse_batchsize=8, create_video=False, refine = source_model_path)

    # from dustrack import _config
    # _config.DLC3_USE_LAST_SNAPSHOT = False
    # dlcp.analyze_videos(create_video=False, batchsize=args.analyze_batchsize)

    # # print complte message
    print(f"Training completed for {dlc_project_config_path}")



