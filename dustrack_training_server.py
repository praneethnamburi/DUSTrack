import sys
from pathlib import Path
import os
import time
from tqdm import tqdm
import numpy as np
import deeplabcut
import argparse
# Add the parent directory to Python path so we can import dustrack
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from dustrack import DUSTrack, DLCProject

if __name__ == "__main__":

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # import torch
    # # print the number of GPUs available
    # print(f"Number of GPUs available: {torch.cuda.device_count()}")

    parser = argparse.ArgumentParser(
        description="DUSTrack training server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dlc-project-config-path", type=str, default=r"\\192.168.1.104\home\piano\DLC_MODELS\participant_models\001\001_LFA-x-2025-11-07\config.yaml", help="Path to DLC project config file")

    parser.add_argument("--max-iters", type=int, default=25, help="Maximum number of iterations")

    # cuda device:
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA device to use")

    # analyse batch size:
    parser.add_argument("--analyse-batch-size", type=int, default=128, help="Batch size for analysis")

    args = parser.parse_args()


    # select random continouse 5 numbers, between 1 to 100
    # get a random number from 1 to 100
    random_number = np.random.randint(1, 10)
    start_core = random_number
    end_core = start_core + 4
    print(f"Start core: {start_core}, End core: {end_core}")
    core = list(range(start_core, end_core))
    print(f"Core: {core}")
    os.sched_setaffinity(0, core)



    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    import torch
    # print the number of GPUs available
    print(f"Number of GPUs available: {torch.cuda.device_count()}")

    dlc_project_config_path = args.dlc_project_config_path
    # check if the project exists
    if not os.path.exists(dlc_project_config_path):
        print(f"Project config file {dlc_project_config_path} does not exist")
        exit()

    dlcp = DLCProject(path = dlc_project_config_path)

    dlcp.process(maxiters=args.max_iters, analyse_batchsize=args.analyse_batch_size)









