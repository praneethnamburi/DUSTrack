from pathlib import Path
import sys
# import dustrack
import numpy as np
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))
from dustrack import DUSTrack, DLCProject


if __name__ == "__main__":
    # select random continouse 5 numbers, between 1 to 100
    # get a random number from 1 to 100 generate real random number
    import random
    import os
    import argparse

    # random_number = random.randint(1, 70)
    # start_core = random_number
    # end_core = start_core + 3
    # print(f"Start core: {start_core}, End core: {end_core}")
    # core_range = (start_core, end_core)
    # core = list(range(start_core, end_core))
    # print(f"Core: {core}")
    # os.sched_setaffinity(0, core)

    parser = argparse.ArgumentParser(
        description="DUSTrack training server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dlc-project-config-path", type=str, default=r"\\192.168.1.104\home\piano\data\overhead_camera\dlc_projects\overhead_camera\keyboard_segmentation-x-2026-02-22\config.yaml", help="Path to DLC project config file")

    parser.add_argument("--max-iters", type=int, default=200, help="Maximum number of iterations")

    # cuda device:
    parser.add_argument("--cuda-device", type=int, default=1, help="CUDA device to use")

    # analyse batch size:
    parser.add_argument("--analyze-batchsize", type=int, default=8, help="Batch size for analysis")

    args = parser.parse_args()

    # source_model_path = r"/home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/participant_models_general/snapshot-best-270.pt"
    # # check if the source model path exists
    # if not os.path.exists(source_model_path):
    #     print(f"Source model file {source_model_path} does not exist")
    #     exit()

    from pathlib import Path
    import sys
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    from dustrack import DUSTrack, DLCProject


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

    dlcp.process(maxiters=args.max_iters, analyse_batchsize=args.analyze_batchsize, create_video=False)

    from dustrack import _config
    _config.DLC3_USE_LAST_SNAPSHOT = False
    dlcp.analyze_videos(create_video=False, batchsize=args.analyze_batchsize)

    # # print complte message
    print(f"Training completed for {dlc_project_config_path}")
