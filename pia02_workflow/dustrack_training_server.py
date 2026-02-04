

if __name__ == "__main__":
    # select random continouse 5 numbers, between 1 to 100
    # get a random number from 1 to 100 generate real random number
    import random
    import os
    import argparse

    random_number = random.randint(1, 70)
    start_core = random_number
    end_core = start_core + 3
    print(f"Start core: {start_core}, End core: {end_core}")
    core_range = (start_core, end_core)
    core = list(range(start_core, end_core))
    print(f"Core: {core}")
    os.sched_setaffinity(0, core)

    parser = argparse.ArgumentParser(
        description="DUSTrack training server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dlc-project-config-path", type=str, default=r"/home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/participant_models/001/001_LFA-x-2025-11-07/config.yaml", help="Path to DLC project config file")

    parser.add_argument("--max-iters", type=int, default=100, help="Maximum number of iterations")

    # cuda device:
    parser.add_argument("--cuda-device", type=int, default=1, help="CUDA device to use")

    # analyse batch size:
    parser.add_argument("--analyze-batchsize", type=int, default=16, help="Batch size for analysis")

    args = parser.parse_args()

    source_model_path = r"/home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/participant_models_general/snapshot-best-270.pt"
    # check if the source model path exists
    if not os.path.exists(source_model_path):
        print(f"Source model file {source_model_path} does not exist")
        exit()

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

    dlcp.process(maxiters=args.max_iters, analyse_batchsize=args.analyze_batchsize, create_video=False, refine = source_model_path)

    from dustrack import _config
    _config.DLC3_USE_LAST_SNAPSHOT = False
    dlcp.analyze_videos(create_video=False, batchsize=args.analyze_batchsize)

    # # print complte message
    print(f"Training completed for {dlc_project_config_path}")



# cd /home/hwjwei ; /usr/bin/env /home/hwjwei/miniconda3/envs/deeplabcut3.13/bin/python /home/hwjwei/.cursor-server/extensions/ms-python.debugpy-2025.14.1-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher 32901 -- /home/hwjwei/projects/pia02/DUSTrack/pia02_workflow/dustrack_training_server.py 
 #--dlc-project-config-path /home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/participant_models/001/001_LFA-x-2025-11-07/config.yaml --max-iters 200 --cuda-device 0 --analyse-batch-size 128
# /home/hwjwei/miniconda3/envs/deeplabcut3.13/bin/python /home/hwjwei/projects/pia02/DUSTrack/pia02_workflow/dustrack_training_server.py


# /home/hwjwei/miniconda3/envs/deeplabcut3.13/bin/python /home/hwjwei/projects/pia02/DUSTrack/pia02_workflow/dustrack_training_server.py --dlc-project-config-path /home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/participant_models/001/001_LFA-x-2025-11-07/config.yaml --max-iters 200 --cuda-device 0 --analyse-batch-size 128

# conda activate deeplabcut3.13; taskset -c 1-5 python /home/hwjwei/projects/pia02/DUSTrack/pia02_workflow/dustrack_training_server.py --dlc-project-config-path /home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/participant_models/001/001_LFA-x-2025-11-07/config.yaml 

#python /home/hwjwei/projects/pia02/DUSTrack/pia02_workflow/dustrack_training_server.py --dlc-project-config-path /home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/participant_models/003/003_RFA-x-2025-11-07/config.yaml --cuda-device 3

# vpath = r"/home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/participant_models/001/001_LFA-x-2025-11-07/videos/pia02_s001_007_LFA2.mp4"
# prediction_path = r'/home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/participant_models/001/001_LFA-x-2025-11-07/videos/iteration-2/pia02_s001_007_LFA2DLC_Resnet50_001_LFANov7shuffle1_snapshot_200.h5'


# python /home/hwjwei/projects/pia02/DUSTrack/pia02_workflow/dustrack_training_server.py --dlc-project-config-path /home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/participant_models_general/s001/LFA/interosseous_pn24-x-2025-10-24/config.yaml --cuda-device 3
# python /home/hwjwei/projects/pia02/DUSTrack/pia02_workflow/dustrack_training_server.py --dlc-project-config-path /home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/participant_models_general/s001/RFA/interosseous_pn24-x-2025-10-24/config.yaml --cuda-device 3

# python /home/hwjwei/projects/pia02/DUSTrack/pia02_workflow/dustrack_training_server.py --dlc-project-config-path /home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/participant_models_general/s002/LFA/interosseous_pn24-x-2025-10-24/config.yaml --cuda-device 3
# python /home/hwjwei/projects/pia02/DUSTrack/pia02_workflow/dustrack_training_server.py --dlc-project-config-path /home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/participant_models_general/s002/RFA/interosseous_pn24-x-2025-10-24/config.yaml --cuda-device 3

# python /home/hwjwei/projects/pia02/DUSTrack/pia02_workflow/dustrack_training_server.py --dlc-project-config-path /home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/participant_models_general/s003/LFA/interosseous_pn24-x-2025-10-24/config.yaml --cuda-device 3
# python /home/hwjwei/projects/pia02/DUSTrack/pia02_workflow/dustrack_training_server.py --dlc-project-config-path /home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/participant_models_general/s003/RFA/interosseous_pn24-x-2025-10-24/config.yaml --cuda-device 3

# python /home/hwjwei/projects/pia02/DUSTrack/pia02_workflow/dustrack_training_server.py --dlc-project-config-path /home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/participant_models_general/s004/LFA/interosseous_pn24-x-2025-10-24/config.yaml --cuda-device 3
# python /home/hwjwei/projects/pia02/DUSTrack/pia02_workflow/dustrack_training_server.py --dlc-project-config-path /home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/participant_models_general/s004/RFA/interosseous_pn24-x-2025-10-24/config.yaml --cuda-device 3

# python /home/hwjwei/projects/pia02/DUSTrack/pia02_workflow/dustrack_training_server.py --dlc-project-config-path /home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/participant_models_general/s005/LFA/interosseous_pn24-x-2025-10-24/config.yaml --cuda-device 3
# python /home/hwjwei/projects/pia02/DUSTrack/pia02_workflow/dustrack_training_server.py --dlc-project-config-path /home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/participant_models_general/s005/RFA/interosseous_pn24-x-2025-10-24/config.yaml --cuda-device 3