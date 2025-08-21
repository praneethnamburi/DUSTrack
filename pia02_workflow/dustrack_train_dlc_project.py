import sys
from pathlib import Path
import os

# Add the parent directory to Python path so we can import dustrack
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from dustrack import DUSTrack, DLCProject


if __name__ == "__main__":

    dlc_project_config_path = r"\\192.168.1.104\home\piano\DLC_MODELS\021\pia02_s021_001_LFA_hw-x-2025-08-21\config.yaml"

    # Dataset size configuration - choose one: 'small', 'medium', 'large', 'extralarge'
    dataset_size = 'small'  # Change this to match your dataset size
    
    max_snapshots_to_keep = 20

    # Initialize variables
    batch_size = None
    maxiters = None
    multi_step = None
    display_iters = None
    save_iters = None

    # Set configuration based on dataset size
    if dataset_size == 'small':
        batch_size = 1
        maxiters = 40_000
        multi_step = [
            [0.001, 16000],
            [0.0005, 28000],
            [0.0002, 36000],
            [0.0001, 40_000],
        ]
        display_iters = 1000
        save_iters = 2000
    elif dataset_size == 'medium':
        batch_size = 4
        maxiters = 50_000
        multi_step = [
            [0.001, 20000],
            [0.0005, 36000],
            [0.0002, 46000],
            [0.0001, 50_000],
        ]
        display_iters = 200
        save_iters = 2500
    elif dataset_size == 'large':
        batch_size = 8
        maxiters = 75_000
        multi_step = [
            [0.001, 30_000],
            [0.0005, 52_500],
            [0.0002, 67_500],
            [0.0001, 75_000],
        ]
        display_iters = 500
        save_iters = 3750
    elif dataset_size == 'extralarge':
        batch_size = 16
        maxiters = 100_000
        multi_step = [
            [0.001, 37_500],
            [0.0005, 75_000],
            [0.0002, 93_750],
            [0.0001, 100_000],
        ]
        display_iters = 200
        save_iters = 5000
    
    elif dataset_size == 'original_settings':
        batch_size = 1
        maxiters = 500000
        multi_step = [[0.005, 10000], [0.02, 350000], [0.002, 425000], [0.001, 1000000]]
        display_iters = 200
        save_iters = 25000
        max_snapshots_to_keep = 20

    else:
        print(f"Invalid dataset_size: {dataset_size}. Please choose 'small', 'medium', 'large', 'extralarge', or 'original_settings'")
        exit()

    print(f"Using {dataset_size} configuration:")
    print(f"  Max iterations: {maxiters}")
    print(f"  Batch size: {batch_size}")
    print(f"  Display every: {display_iters} iterations")
    print(f"  Save every: {save_iters} iterations")
    print(f"  Learning rate schedule: {multi_step}")

    # check if the config file exists
    if not os.path.exists(dlc_project_config_path):
        print(f"Config file {dlc_project_config_path} does not exist")
        exit()

    dlcp = DLCProject(path = dlc_project_config_path)


    dlcp.process(maxiters=maxiters, multi_step=multi_step, batch_size=batch_size, display_iters=display_iters, save_iters=save_iters, max_snapshots_to_keep=max_snapshots_to_keep)

    


