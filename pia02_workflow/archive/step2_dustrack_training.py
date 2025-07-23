import datanavigator
import os
import matplotlib.pyplot as plt
from dustrack import DUSTrack, DLCProject


if __name__ == "__main__":

    dlc_project_config_path = r"C:\Users\haowe\OneDrive\Desktop\MIT\PianoProject\Code\PianoProjectVenv\data\pia_02\021\pia02_s021_001_LFA_hw-2025-07-22\config.yaml"
    
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
        maxiters = 700
        multi_step = [
            [0.001, 4000],
            [0.0005, 7000],
            [0.0002, 9000],
            [0.0001, 10_000],
        ]
        display_iters = 20
        save_iters = 100
    elif dataset_size == 'medium':
        batch_size = 4
        maxiters = 12_500
        multi_step = [
            [0.004, 5000],
            [0.002, 9000],
            [0.001, 11_500],
            [0.0003, 12_500],
        ]
        display_iters = 200
        save_iters = 1000
    elif dataset_size == 'large':
        batch_size = 8
        maxiters = 25_000
        multi_step = [
            [0.006, 10_000],
            [0.003, 18_000],
            [0.0015, 23_000],
            [0.0007, 25_000],
        ]
        display_iters = 500
        save_iters = 2000
    elif dataset_size == 'extralarge':
        batch_size = 16
        maxiters = 32_000
        multi_step = [
            [0.010, 12_000],
            [0.005, 24_000],
            [0.002, 30_000],
            [0.0008, 32_000],
        ]
        display_iters = 500
        save_iters = 2000
    else:
        print(f"Invalid dataset_size: {dataset_size}. Please choose 'small', 'medium', 'large', or 'extralarge'")
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

    # dlcp.process(maxiters=maxiters)