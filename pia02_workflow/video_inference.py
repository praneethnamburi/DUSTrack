#!/usr/bin/env python3
"""
Simple single-process video inference for DeepLabCut.

Scans a directory of .mp4 files and runs DLC inference on all of them
on a single GPU. No multiprocessing, no log files.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to Python path so we can import dustrack
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

DEFAULT_VIDEO_DIR   = r"\\192.168.1.104\home\piano\us_videos_for_tracking2"
DEFAULT_CONFIG_PATH = r"\\192.168.1.104\home\piano\DLC_MODELS\general\interosseous_pn24-x-2025-10-24\config.yaml"
DEFAULT_CUDA_DEVICE = "0"
DEFAULT_BATCH_SIZE  = 16
DEFAULT_ITERATION   = 0


def main():
    parser = argparse.ArgumentParser(
        description="Simple single-process DLC inference over a directory of .mp4 videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video-dir",     default=DEFAULT_VIDEO_DIR,
                        help="Directory containing .mp4 videos")
    parser.add_argument("--config-path",   default=DEFAULT_CONFIG_PATH,
                        help="Path to DLC config.yaml")
    parser.add_argument("--cuda-device",   default=DEFAULT_CUDA_DEVICE,
                        help="GPU ID to use (single device, e.g. '0')")
    parser.add_argument("--batch-size",    type=int, default=DEFAULT_BATCH_SIZE,
                        help="Inference batch size (frames per forward pass)")
    parser.add_argument("--iteration-num", type=int, default=DEFAULT_ITERATION,
                        help="DLC iteration number to use")
    args = parser.parse_args()

    # Must set CUDA_VISIBLE_DEVICES before torch is imported.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    parent_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(parent_dir))

    import torch
    from dustrack import DLCProject

    print(f"CUDA available: {torch.cuda.is_available()}, GPUs visible: {torch.cuda.device_count()}")

    if not os.path.isfile(args.config_path):
        print(f"ERROR: Config file not found: {args.config_path}")
        sys.exit(1)

    if not os.path.isdir(args.video_dir):
        print(f"ERROR: Video directory not found: {args.video_dir}")
        sys.exit(1)

    video_list = sorted(
        os.path.join(args.video_dir, f)
        for f in os.listdir(args.video_dir)
        if f.endswith(".mp4")
    )

    if not video_list:
        print(f"ERROR: No .mp4 videos found in {args.video_dir}")
        sys.exit(1)

    print("=" * 60)
    print(f"  GPU:           {args.cuda_device}")
    print(f"  Config:        {args.config_path}")
    print(f"  Video dir:     {args.video_dir}")
    print(f"  Videos found:  {len(video_list)}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Iteration:     {args.iteration_num}")
    print("=" * 60)

    dlcp = DLCProject(path=args.config_path)
    dlcp.analyze_videos(
        iteration_num=args.iteration_num,
        create_video=False,
        batchsize=args.batch_size,
        videos=video_list,
    )

    print("Done.")


if __name__ == "__main__":
    main()
