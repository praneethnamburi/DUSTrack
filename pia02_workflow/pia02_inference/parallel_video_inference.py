#!/usr/bin/env python3
"""
Parallel Video Inference Script for DeepLabCut
Processes videos in parallel across multiple GPUs with CPU affinity control.
"""

import os
import sys
import argparse
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import time
from datetime import datetime
import traceback


def worker_process(
    worker_id,
    gpu_id,
    core_range,
    video_queue,
    progress_counter,
    config_path,
    iteration_num,
    create_video,
    batch_size,
    videos_per_batch,
    log_dir,
    init_lock
):
    """
    Worker process that processes batches of videos.

    Args:
        worker_id: Unique worker identifier
        gpu_id: GPU ID to assign to this worker
        core_range: Tuple of (start_core, end_core) for CPU affinity
        video_queue: Multiprocessing queue with video batches
        progress_counter: Shared counter for progress tracking
        config_path: Path to DLC config.yaml
        iteration_num: DLC iteration number
        create_video: Whether to create labeled videos
        batch_size: Batch size for inference
        videos_per_batch: Number of videos to process per batch
        log_dir: Directory for log files
        init_lock: Multiprocessing lock to prevent concurrent config access
    """
    # Set up logging for this worker
    log_file = os.path.join(log_dir, f"worker_{worker_id}.log")

    def log(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] {msg}\n")
        print(f"[Worker {worker_id}] {msg}")

    try:
        # Set CPU affinity
        start_core, end_core = core_range
        cores = list(range(start_core, end_core))
        os.sched_setaffinity(0, cores)
        log(f"Set CPU affinity to cores: {cores}")

        # Set GPU before importing torch
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        log(f"Set CUDA_VISIBLE_DEVICES={gpu_id}")

        # Now import torch and DLC-related modules
        import torch

        # Add the parent directory to Python path so we can import dustrack
        from pathlib import Path
        parent_dir = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(parent_dir))

        from dustrack import DLCProject

        log(f"Torch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

        # Initialize DLCProject with lock to prevent concurrent config file access
        # Workers initialize sequentially to avoid race conditions
        log(f"Waiting to load DLC project from {config_path}")
        with init_lock:
            log("Loading DLC project (acquired lock)")
            dlcp = DLCProject(path=config_path)
            log("DLC project loaded successfully (releasing lock)")

        videos_processed = 0

        # wait for 4 seconds so all the process are initialized
        time.sleep(2)
        log(f"Worker {worker_id:2d}: Sleeping for 4 seconds")

        # Process video batches from queue
        while True:

            # use the init lock to prevent concurrent access to the DLC project so delay for 4 seconds
            with init_lock:
                time.sleep(4)
                log(f"Worker {worker_id:2d}: Sleeping for 4 seconds before processing batch")

            try:
                # Get a batch of videos (non-blocking with timeout)
                video_batch = video_queue.get(timeout=1)

                # Check for poison pill (None signals shutdown)
                if video_batch is None:
                    log("Received shutdown signal")
                    break

                # Process this batch
                batch_size_actual = len(video_batch)
                log(f"Processing batch of {batch_size_actual} videos")
                # print start processing video names to the log file
                log(f"Start processing videos: {video_batch}")
                try:
                    # Run inference on the batch
                    dlcp.analyze_videos(
                        iteration_num=iteration_num,
                        create_video=create_video,
                        batchsize=batch_size,
                        videos=video_batch
                    )

                    videos_processed += batch_size_actual

                    # Update progress counter
                    with progress_counter.get_lock():
                        progress_counter.value += batch_size_actual
                    
                    log(f"End processing videos: {video_batch}")
                    log(f"Successfully processed {batch_size_actual} videos (total: {videos_processed})")
                    # log video names to the log file

                except Exception as e:
                    log(f"ERROR processing batch: {str(e)}")
                    log(f"Traceback: {traceback.format_exc()}")
                    log(f"Failed videos: {video_batch}")

            except Empty:
                # Queue is empty, check if we should continue waiting
                continue
            except Exception as e:
                log(f"ERROR getting from queue: {str(e)}")
                break

        log(f"Worker shutting down. Total videos processed: {videos_processed}")

    except Exception as e:
        log(f"FATAL ERROR in worker: {str(e)}")
        log(f"Traceback: {traceback.format_exc()}")


def main():
    parser = argparse.ArgumentParser(
        description="Parallel video inference with DeepLabCut",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments (with defaults for convenience)
    parser.add_argument("--video-dir",
                        default="/home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/general/interosseous_pn24-x-2025-10-24/videos",
                        help="Directory containing .mp4 videos")
    parser.add_argument("--config-path",
                        default="/home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/general/interosseous_pn24-x-2025-10-24/config.yaml",
                        help="Path to DLC config.yaml file")
    parser.add_argument("--gpus",
                        default="4,5,6,7",
                        help="Comma-separated GPU IDs (e.g., '4,5,6')")

    # Worker configuration
    parser.add_argument("--num-workers", type=int, default=12,
                        help="Number of parallel worker processes")
    parser.add_argument("--cores-per-worker", type=int, default=3,
                        help="Number of CPU cores per worker (non-overlapping)")

    # Video processing configuration
    parser.add_argument("--videos-per-batch", type=int, default=3,
                        help="Number of videos each worker processes at once")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for inference (affects GPU memory)")

    # DLC configuration
    parser.add_argument("--iteration-num", type=int, default=0,
                        help="DLC iteration number to use")
    parser.add_argument("--create-video", action="store_true",
                        help="Create labeled videos (slower, disabled by default)")

    # Output configuration
    parser.add_argument("--log-dir", default="./logs",
                        help="Directory for worker log files")

    args = parser.parse_args()

    # Validate and parse GPU list
    try:
        gpu_list = [int(g.strip()) for g in args.gpus.split(",")]
        print(f"Using GPUs: {gpu_list}")
    except ValueError:
        print("ERROR: --gpus must be comma-separated integers (e.g., '4,5,6')")
        sys.exit(1)

    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    print(f"Worker logs will be saved to: {args.log_dir}")

    # Validate paths
    if not os.path.isdir(args.video_dir):
        print(f"ERROR: Video directory not found: {args.video_dir}")
        sys.exit(1)

    if not os.path.isfile(args.config_path):
        print(f"ERROR: Config file not found: {args.config_path}")
        sys.exit(1)

    # Scan for videos
    print(f"Scanning for videos in: {args.video_dir}")
    video_files = [
        os.path.join(args.video_dir, f)
        for f in os.listdir(args.video_dir)
        if f.endswith('.mp4')
    ]
    video_files.sort()  # Sort for consistent ordering

    print(f"Found {len(video_files)} .mp4 videos")

    if len(video_files) == 0:
        print("ERROR: No .mp4 videos found")
        sys.exit(1)

    # Split videos into batches
    video_batches = []
    for i in range(0, len(video_files), args.videos_per_batch):
        batch = video_files[i:i + args.videos_per_batch]
        video_batches.append(batch)

    print(f"Split into {len(video_batches)} batches of up to {args.videos_per_batch} videos each")

    # Create work queue
    video_queue = mp.Queue()
    for batch in video_batches:
        video_queue.put(batch)

    # Add poison pills for workers (None signals shutdown)
    for _ in range(args.num_workers):
        video_queue.put(None)

    # Create shared progress counter and initialization lock
    progress_counter = mp.Value('i', 0)
    init_lock = mp.Lock()  # Prevent concurrent config file access

    # Calculate CPU core assignments (non-overlapping)
    print(f"\nStarting {args.num_workers} workers:")
    print(f"  - CPU cores per worker: {args.cores_per_worker}")
    print(f"  - Videos per batch: {args.videos_per_batch}")
    print(f"  - Inference batch size: {args.batch_size}")
    print(f"  - Create videos: {args.create_video}")
    print(f"  - DLC iteration: {args.iteration_num}")
    print()

    # Launch workers
    workers = []
    for worker_id in range(args.num_workers):
        # Assign GPU (round-robin)
        gpu_id = gpu_list[worker_id % len(gpu_list)]

        # Assign CPU cores (non-overlapping)
        start_core = worker_id * args.cores_per_worker
        end_core = start_core + args.cores_per_worker
        core_range = (start_core, end_core)

        print(f"Worker {worker_id:2d}: GPU {gpu_id}, CPU cores {start_core}-{end_core-1}")

        # Create and start worker process
        p = mp.Process(
            target=worker_process,
            args=(
                worker_id,
                gpu_id,
                core_range,
                video_queue,
                progress_counter,
                args.config_path,
                args.iteration_num,
                args.create_video,
                args.batch_size,
                args.videos_per_batch,
                args.log_dir,
                init_lock
            )
        )
        p.start()
        workers.append(p)

    print(f"\nAll workers launched. Processing {len(video_files)} videos...")
    print("Monitor progress in worker log files or below:\n")

    # Monitor progress
    start_time = time.time()
    last_count = 0

    try:
        while any(w.is_alive() for w in workers):
            time.sleep(5)  # Update every 5 seconds

            current_count = progress_counter.value
            elapsed = time.time() - start_time

            if current_count > 0:
                videos_per_sec = current_count / elapsed
                eta_seconds = (len(video_files) - current_count) / videos_per_sec if videos_per_sec > 0 else 0
                eta_minutes = eta_seconds / 60

                print(f"Progress: {current_count}/{len(video_files)} videos "
                      f"({current_count/len(video_files)*100:.1f}%) | "
                      f"Rate: {videos_per_sec:.2f} videos/sec | "
                      f"ETA: {eta_minutes:.1f} min")

            last_count = current_count

    except KeyboardInterrupt:
        print("\n\nReceived interrupt signal. Shutting down workers...")
        for w in workers:
            w.terminate()

    # Wait for all workers to finish
    print("\nWaiting for all workers to complete...")
    for w in workers:
        w.join()

    # Final summary
    total_time = time.time() - start_time
    final_count = progress_counter.value

    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total videos processed: {final_count}/{len(video_files)}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average rate: {final_count/total_time:.2f} videos/sec")
    print(f"Worker logs saved to: {args.log_dir}")
    print("="*60)

    if final_count < len(video_files):
        print(f"\nWARNING: Not all videos were processed ({final_count}/{len(video_files)})")
        print("Check worker logs for errors")
        sys.exit(1)


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for better CUDA compatibility
    mp.set_start_method('spawn', force=True)
    main()
