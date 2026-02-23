"""Hand landmark detection using the MediaPipe Tasks API.

Processes a video file, detects hand landmarks in each frame, and saves
the results as a .pt file containing left/right hand landmark tensors.

Usage:
    python hand_tracking.py <video_path>
"""

import argparse
import os
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from tqdm import tqdm

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

NUM_LANDMARKS = 21  # MediaPipe hand model outputs 21 landmarks per hand


def ensure_model():
    """Download the hand_landmarker.task model if it doesn't exist locally."""
    if MODEL_PATH.exists():
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand_landmarker.task to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")


def detect_hands(video_path: str) -> dict:
    """Detect hand landmarks in every frame of a video.

    Args:
        video_path: Path to an .mp4 video file.

    Returns:
        Dictionary with keys:
            left_hand  - torch.Tensor of shape (num_frames, 21, 3)
            right_hand - torch.Tensor of shape (num_frames, 21, 3)
            fps        - float, video frame rate
            frame_count - int, total frames processed
            video_path - str, original video path
    """
    ensure_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval_ms = 1000.0 / fps if fps > 0 else 33.33

    # Configure HandLandmarker in VIDEO mode
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    left_hand_frames = []
    right_hand_frames = []

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        for frame_idx in tqdm(range(total_frames), desc="Detecting hands"):
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(frame_idx * frame_interval_ms)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            left = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
            right = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)

            if result.hand_landmarks:
                for hand_landmarks, handedness in zip(
                    result.hand_landmarks, result.handedness
                ):
                    coords = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                        dtype=np.float32,
                    )
                    label = handedness[0].category_name
                    if label == "Left":
                        left = coords
                    elif label == "Right":
                        right = coords

            left_hand_frames.append(left)
            right_hand_frames.append(right)

    cap.release()

    return {
        "left_hand": torch.tensor(np.stack(left_hand_frames)),
        "right_hand": torch.tensor(np.stack(right_hand_frames)),
        "fps": fps,
        "frame_count": len(left_hand_frames),
        "video_path": video_path,
    }


def save_results(data: dict, video_path: str) -> Path:
    """Save detection results as a .pt file in a pose/ subdirectory.

    The output mirrors the input directory structure:
        <video_dir>/pose/<video_stem>.pt
    """
    video = Path(video_path)
    output_dir = video.parent / "pose"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video.stem}.pt"
    torch.save(data, output_path)
    return output_path


def main():
    # parser = argparse.ArgumentParser(description="Detect hand landmarks in a video.")
    # parser.add_argument("video_path", help="Path to an .mp4 video file")
    # args = parser.parse_args()

    video_path = r"\\192.168.1.104\home\piano\data\overhead_camera\hand_tracking\videos\fx30_2_0894.MP4"
    if not os.path.isfile(video_path):
        print(f"Error: file not found: {video_path}")
        return

    print(f"Processing: {video_path}")
    data = detect_hands(video_path)
    output_path = save_results(data, video_path)

    print(f"\nDone — {data['frame_count']} frames processed")
    print(f"  left_hand  : {data['left_hand'].shape}")
    print(f"  right_hand : {data['right_hand'].shape}")
    print(f"  Saved to   : {output_path}")


if __name__ == "__main__":
    main()
