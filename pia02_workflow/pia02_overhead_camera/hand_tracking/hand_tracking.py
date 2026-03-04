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

# MediaPipe hand skeleton connections (21-landmark topology)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm base chain
    (5, 9), (9, 13), (13, 17),
]


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


def save_overlay_video(video_path: str) -> Path:
    """Save a copy of the video with detected hand landmarks overlaid.

    Reads the saved .pt file from the pose/ subdirectory to get landmark data.

    Args:
        video_path: Path to the original video file.

    Returns:
        Path to the saved overlay video.
    """
    video = Path(video_path)
    pt_path = video.parent / "pose" / f"{video.stem}.pt"
    data = torch.load(pt_path, weights_only=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = video.parent / "pose" / f"{video.stem}_hands.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    left_hand = data["left_hand"].numpy()   # (N, 21, 3)
    right_hand = data["right_hand"].numpy()  # (N, 21, 3)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_idx in tqdm(range(total_frames), desc="Saving overlay video"):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= len(left_hand):
            break

        for landmarks, color in [
            (left_hand[frame_idx], (0, 255, 0)),   # green for left
            (right_hand[frame_idx], (255, 0, 0)),   # blue for right
        ]:
            if np.allclose(landmarks, 0):
                continue
            # Convert normalized coords to pixel coords
            pts = np.column_stack([
                (landmarks[:, 0] * w).astype(int),
                (landmarks[:, 1] * h).astype(int),
            ])
            for start, end in HAND_CONNECTIONS:
                cv2.line(frame, tuple(pts[start]), tuple(pts[end]), color, 2)
            for x, y in pts:
                cv2.circle(frame, (x, y), 4, color, -1)

        writer.write(frame)

    cap.release()
    writer.release()
    return output_path


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

    video_path = r"\\192.168.1.104\home\piano\data\overhead_camera\hand_tracking\videos\fx30_2_0587.MP4"
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

    overlay_path = save_overlay_video(video_path)
    print(f"  Overlay video: {overlay_path}")


if __name__ == "__main__":
    main()
