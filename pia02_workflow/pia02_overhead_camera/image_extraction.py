import csv
import os
from pathlib import Path

import cv2


def extract_frame(video_path, frame_index, save_path):
    """Extract a single frame from a video and save it as an image.

    Returns True on success, False on failure.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            return False
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            return False
        cv2.imwrite(str(save_path), frame)
        return True
    finally:
        cap.release()


participant_root_dir = r"\\192.168.1.104\home\piano\data"
participant_root_dir = Path(participant_root_dir)

image_save_dir = r"\\192.168.1.104\home\piano\data\overhead_camera\extracted_frames"
# check if the dir exists, if not, break
if not os.path.exists(image_save_dir):
    raise FileNotFoundError(f"Image save directory not found: {image_save_dir}")

participant_folders = sorted(
    [p for p in participant_root_dir.iterdir() if p.is_dir() and p.name.isdigit()],
    key=lambda p: int(p.name),
)

print(f"There are {len(participant_folders)} participant folders")

for participant_folder in participant_folders:
    print(f"Checking participant {participant_folder.name}")
    print("--------------------------------")
    overhead_camera_folder = participant_folder / "overhead camera"
    if overhead_camera_folder.exists():
        mp4_files = sorted(overhead_camera_folder.glob("*.MP4"))
        print(f"Found {len(mp4_files)} .MP4 files")

        participant_save_dir = os.path.join(image_save_dir, participant_folder.name)
        os.makedirs(participant_save_dir, exist_ok=True)

        for i, mp4_file in enumerate(mp4_files):
            cap = cv2.VideoCapture(str(mp4_file))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if frame_count == 0:
                print(f"  [{i}] {mp4_file.name}: no frames found, skipping")
                continue

            for frame_index in [10, frame_count - 10]:
                save_path = os.path.join(participant_save_dir, f"{participant_folder.name}-{i}-{frame_index}-{mp4_file.stem}.png")
                # if already exists, skip
                if os.path.exists(save_path):
                    print(f"  [{i}] {mp4_file.name} frame {frame_index}: already exists, skipping")
                    continue

                success = extract_frame(mp4_file, frame_index, save_path)
                if not success:
                    print(f"  [{i}] {mp4_file.name} frame {frame_index}: failed to read frame")
                    continue

                print(f"  [{i}] {mp4_file.name} frame {frame_index} -> {save_path}")

        # Write CSV for this participant
        csv_path = os.path.join(participant_save_dir, f"{participant_folder.name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["participant_id", "video_name", "reference_image", "camera_shifted_during_trial", "note"])
            for mp4_file in mp4_files:
                writer.writerow([participant_folder.name, mp4_file.name, "", "", ""])
        print(f"  CSV saved: {csv_path}")

    else:
        print(f"Overhead camera folder does not exist for participant {participant_folder.name}")
    print("--------------------------------")
    print()


