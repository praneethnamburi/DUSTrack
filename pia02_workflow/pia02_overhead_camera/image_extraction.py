import os
from pathlib import Path

import cv2


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

        for i, mp4_file in enumerate(mp4_files):
            save_path = os.path.join(image_save_dir, f"{participant_folder.name}-{i}-{mp4_file.stem}.png")
            # if already exists, skip
            if os.path.exists(save_path):
                print(f"  [{i}] {mp4_file.name}: already exists, skipping")
                continue

            cap = cv2.VideoCapture(str(mp4_file))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0:
                print(f"  [{i}] {mp4_file.name}: no frames found, skipping")
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 10)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                print(f"  [{i}] {mp4_file.name}: failed to read last frame")
                continue


            cv2.imwrite(save_path, frame)
            print(f"  [{i}] {mp4_file.name} -> {save_path}")

    else:
        print(f"Overhead camera folder does not exist for participant {participant_folder.name}")
    print("--------------------------------")
    print()


