import os
import cv2
import numpy as np
from tqdm import tqdm

image_root_dir = r"\\192.168.1.104\home\piano\data\overhead_camera\last_frames"
video_output_path = r"\\192.168.1.104\home\piano\data\overhead_camera\last_frames.mp4"

# collect and sort image files
image_files = sorted(f for f in os.listdir(image_root_dir) if f.endswith(".png"))
print(f"Number of images: {len(image_files)}")

if not image_files:
    raise ValueError("No .png images found in the directory.")

# read the first image to get dimensions
first_img = cv2.imread(os.path.join(image_root_dir, image_files[0]))
height, width = first_img.shape[:2]

# create video writer
fps = 30
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

for fname in tqdm(image_files, desc="Writing video"):
    img = cv2.imread(os.path.join(image_root_dir, fname))
    if img is None:
        print(f"Warning: could not read {fname}, skipping.")
        continue
    writer.write(img)

writer.release()
print(f"Video saved to {video_output_path}")
