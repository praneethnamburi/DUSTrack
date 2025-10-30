import os
import numpy as np

splited_video_folder = r"\\192.168.1.104\home\piano\us_videos_for_tracking"
pia02_ultrasound_mp4_check_folder = r"\\192.168.1.104\home\piano\pia02_ultrasound_mp4_check"

# get all the files in the original folder as a list
original_files = [f for f in os.listdir(splited_video_folder) if f.endswith(".mp4")]
print(f"Number of original files: {len(original_files)}")

# get all the files in the pia02_ultrasound_mp4_check folder as a list
pia02_ultrasound_mp4_check_files = [f for f in os.listdir(pia02_ultrasound_mp4_check_folder) if f.endswith(".png")]
print(f"Number of pia02_ultrasound_mp4_check files: {len(pia02_ultrasound_mp4_check_files)}")

# get a list of file prefix from the pia02_ultrasound_mp4_check_files
file_prefixes = [f[:14] for f in pia02_ultrasound_mp4_check_files]

# check if there is any duplicate in the file_prefixes list and print the duplicate file prefixes
if len(file_prefixes) != len(set(file_prefixes)):
    duplicate_file_prefixes = [f for f in file_prefixes if file_prefixes.count(f) > 1]
    print(f"Duplicate file prefixes: {duplicate_file_prefixes}")
    exit()













