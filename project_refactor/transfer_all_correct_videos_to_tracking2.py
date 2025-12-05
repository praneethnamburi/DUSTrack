import os
import shutil
from tqdm import tqdm

original_folder = r"\\192.168.1.104\home\piano\us_videos_for_tracking"
new_folder = r"\\192.168.1.104\home\piano\us_videos_for_tracking2"

incorrect_participants = [
    "s043",
    "s044",
    "s045",
    "s046",
    "s047",
    "s048",
    "s049",
    "s050",
]

# get all the files in the original folder as a list
original_files = [f for f in os.listdir(original_folder) if f.endswith(".mp4")]
# sort the original files
original_files.sort()
print(f"Number of original files: {len(original_files)}")

# use tqdm to show the progress
for original_file in tqdm(original_files):
    # get the original file path
    original_file_path = os.path.join(original_folder, original_file)
    # the file name is in format pia02_sxxx_xxx_LFA.mp4 or pia02_sxxx_xxx_RFA.mp4
    # first the participant id sxxx
    participant_id = original_file.split("_")[1]
    if participant_id not in incorrect_participants:
        # copy the file to the new folder and add 2 to the file name like s043_001_LFA.mp4 -> s043_001_LFA2.mp4
        new_file_name = original_file.replace(".mp4", "2.mp4")
        new_file_path = os.path.join(new_folder, new_file_name)
        shutil.copy(original_file_path, new_file_path)


