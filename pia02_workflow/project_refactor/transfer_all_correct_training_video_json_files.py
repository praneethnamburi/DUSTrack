import os
import shutil
from tqdm import tqdm

original_folder = r"\\192.168.1.104\home\piano\DLC_MODELS\general\interosseous_pn23-x-2025-09-11\videos"

new_folder = r"\\192.168.1.104\home\piano\DLC_MODELS\general\interosseous_pn24-x-2025-10-24\videos"

# check if the root folder exists
if not os.path.exists(original_folder):
    raise FileNotFoundError(f"Root folder {original_folder} does not exist")

incorrect_participants = [
    "s043",
    "s044",
    "s045",
    "s046",
    "s047",
    "s048",
    "s049",
    "s050",
    "s051"
]


# # get all the files in the original folder as a list
# original_files = [f for f in os.listdir(original_folder) if f.endswith(".mp4")]
# # sort the original files
# original_files.sort()
# print(f"Number of original files: {len(original_files)}")

# # use tqdm to show the progress
# for original_file in tqdm(original_files):
#     # get the original file path
#     original_file_path = os.path.join(original_folder, original_file)
#     # the file name is in format pia02_sxxx_xxx_LFA.mp4 or pia02_sxxx_xxx_RFA.mp4
#     # first the participant id sxxx
#     participant_id = original_file.split("_")[1]
#     if participant_id not in incorrect_participants:
#         # copy the file to the new folder and add 2 to the file name like s043_001_LFA.mp4 -> s043_001_LFA2.mp4
#         new_file_name = original_file.replace(".mp4", "2.mp4")
#         new_file_path = os.path.join(new_folder, new_file_name)
#         shutil.copy(original_file_path, new_file_path)


# json file copying

json_files = [f for f in os.listdir(original_folder) if f.endswith(".json")]
json_files.sort()
# the json file is in the format like this: pia02_s001_007_LFA_annotations_iteration-4.json we need to add 2 as well

for json_file in tqdm(json_files):

    json_file_split = json_file.split("_")
    
    participant_id = json_file_split[1]

    if participant_id not in incorrect_participants:
        # add 2 to the LFA or RFA
        json_file_split[3] = json_file_split[3] + "2"
        
        new_json_file = "_".join(json_file_split)

        new_json_file_path = os.path.join(new_folder, new_json_file)
        shutil.copy(os.path.join(original_folder, json_file), new_json_file_path)
        # print(f"Copying {os.path.join(original_folder, json_file)} to {new_json_file_path}")
        

        






