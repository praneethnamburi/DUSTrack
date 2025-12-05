import os
import cv2

target_folder = r"\\192.168.1.104\home\piano\pia02_ultrasound_mp4_check"
data_path = r"\\192.168.1.104\home\piano\data"

if __name__ == "__main__":
    print("extract first frame from mp4")

    # loop through all the folders with file name as "001", "002", "003", "004", "005", "006", "007", "008", "009", "010"
    # get all the folders start with "0" and sort them
    participant_folders = [f for f in os.listdir(data_path) if f.startswith("0")]
    participant_folders.sort()

    for participant_folder in participant_folders:
        # print folder name in participant xxx
        print(f"Participant {participant_folder}")
        # append folder "telemed" to the folder name
        participant_telemed_path = os.path.join(data_path, participant_folder, "telemed")

        # check if the participant_telemed_path exists
        if not os.path.exists(participant_telemed_path):
            print(f"Participant {participant_folder} does not have a telemed folder")
            continue

        # get all the mp4 file path in the participant_telemed_path
        mp4_files = [f for f in os.listdir(participant_telemed_path) if f.endswith(".mp4")]
        mp4_files.sort()

        # print number of the mp4 files in participant xxx
        print(f"Number of mp4 files in participant {participant_folder}: {len(mp4_files)}")

        # loop through all the mp4 files
        for mp4_file in mp4_files:
            # print mp4 file name
            print(f"MP4 file: {mp4_file}")
            mp4_file_path = os.path.join(participant_telemed_path, mp4_file)

            cap = cv2.VideoCapture(mp4_file_path)
            ok, frame = cap.read()
            if ok:
                # ignore the .mp4 extension
                mp4_file_name = mp4_file.replace(".mp4", "")
                cv2.imwrite(os.path.join(target_folder, f"{mp4_file_name}.png"), frame)
            else:
                print(f"Failed to read frame from {mp4_file}")
            cap.release()

    print("Done")

        




        
