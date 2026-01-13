import os
import pandas as pd
import cv2




def get_video_frame_count(video_path):
    video = cv2.VideoCapture(video_path)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    return frame_count






dataset_root_path = r"\\192.168.1.104\home\piano\data"
model_root_path   = r"\\192.168.1.104\home\piano\DLC_MODELS\participant_models_general"

# find the participant folders in the model_root_path
participant_folders = sorted([
    f for f in os.listdir(model_root_path)
    if f.startswith("s0") and os.path.isdir(os.path.join(model_root_path, f))
])

print(f"Number of participant folders: {len(participant_folders)}")

hands = ["LFA", "RFA"]

for participant_folder in participant_folders:
    participant_id = participant_folder.replace("s", "")  # e.g., s035 -> 035

    original_video_folder_path = os.path.join(dataset_root_path, participant_id, "telemed")
    if not os.path.exists(original_video_folder_path):
        print(f"[SKIP] Original video folder path does not exist: {original_video_folder_path}")
        continue

    original_videos_name = sorted([f for f in os.listdir(original_video_folder_path) if f.lower().endswith(".mp4")])

    for hand in hands:
        model_folder_path = os.path.join(
            model_root_path,
            participant_folder,
            hand,
            "interosseous_pn24-x-2025-10-24"
        )
        if not os.path.exists(model_folder_path):
            print(f"[SKIP] Model folder path does not exist: {model_folder_path}")
            continue

        dlc_videos_folder_path = os.path.join(model_folder_path, "videos")
        if not os.path.exists(dlc_videos_folder_path):
            print(f"[SKIP] DLC videos folder path does not exist: {dlc_videos_folder_path}")
            continue

        dlc_videos_name = sorted([f for f in os.listdir(dlc_videos_folder_path) if f.lower().endswith(".mp4")])

        # sanity check
        if len(dlc_videos_name) != len(original_videos_name):
            print(f"[WARN] #DLC videos ({len(dlc_videos_name)}) != #Original videos ({len(original_videos_name)})")
            print(f"       Participant id: {participant_id}, Hand: {hand}")
        

        # build mapping list
        rows = []
        for i, dlc_video_name in enumerate(dlc_videos_name):
            base = dlc_video_name.replace("_LFA2.mp4", "").replace("_RFA2.mp4", "")
            corresponding = [f for f in original_videos_name if f.startswith(base)]
            dlc_video_path = os.path.join(dlc_videos_folder_path, dlc_video_name)
            dlc_video_frame_count = get_video_frame_count(dlc_video_path)

            if len(corresponding) != 1:
                print(f"[WARN] {participant_id} {hand}: expected 1 match for '{dlc_video_name}', got {len(corresponding)}")
                original_name = "N/A"
            else:
                original_name = corresponding[0]
                original_video_path = os.path.join(original_video_folder_path, original_name)
                # check if their frame count is the same
                original_video_frame_count = get_video_frame_count(original_video_path)
                if dlc_video_frame_count != original_video_frame_count:
                    print(f"[WARN] {participant_id} {hand}: dlc video frame count ({dlc_video_frame_count}) != original video frame count ({original_video_frame_count})")
                    print(f"       DLC video name: {dlc_video_name}, Original video name: {original_name}")
                    print(f"       DLC video frame count: {dlc_video_frame_count}, Original video frame count: {original_video_frame_count}")


            rows.append({
                "dlc_video_name": dlc_video_name,
                "original_video_name": original_name,
                "frame_count": dlc_video_frame_count,
                "dlc_video_index": i,
                "iteraition": pd.NA,  # keep empty (your spelling kept)
                "snapshot": pd.NA,      # keep empty
                "correction": pd.NA,
                "notes": pd.NA
            })

        df = pd.DataFrame(rows, columns=[
            "dlc_video_name",
            "original_video_name",
            "frame_count",
            "dlc_video_index",
            "iteraition",
            "snapshot",
            "correction",
            "notes"
        ])

        # save spreadsheet

        # only create if the file does not exist
        out_csv  = os.path.join(model_root_path, 'spreadsheets', f"{participant_folder}_{hand}_tracking.csv")
        # df.to_csv(out_csv, index=False)
        # df.to_csv(out_csv, index=False)
        if not os.path.exists(out_csv):
            df.to_csv(out_csv, index=False)
            print(f"[OK] Saved: {out_csv}")
        else:
            print(f"[SKIP] File already exists: {out_csv}")

        
            
