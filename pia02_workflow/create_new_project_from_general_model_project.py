import datanavigator
import os
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import deeplabcut
# Add the parent directory to Python path so we can import dustrack
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
import numpy as np
import shutil
from dustrack import DUSTrack, DLCProject
from ruamel.yaml.comments import CommentedMap

if __name__ == "__main__":

    
    # participant_ids are a list of participant id with format 001, 002, 003, 004, etc
    video_root_path = r'\\192.168.1.104\home\piano\us_videos_for_tracking2'
    participant_models_general_path = r'\\192.168.1.104\home\piano\DLC_MODELS\participant_models_general'
    general_model_project_path = r'\\192.168.1.104\home\piano\DLC_MODELS\general\interosseous_pn24-x-2025-10-24'
    

    # get all .mp4 files names:
    video_files = [f for f in os.listdir(video_root_path) if f.endswith('.mp4')]

    # the video files are in the format: pia02_s001_000_LFA2
    # we need a dictionary with format:
    # {
    #     '001': {
    #         'LFA': ['pia02_s001_000_LFA2.mp4', 'pia02_s001_001_LFA2.mp4', 'pia02_s001_002_LFA2.mp4'],
    #         'RFA': ['pia02_s001_000_RFA2.mp4', 'pia02_s001_001_RFA2.mp4', 'pia02_s001_002_RFA2.mp4']
    #     },
    #     '002': {
    #         'LFA': ['pia02_s002_000_LFA2.mp4', 'pia02_s002_001_LFA2.mp4', 'pia02_s002_002_LFA2.mp4'],
    #         'RFA': ['pia02_s002_000_RFA2.mp4', 'pia02_s002_001_RFA2.mp4', 'pia02_s002_002_RFA2.mp4']
    #     },
    # }
    video_dict = {}
    for video_file in video_files:
        # get the participant id from the video file name
        participant_id = video_file.split('_')[1]
        if participant_id not in video_dict:
            video_dict[participant_id] = {
                'LFA': [],
                'RFA': []
            }
        if video_file.endswith('LFA2.mp4'):
            video_dict[participant_id]['LFA'].append(video_file)
        elif video_file.endswith('RFA2.mp4'):
            video_dict[participant_id]['RFA'].append(video_file)
        else:
            print(f"Video file {video_file} is not in the correct format")
            exit()
    print(f"Video dictionary: {video_dict}")

    for participant_id, participant_video_dict in video_dict.items():
        # check if a folder with name {participant_id} exists in the participant_models_general_path
        # if not, create a folder with name {participant_id}
        participant_working_directory = os.path.join(participant_models_general_path, participant_id)
        if not os.path.exists(participant_working_directory):
            os.makedirs(participant_working_directory)
        for hand, videos in participant_video_dict.items():
            print(f"Participant {participant_id} and hand {hand} has {len(videos)} videos")
            print(f"Videos: {videos}")
            hand_working_directory = os.path.join(participant_working_directory, hand)
            if not os.path.exists(hand_working_directory):
                os.makedirs(hand_working_directory)
            # now we need to copy things from the general model project to the hand_working_directory: hand_working_directory/interosseous_pn24-x-2025-10-24
            # get general_model_project_path folder name
            general_model_project_name = os.path.basename(general_model_project_path)
            model_project_path = os.path.join(hand_working_directory, general_model_project_name)
            # if already exist, skip
            if os.path.exists(model_project_path):
                print(f"Participant {participant_id} and hand {hand} already has a model project")
                continue
            # create folder with name {general_model_project_name} in the hand_working_directory
            os.makedirs(model_project_path)


            # 1. copy \\192.168.1.104\home\piano\DLC_MODELS\general\interosseous_pn24-x-2025-10-24/dlc-models directory
            dlc_models_directory = os.path.join(general_model_project_path, 'dlc-models')
            # copy the dlc-models directory to the model_project_path
            shutil.copytree(dlc_models_directory, os.path.join(model_project_path, 'dlc-models'))
            # 2. copy \\192.168.1.104\home\piano\DLC_MODELS\general\interosseous_pn24-x-2025-10-24/dlc-models-pytorch
            dlc_models_pytorch_directory = os.path.join(general_model_project_path, 'dlc-models-pytorch')
            # copy the dlc-models-pytorch directory to the model_project_path
            shutil.copytree(dlc_models_pytorch_directory, os.path.join(model_project_path, 'dlc-models-pytorch'))
            # 3. copy evaluation-results-pytorch
            evaluation_results_pytorch_directory = os.path.join(general_model_project_path, 'evaluation-results-pytorch')
            # copy the evaluation-results-pytorch directory to the model_project_path
            shutil.copytree(evaluation_results_pytorch_directory, os.path.join(model_project_path, 'evaluation-results-pytorch'))
            # 4. copy training-datasets
            training_datasets_directory = os.path.join(general_model_project_path, 'training-datasets')
            # copy the training-datasets directory to the model_project_path
            shutil.copytree(training_datasets_directory, os.path.join(model_project_path, 'training-datasets'))
            # 5. create a labeled-data folder
            labeled_data_directory = os.path.join(model_project_path, 'labeled-data')
            # create the labeled-data folder
            os.makedirs(labeled_data_directory)

            # copy the entire video directory from the general_model_project_path to the model_project_path with filter: [f for f in os.listdir(source_iteration_0_directory) if f.startswith(f'pia02_{participant_id}_') and f.split('_')[3].startswith(hand)]
            source_video_directory = os.path.join(general_model_project_path, 'videos')
            dst_video_directory = os.path.join(model_project_path, 'videos')
            # make directory if not exists
            # os.makedirs(dst_video_directory, exist_ok=True)
            for dp, _, files in os.walk(source_video_directory):
                rel = os.path.relpath(dp, source_video_directory)
                out_dir = os.path.join(dst_video_directory, rel)
                os.makedirs(out_dir, exist_ok=True)
                for f in files:
                    if f.startswith(f"pia02_{participant_id}_") and f.split("_")[3].startswith(hand):
                        shutil.copy2(os.path.join(dp, f), os.path.join(out_dir, f))
            
                        # copy the config.yaml from the general_model_project_path to the model_project_path
            config_yaml_file = os.path.join(general_model_project_path, 'config.yaml')
            shutil.copy(config_yaml_file, os.path.join(model_project_path, 'config.yaml'))

            # first remove all the videos from the video_sets in the config.yamlm, make it a empty CommentedMap
            deeplabcut.auxiliaryfunctions.edit_config(os.path.join(model_project_path, 'config.yaml'), edits={'video_sets': {}})

            # add videos for this specific participant and hand based on the videos list with root path: video_root_path
            videos_path = [os.path.join(video_root_path, f) for f in videos]
            # sort the videos_to_add
            videos_path.sort()
            deeplabcut.add_new_videos(os.path.join(model_project_path, 'config.yaml'), videos_path, copy_videos=True)

            # print success message
            print(f"Successfully created a new project for participant {participant_id} and hand {hand}")
            



            

            


        
            
    # participant_ids = np.arange(0, 56)

    # participant_ids = [f'0{participant_id:02d}' for participant_id in participant_ids]
    # print(f"Number of participant ids: {len(participant_ids)}")
    # # print all participant ids
    # print(f"Participant ids: {participant_ids}")


    # hands = ['LFA', 'RFA']

    # annotator_name = 'x' # modify the name to the annotator name
    
    # video_root_path = r'\\192.168.1.104\home\piano\us_videos_for_tracking2'
    # root_participant_working_directory = r'\\192.168.1.104\home\piano\DLC_MODELS\participant_models'

    # # check if the video root path exists
    # if not os.path.exists(video_root_path):
    #     print(f"Video root path {video_root_path} does not exist")
    #     exit()

    # # check if the video root path is a directory
    # if not os.path.isdir(video_root_path):
    #     print(f"Video root path {video_root_path} is not a directory")
    #     exit()

    # for participant_id in participant_ids:

    #     for hand in hands:

    #         # get any .mp4 file with format: pia02_s{participant_id}_xxx_{hand}2.mp4
    #         # start with pia02_s{participant_id}_ and end with _{hand}2.mp4
    #         video_files = [f for f in os.listdir(video_root_path) if f.startswith(f'pia02_s{participant_id}_') and f.endswith(f'_{hand}2.mp4')]
    #         # sort the video file strings
    #         video_files.sort()
            
    #         if len(video_files) == 0:
    #             print(f"No video files found for participant {participant_id} and hand {hand}")
    #             exit()

    #         video_paths = [os.path.join(video_root_path, f) for f in video_files]
    #         project_name = f'{participant_id}_{hand}'

    #         # create folder name with participant id:
    #         participant_working_directory = os.path.join(root_participant_working_directory, f'{participant_id}')
    #         if not os.path.exists(participant_working_directory):
    #             os.makedirs(participant_working_directory)

    #         project_config_path = deeplabcut.create_new_project(
    #         project_name,
    #         annotator_name,
    #         video_paths,
    #         working_directory=participant_working_directory,
    #         copy_videos=True,
    #         multianimal=False
    #         )

    #         # create a folder with name dlc-models-pytorch in the parent of the config file
    #         dlc_models_pytorch_folder = os.path.join(os.path.dirname(project_config_path), 'dlc-models-pytorch')
    #         if not os.path.exists(dlc_models_pytorch_folder):
    #             os.makedirs(dlc_models_pytorch_folder)

    #         # edit config file, make the bodypart is ['point0', 'point1'] and skeleton is None
    #         config_dict = {'bodyparts': ['point0', 'point1'], 'skeleton': None}
    #         deeplabcut.auxiliaryfunctions.edit_config(project_config_path, config_dict)
        