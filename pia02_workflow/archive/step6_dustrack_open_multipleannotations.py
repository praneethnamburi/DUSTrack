

import datanavigator
import os
import matplotlib.pyplot as plt
from dustrack import DUSTrack, DLCProject


if __name__ == "__main__":
    
    root_dir = r"C:\Users\haowe\OneDrive\Desktop\MIT\PianoProject\Code\PianoProjectVenv\data\pia_02"
    participant_id = '021'
    video_name = 'pia02_s021_013_LFA.mp4'
    annotation_dir = r"C:\Users\haowe\OneDrive\Desktop\MIT\PianoProject\Code\PianoProjectVenv\data\pia_02\021"

    # Check if the root directory exists
    if not os.path.exists(root_dir):
        print(f"Root directory {root_dir} does not exist")
        exit()

    vpath = os.path.join(root_dir, participant_id, video_name)

    # check if the video path is valid
    if not os.path.exists(vpath):
        print(f"Video path {vpath} does not exist")
        exit()

    # Define the annotation layers to load
    annotation_layers = {
        'hw': os.path.join(annotation_dir, 'pia02_s021_013_LFA_annotations_hw.json'),
        'iteration0': os.path.join(annotation_dir, 'pia02_s021_013_LFA_annotations_iteration0.json')
    }
    
    # Check if annotation files exist
    for layer_name, file_path in annotation_layers.items():
        if not os.path.exists(file_path):
            print(f"Annotation file {file_path} does not exist")
            exit()
        else:
            print(f"Found annotation layer '{layer_name}': {file_path}")

    # Create DUSTrack with multiple annotation layers        
    d = DUSTrack(vpath, annotation_layers)
    
    # Keep the GUI window open
    plt.show()


   
