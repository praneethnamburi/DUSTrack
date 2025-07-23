import datanavigator
import os
import matplotlib.pyplot as plt
from dustrack import DUSTrack, DLCProject


if __name__ == "__main__":
    annotator_name = 'hw'
    
    root_dir = r"C:\Users\haowe\OneDrive\Desktop\MIT\PianoProject\Code\PianoProjectVenv\data\pia_02"
    # this would be the path to the DLC_MODELS folder for example: M:\DLC_MODELS

    participant_id = '021'
    video_name = 'pia02_s021_013_LFA.mp4'

    # Check if the root directory exists
    if not os.path.exists(root_dir):
        print(f"Root directory {root_dir} does not exist")
        exit()

    vpath = os.path.join(root_dir, participant_id, video_name)


    # check if the video path is valid
    if not os.path.exists(vpath):
        print(f"Video path {vpath} does not exist")
        exit()
    
    d = DUSTrack(vpath, annotator_name)
    
    # Keep the GUI window open
    plt.show()

