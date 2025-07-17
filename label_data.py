import datanavigator
import os
import matplotlib.pyplot as plt
from datanavigator import VideoPointAnnotator, VideoAnnotation
from dustrack import DUSTrack, DLCProject

if __name__ == "__main__":
    annotator_name = 'hw'

    vpath = r'M:\DLC_MODELS\021\pia02_s021_001_LFA.mp4'

    # check if the video path is valid
    if not os.path.exists(vpath):
        print(f"Video path {vpath} does not exist")
        exit()

    d = DUSTrack(vpath, "hw")
    
    # Keep the GUI window open
    plt.show()
    


