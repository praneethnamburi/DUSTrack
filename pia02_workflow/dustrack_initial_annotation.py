import datanavigator
import os
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the parent directory to Python path so we can import dustrack
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


from dustrack import DUSTrack, DLCProject


if __name__ == "__main__":
    annotator_name = 'hw'
    
    vpath = r"\\192.168.1.104\home\piano\DLC_MODELS\022\pia02_s022_001_RFA.mp4"

    # Check if the video exists
    if not os.path.exists(vpath):
        print(f"Root directory {vpath} does not exist")
        exit()

    
    d = DUSTrack(vpath, annotator_name)
    
    # Keep the GUI window open  
    plt.show()

