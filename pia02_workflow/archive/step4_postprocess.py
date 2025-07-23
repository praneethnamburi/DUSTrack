from dustrack.dlcinterface import VideoAnnotation
import os


if __name__ == "__main__":

    annotator_name = 'hw'
    
    root_dir = r"C:\Dataset\pia02\021"
    participant_id = '021'

        # Check if the root directory exists
    if not os.path.exists(root_dir):
        print(f"Root directory {root_dir} does not exist")
        exit()

    
    video_path = r'C:\Users\haowe\OneDrive\Desktop\MIT\PianoProject\Code\PianoProjectVenv\data\pia_02\021\pia02_s021_001_LFA_hw-2025-07-22\videos\pia02_s021_001_LFA.mp4'

    # check if the video path is valid
    if not os.path.exists(video_path):
        print(f"Video path {video_path} does not exist")
        exit()

    iteration0_label = r'C:\Users\haowe\OneDrive\Desktop\MIT\PianoProject\Code\PianoProjectVenv\data\pia_02\021\pia02_s021_001_LFA_hw-2025-07-22\videos\iteration-0\pia02_s021_001_LFADLC_resnet50_pia02_s021_001_LFA_hwJul22shuffle1_700.h5'

    # check if the overlay label path is valid
    if not os.path.exists(iteration0_label):
        print(f"Overlay label path {iteration0_label} does not exist")
        exit()

    ann = VideoAnnotation(iteration0_label, video_path)
    ann.postprocess()





