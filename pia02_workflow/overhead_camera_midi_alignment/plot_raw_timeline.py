import midi
import os

disklavier_folder_root = r"\\192.168.1.104\home\piano\data\009\disklavier"
overhead_camera_folder_root = r"\\192.168.1.104\home\piano\data\009\overhead camera"

# get list of .mid files from the disklavier_folder_root
disklavier_midi_files = [f for f in os.listdir(disklavier_folder_root) if f.endswith(".mid")]

overhead_camera_mp4_files = [f for f in os.listdir(overhead_camera_folder_root) if f.endswith(".MP4")]





