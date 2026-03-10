import midi
import overhead_camera
import os

disklavier_folder_root = r"\\192.168.1.104\home\piano\data\009\disklavier"
overhead_camera_folder_root = r"\\192.168.1.104\home\piano\data\009\overhead camera"

# get list of .mid files from the disklavier_folder_root
disklavier_midi_files = [f for f in os.listdir(disklavier_folder_root) if f.endswith(".mid")]

overhead_camera_mp4_files = [f for f in os.listdir(overhead_camera_folder_root) if f.endswith(".MP4")]
overhead_camera_xml_files = [f for f in os.listdir(overhead_camera_folder_root) if f.endswith(".XML")]

first_disklavier_midi_file = disklavier_midi_files[0]
first_disklavier_midi_file_path = os.path.join(disklavier_folder_root, first_disklavier_midi_file)
first_disklavier_midi = midi.Log(first_disklavier_midi_file_path)
first_overhead_camera_mp4 = overhead_camera.OverheadCamera(os.path.join(overhead_camera_folder_root, overhead_camera_xml_files[0]), os.path.join(overhead_camera_folder_root, overhead_camera_mp4_files[0]))

print(first_disklavier_midi.get_recording_time_range("datetime"))
print(first_overhead_camera_mp4.get_recording_time_range("datetime"))


