import os
from datetime import datetime, timezone, timedelta

import matplotlib.pyplot as plt

import midi
import overhead_camera

disklavier_folder_root = r"\\192.168.1.104\home\piano\data\011\disklavier"
overhead_camera_folder_root = r"\\192.168.1.104\home\piano\data\011\overhead camera"

# Discover files (sorted for correct XML/MP4 pairing)
disklavier_midi_files = sorted([f for f in os.listdir(disklavier_folder_root) if f.endswith(".mid")])
overhead_camera_mp4_files = sorted([f for f in os.listdir(overhead_camera_folder_root) if f.endswith(".MP4")])
overhead_camera_xml_files = sorted([f for f in os.listdir(overhead_camera_folder_root) if f.endswith(".XML")])

# Collect MIDI time ranges
midi_time_ranges = []
for f in disklavier_midi_files:
    path = os.path.join(disklavier_folder_root, f)
    log = midi.Log(path)
    start, end, duration = log.get_recording_time_range("unix", utc_offset=-5)
    midi_time_ranges.append((start, duration))

# Collect overhead camera time ranges
camera_time_ranges = []
for xml_file, mp4_file in zip(overhead_camera_xml_files, overhead_camera_mp4_files):
    xml_path = os.path.join(overhead_camera_folder_root, xml_file)
    mp4_path = os.path.join(overhead_camera_folder_root, mp4_file)
    cam = overhead_camera.OverheadCamera(xml_path, mp4_path)
    start, end, duration = cam.get_recording_time_range("unix")
    camera_time_ranges.append((start, duration))

# Plot timeline
fig, ax = plt.subplots(figsize=(14, 3))
ax.broken_barh(midi_time_ranges, (1, 0.8), facecolors='tab:blue')
ax.broken_barh(camera_time_ranges, (0, 0.8), facecolors='tab:orange')
ax.set_yticks([0.4, 1.4])
ax.set_yticklabels(['Overhead Camera', 'MIDI'])
ax.set_xlabel('Unix Time (s)')
ax.set_title('Recording Timeline')
plt.tight_layout()
plt.show()
