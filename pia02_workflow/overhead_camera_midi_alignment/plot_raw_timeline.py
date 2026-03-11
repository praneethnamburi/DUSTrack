import os
from datetime import datetime, timezone, timedelta

import matplotlib.pyplot as plt

import midi
import overhead_camera

disklavier_folder_root = r"\\192.168.1.104\home\piano\data\061\disklavier"
overhead_camera_folder_root = r"\\192.168.1.104\home\piano\data\061\overhead camera"

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
    midi_time_ranges.append((f, start, duration))

# Collect overhead camera time ranges
camera_time_ranges = []
for xml_file, mp4_file in zip(overhead_camera_xml_files, overhead_camera_mp4_files):
    xml_path = os.path.join(overhead_camera_folder_root, xml_file)
    mp4_path = os.path.join(overhead_camera_folder_root, mp4_file)
    cam = overhead_camera.OverheadCamera(xml_path, mp4_path)
    start, end, duration = cam.get_recording_time_range("unix")
    camera_time_ranges.append((mp4_file, start, duration))

# Plot timelines
n_midi = len(midi_time_ranges)
n_camera = len(camera_time_ranges)
fig_height = 2 + 0.5 * max(n_midi, n_camera)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, fig_height))

# Subplot 1: MIDI files
for i, (name, start, duration) in enumerate(midi_time_ranges):
    ax1.broken_barh([(start, duration)], (i, 0.8), facecolors='tab:blue')
ax1.set_yticks([i + 0.4 for i in range(n_midi)])
ax1.set_yticklabels([name for name, _, _ in midi_time_ranges])
ax1.set_xlabel('Unix Time (s)')
ax1.set_title('MIDI Recording Timeline')

# Subplot 2: Overhead Camera files
for i, (name, start, duration) in enumerate(camera_time_ranges):
    ax2.broken_barh([(start, duration)], (i, 0.8), facecolors='tab:orange')
ax2.set_yticks([i + 0.4 for i in range(n_camera)])
ax2.set_yticklabels([name for name, _, _ in camera_time_ranges])
ax2.set_xlabel('Unix Time (s)')
ax2.set_title('Overhead Camera Recording Timeline')

plt.tight_layout()
plt.show()
