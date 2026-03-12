import os
from datetime import datetime, timezone, timedelta

import matplotlib.pyplot as plt

import midi
import overhead_camera

participants_root = r"\\192.168.1.104\home\piano\data"
output_folder = r"\\192.168.1.104\home\piano\data\overhead_camera\midi_overheadcam_sync\raw_sync_plots"
os.makedirs(output_folder, exist_ok=True)

for participant_id in range(1, 62):
    pid = f"{participant_id:03d}"
    disklavier_folder = os.path.join(participants_root, pid, "disklavier")
    overhead_camera_folder = os.path.join(participants_root, pid, "overhead camera")

    # Check if both folders exist
    if not os.path.isdir(disklavier_folder):
        print(f"[{pid}] Skipping — missing disklavier folder")
        continue
    if not os.path.isdir(overhead_camera_folder):
        print(f"[{pid}] Skipping — missing overhead camera folder")
        continue

    try:
        # Discover files (sorted for correct XML/MP4 pairing)
        disklavier_midi_files = sorted([f for f in os.listdir(disklavier_folder) if f.endswith(".mid")])
        overhead_camera_mp4_files = sorted([f for f in os.listdir(overhead_camera_folder) if f.endswith(".MP4")])
        overhead_camera_xml_files = sorted([f for f in os.listdir(overhead_camera_folder) if f.endswith(".XML")])

        if not disklavier_midi_files:
            print(f"[{pid}] Skipping — no .mid files found")
            continue
        if not overhead_camera_mp4_files or not overhead_camera_xml_files:
            print(f"[{pid}] Skipping — no .MP4 or .XML files found")
            continue

        # Collect MIDI time ranges
        midi_time_ranges = []
        for f in disklavier_midi_files:
            path = os.path.join(disklavier_folder, f)
            log = midi.Log(path)
            start, end, duration = log.get_recording_time_range("unix", utc_offset=-5)
            midi_time_ranges.append((start, duration))

        # Collect overhead camera time ranges
        camera_time_ranges = []
        for xml_file, mp4_file in zip(overhead_camera_xml_files, overhead_camera_mp4_files):
            xml_path = os.path.join(overhead_camera_folder, xml_file)
            mp4_path = os.path.join(overhead_camera_folder, mp4_file)
            cam = overhead_camera.OverheadCamera(xml_path, mp4_path)
            start, end, duration = cam.get_recording_time_range("unix")
            camera_time_ranges.append((start, duration))

        # Align to first MIDI start time (t=0)
        t0 = midi_time_ranges[0][0]
        midi_time_ranges = [(s - t0, d) for s, d in midi_time_ranges]
        camera_time_ranges = [(s - t0, d) for s, d in camera_time_ranges]

        # Plot timeline
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.broken_barh(midi_time_ranges, (1, 0.8), facecolors='tab:blue')
        ax.broken_barh(camera_time_ranges, (0, 0.8), facecolors='tab:orange')
        ax.set_yticks([0.4, 1.4])
        ax.set_yticklabels(['Overhead Camera', 'MIDI'])
        ax.set_xlabel('Time (s)')
        ax.set_title(f'Recording Timeline — Participant {pid}')
        plt.tight_layout()

        output_path = os.path.join(output_folder, f"{pid}_raw_sync.png")
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"[{pid}] Saved: {output_path}")

    except Exception as e:
        print(f"[{pid}] Error: {e}")
