import sys
sys.path.insert(0, r"C:\Users\mitim\Desktop\MITHIC\code\DUSTrack")
from pia02_workflow.overhead_camera_midi_alignment import midi

# add 

midi_file_path = r"\\192.168.1.104\home\piano\data\045\disklavier\20250815_144525_pia02_s045_007_tempo_ramp_ap345.mid"
midi_log = midi.Log(midi_file_path)
midi_log.show_roll()