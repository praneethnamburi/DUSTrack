"""
Configuration file for DUSTrack.

This file contains global settings used throughout the DUSTrack application,
particularly for DeepLabCut project management and NAS connectivity.
"""

# Experimenter name used when creating DeepLabCut projects.
# This identifier is embedded in project paths and configuration files.
EXPERIMENTER = "x"

# IP address of the Network Attached Storage (NAS) device.
# Used to update video paths in DeepLabCut projects when accessing
# files from different machines or network locations.
# Set to None (default) if the access path does not change.
NAS_IP = None

# For DeepLabCut 3.x: whether to use the last trained snapshot instead of
# the snapshot marked as "best" during evaluation.
# - True: Use the most recent snapshot (last training iteration)
# - False: Use the snapshot with the lowest test error (best performance)
DLC3_USE_LAST_SNAPSHOT = True
