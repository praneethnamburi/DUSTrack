"""
Configuration file for DUSTrack.

This file contains global settings used throughout the DUSTrack application,
particularly for DeepLabCut project management.
"""

# Experimenter name used when creating DeepLabCut projects.
# This identifier is embedded in project paths and configuration files.
EXPERIMENTER = "x"

# For DeepLabCut 3.x: whether to use the last trained snapshot instead of
# the snapshot marked as "best" during evaluation.
# - True: Use the most recent snapshot (last training iteration)
# - False: Use the snapshot with the lowest test error (best performance)
DLC3_USE_LAST_SNAPSHOT = True