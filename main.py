import os
import sys

def check_path_exists(path):
    """Check if a file path exists on the computer."""
    if os.path.exists(path):
        print(f"✅ Path exists: {path}")
        return True
    else:
        print(f"❌ Path does not exist: {path}")
        return False

if __name__ == "__main__":
    # Check if the specified path exists
    target_path = r"M:\us_videos_for_tracking"
    check_path_exists(target_path)




