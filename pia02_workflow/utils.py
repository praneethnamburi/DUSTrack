import os
import shutil
from pathlib import Path
from typing import Union
import deeplabcut

def dlc_edit_config(config_file, **kwargs):
    """Edit the configuration file.
    
    Args:
        config_file (str): Path to the configuration file to edit.
        **kwargs: Additional keyword arguments to pass to the config editor.
        
    Returns:
        The result of deeplabcut.auxiliaryfunctions.edit_config()
    """
    assert os.path.exists(config_file), f"Config file does not exist: {config_file}"
    return deeplabcut.auxiliaryfunctions.edit_config(config_file, kwargs)


def copy_annotations(video_name: Union[Path, list], config_path: str, annotation_suffix: str = ""):
    """Copy annotation files from video location to DLC project folder.
    
    If frames were labeled using VideoPointAnnotator, then copy those files into the DLC project folder as well.
    Copies all annotation files that match the video stem pattern.
    
    Args:
        video_name (Union[Path, list]): Path to video file(s) or list of video paths.
        config_path (str): Path to the DLC config file.
        annotation_suffix (str, optional): Suffix to append to annotation file names. Defaults to "".
        
    Returns:
        Union[list, None]: List of copied annotation file paths, or None if no annotation files exist.
    """
    if isinstance(video_name, list):
        all_copied_files = []
        for this_video_name in video_name:
            copied_files = copy_annotations(this_video_name, config_path, annotation_suffix)
            if copied_files is not None:
                all_copied_files.extend(copied_files)
        return all_copied_files if all_copied_files else None
    
    v = Path(video_name)
    # Find all files that start with {video_stem}_annotations
    annotation_pattern = f'{v.stem}_annotations*'
    annotation_files_src = list(v.parent.glob(annotation_pattern))
    
    if not annotation_files_src:
        print(f"No annotation files found for pattern: {annotation_pattern}")
        return None
    
    copied_files = []
    videos_dir = Path(config_path).parent / 'videos'
    # Ensure destination directory exists
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    for annotation_file_src in annotation_files_src:
        print(f"Copying: {annotation_file_src.name}")
        annotation_file_dest = videos_dir / annotation_file_src.name
        shutil.copyfile(annotation_file_src, annotation_file_dest)
        copied_files.append(annotation_file_dest)
    
    return copied_files




