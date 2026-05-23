"""
Main DUSTrack module, including an interface to manage DeepLabCut (DLC) projects.
"""
from __future__ import annotations

import fnmatch
import functools
import importlib
import importlib.util
import os
import queue
import re
import shutil
import sys
import threading
import traceback
import warnings
from pathlib import Path, PureWindowsPath, PurePosixPath
from typing import Literal, Mapping, Optional, Union

import numpy as np
import pandas as pd
import cv2 as cv
import pyfilemanager
import pysampled
from skimage import io, img_as_ubyte

import matplotlib.pyplot as plt
import datanavigator as dnav
from datanavigator import VideoReader, cpu

from .lk_filter import lk_moving_average_filter
from .annotations import VideoAnnotation, VideoAnnotations
from .seed import (
    get_seed_bundles_root,
    import_seed_bundle_into_project,
    inspect_seed_bundle,
    list_seed_bundles,
    set_seed_bundles_root,
)
from . import _config
from ._bundle import (
    HYDRATION_FAILED,
    HYDRATION_HYDRATING,
    HYDRATION_PENDING,
    HYDRATION_READY,
    _BgHydrationWorker,
    _BundleState,
    _HDF5_LOCK,
)
from ._layer_names import (
    _DENSE_LAYER_PREFIXES,
    _DENSE_LAYER_SUBSTRINGS,
    _dlc_bodyparts_to_layer_labels,
    _is_dense_layer_name,
)
from ._qt_styling import _make_group_styler, _pin_qt_palette, _qss_for_group
from ._image_enhance import (
    _CLAHE_CLIP_MAX,
    _CLAHE_CLIP_MIN,
    _GAMMA_MAX,
    _GAMMA_MIN,
    _SLIDER_TICKS,
    _apply_gamma_only,
    _auto_enhance_params,
    _clahe_clip_to_slider,
    _enhance_is_passthrough,
    _gamma_to_slider,
    _make_enhance_widget_class,
    _slider_to_clahe_clip,
    _slider_to_gamma,
    enhance_ultrasound_image,
)
# Lazy DLC loader -- the plumbing lives in dustrack.dlcloader after the
# 1.2.0rc1 refactor. We import the loader module and re-export the
# function-y names directly. The *mutating* names (``DLC3``,
# ``deeplabcut``, ``VideoWriter``, ``ScannerError``, ``_DLC_LOAD_STATE``,
# ``_DLC_LOAD_THREAD``) are routed through the module-level
# ``__getattr__`` defined at the end of this file -- ``from .dlcloader
# import DLC3`` would snapshot the value at import time and miss
# mutations done by ``_ensure_dlc_loaded()`` on the loader's globals.
from . import dlcloader as _dlcloader
from .dlcloader import (
    HAS_DLC,
    _DLC_LOAD_CALLBACKS,
    _DLC_LOAD_LOCK,
    _dlc_load_state,
    _ensure_dlc_loaded,
    _ensure_dlc_loaded_async,
    _fire_dlc_load_callbacks,
    register_dlc_load_callback,
)
from ._overlays import (
    _VIDEO_PICKER_EXTENSIONS,
    _default_training_options,
    _make_confirm_overlay_class,
    _make_open_video_overlay_class,
    _make_progress_overlay_class,
    _make_seed_bundle_picker_class,
    _make_training_options_class,
    _prompt_for_videos,
    _render_recent_session_label,
    _show_first_paint_notice,
    _training_options_to_train_iteration_kwargs,
    _QueueWriter,
    _Tee,
)
from ._file_management import (
    VideoFileManager,
    _extract_frames,
    _extract_frames_decord,
    get_annotation_file_name,
    make_annotation_file_name,
    merge_annotations_in_folder,
    rebase_to_config,
)


EXPERIMENTER = _config.EXPERIMENTER


# Layer-name patterns that indicate "dense" tracking output (data on
# every frame, like a model prediction or a smoothed trajectory) --
# the default rendering for these is a line plot, vs the dnav default
# of "dot" which is right for sparse manual annotations. Kept here as
# data, not a hardcoded predicate, so adding a new smoothing recipe
# (e.g. a second post-processing filter that writes
# <stem>_kalman_<param>.json) is a one-line tuple edit. See
# :func:`_is_dense_layer_name`.






class DLCData(pysampled.Data):
    """
    Data container for DeepLabCut tracking results.
    
    Provides convenient loading and manipulation of DLC output files (HDF5 format),
    with automatic extraction of metadata like body part names and coordinate labels.
    
    Attributes:
        signal_names (list): Names of tracked body parts (e.g., ['nose', 'left_ear']).
        signal_coords (list): Coordinate names (typically ['x', 'y', 'likelihood']).
    
    Example:
        >>> # Load from DLC output file
        >>> data = DLCData.from_hdf('video_dlc_resnet50_model_name.h5')
        >>> 
        >>> # Load from video (finds associated HDF5 file)
        >>> data = DLCData.from_video('video.mp4', iter_num=250000)
    """
    def __setstate__(self, state):
        """
        Restore object state with backwards compatibility.
        
        Handles legacy attribute names ('coords', 'label_names') by converting
        them to current naming convention ('signal_coords', 'signal_names').
        """
        super().__setstate__(state)
        if "coords" in self.meta:
            self.signal_coords = self.meta.pop("coords")
        if "label_names" in self.meta:
            self.signal_names = self.meta.pop("label_names")
    
    @classmethod
    def from_hdf(cls, file_path):
        """
        Load DLC data from an HDF5 file.
        
        Args:
            file_path (str): Path to the DLC output HDF5 file.
        
        Returns:
            DLCData: Loaded data with extracted metadata.
        
        Raises:
            AssertionError: If file doesn't exist.
            FileNotFoundError: If corresponding labeled video cannot be found.
        """
        assert os.path.exists(file_path)
        df_h5 = pd.read_hdf(file_path)
        label_names = list(df_h5.columns.unique(level='bodyparts'))
        coords = list(df_h5.columns.unique(level='coords'))
        vid_paths = pyfilemanager.FileManager(Path(file_path).parent).add()[f'*{Path(file_path).stem}*_labeled.mp4']
        if len(vid_paths) == 0:
            raise FileNotFoundError('Could not find the video file')
        sr = int(cv.VideoCapture(vid_paths[0]).get(cv.CAP_PROP_FPS))
        return DLCData(df_h5.values, sr, meta=dict(label_names=label_names, coords=coords))
    
    @classmethod
    def from_video(cls, vid_path, iter_num=None):
        """
        Load DLC data associated with a video file.
        
        Automatically searches for HDF5 files matching the video name and
        loads the specified training iteration (or the highest if not specified).
        
        Args:
            vid_path (str): Path to the video file.
            iter_num (int, optional): Training iteration number. If None, uses highest.
        
        Returns:
            DLCData: Loaded tracking data.
        
        Raises:
            AssertionError: If video file doesn't exist or requested iteration not found.
        """
        assert os.path.exists(vid_path)
        # find the hdf file
        vid_path = Path(vid_path)
        h5_list = pyfilemanager.FileManager(vid_path.parent).add()[f'{vid_path.stem}*.h5']
        iter_num_to_fname = {int(Path(x).stem.split('_')[-1]):x for x in h5_list}
        if iter_num is None:
            # pick the highest iteration number
            iter_num = max(iter_num_to_fname)
        assert iter_num in iter_num_to_fname
        h5_file = iter_num_to_fname[iter_num]
        return cls.from_hdf(h5_file)


class DLCProject:
    """Interface to deeplabcut training and inference
    Current workflow:
        1. Create a project with some videos. Videos will be copied.
            d = DLCProject(r'C:/data_opr02/004_02/ml_models/dlc', name='opr02_s004_muscles', experimenter='praneeth', videos=[<video_list>])
        2. Launch the initial annnotator for video 0, repeat if there are more videos
            d.annotate(0) 
        3. Extract frames, train network, evaluate network, analyze videos, and create labeled video
            d.process()
        4. Refine the labels
            d.annotate(0, 'praneeth_2') # the second argument determines the suffix for the annotations file.
            **CAUTION**: Make sure that the files are read by extract_frames in the correct order! 
            Pay attention to the output of this method.
        5. Re-train network with refined labels
            d.process()

        Repeat steps 4 and 5 until satisfied with the results.
    """
    def __init__(self, path, videos=[], name='test_01', experimenter=_config.EXPERIMENTER, annotation_suffix='', internal_to_dlc_labels: dict=None):
        """
        Initialize or load a DeepLabCut project.
        
        If a config.yaml exists at the path, loads the existing project.
        Otherwise, creates a new project with the provided videos.
        
        Args:
            path (str): Directory containing or for the project.
            videos (list): List of video file paths to include.
            name (str): Project name (must contain underscore for proper config handling).
            experimenter (str): Experimenter identifier.
            annotation_suffix (str): Suffix for annotation files (e.g., 'manual', 'refined').
            internal_to_dlc_labels (dict, optional): Custom label name mapping.
        
        Note:
            Videos are copied into the project folder by default.
            Project names without underscores may cause config issues with network paths.
        """
        if not HAS_DLC:
            raise RuntimeError('Install deeplabcut to use DLCProject functionality.')
        # Block here for the real ``deeplabcut`` import. If
        # ``_ensure_dlc_loaded_async`` already kicked off the loader
        # this returns immediately; otherwise it does the (~7 s) import
        # synchronously. Either way, by the time ``DLCProject.__init__``
        # returns, the module-level ``deeplabcut``, ``VideoWriter``,
        # ``ScannerError`` and ``DLC3`` globals are bound -- every
        # method on this instance can use them without a per-call
        # ensure.
        _ensure_dlc_loaded()
        config_path = None
        if os.path.isfile(path):
            assert Path(path).stem == 'config' and Path(path).suffix == '.yaml'
            config_path = path
        if os.path.isdir(path):
            if os.path.exists(Path(path) / 'config.yaml'):
                config_path = Path(path) / 'config.yaml'
        self.path = path

        assert isinstance(annotation_suffix, str)
        self.annotation_suffix = annotation_suffix

        if isinstance(videos, str):
            videos = [videos]

        new_project = False
        if config_path is None:
            assert len(videos) > 0
            config_path = _dlcloader.deeplabcut.create_new_project(name, experimenter, videos, working_directory=path, copy_videos=True)
            new_project = True
        
        self.config_path = config_path

        self.internal_to_dlc_labels = internal_to_dlc_labels

        if new_project:
            annotation_file_names = self.copy_annotations(videos)
            n_annotations_set = {len(VideoAnnotation(fname, vname).labels) for fname, vname in zip(annotation_file_names, videos)}
            assert len(n_annotations_set) == 1 # number of annotations in all the files should match
            annotation_names = [set(VideoAnnotation(fname, vname).labels) for fname, vname in zip(annotation_file_names, videos)]
            common_labels = functools.reduce(lambda x, y: x.intersection(y), annotation_names)
            all_labels = functools.reduce(lambda x, y: x.union(y), annotation_names)
            assert common_labels == all_labels
            annotation_names = sorted(list(common_labels))
            bodyparts = [f'point{x}' for x in annotation_names]
            self.edit_config(bodyparts=bodyparts, skeleton=None)
            self.edit_config(snapshotindex='all') # evaluate all snapshots
            if not os.path.exists(self.paths['models']):
                os.makedirs(self.paths['models'])

        # Re-anchor each video path so it shares config.yaml's root, regardless
        # of which NIC / drive letter / OS was used when the project was created.
        new_video_sets = {}
        for k, v in self.config["video_sets"].items():
            try:
                new_video_sets[rebase_to_config(self.config_path, k)] = v
            except ValueError as e:
                print(f"rebase_to_config: leaving path unchanged ({e})")
                new_video_sets[k] = v
        self.edit_config(video_sets=new_video_sets)

        try:
            _dlcloader.deeplabcut.auxiliaryfunctions.read_config(self.config_path)
        except _dlcloader.ScannerError:
            print("Config file is corrupted. Fix it manually.")
            print("If there is no _ in the name, then the config file has issues "
                  "when dealing with folders on the server.")

    @property
    def paths(self) -> Mapping[str, Path]:
        """
        Full paths to project folder and standard DLC subfolders.
        
        Returns:
            dict: Mapping of folder names to Path objects with keys:
                - 'project': Main project directory
                - 'models': Trained model weights (dlc-models or dlc-models-pytorch)
                - 'results': Evaluation results
                - 'labels': Labeled frame data
                - 'training_data': Training datasets
                - 'videos': Video files
        """
        project_path = Path(self.config_path).parent
        model_folder_name = 'dlc-models-pytorch' if _dlcloader.DLC3 else 'dlc-models'
        evaluation_folder_name = 'evaluation-results-pytorch' if _dlcloader.DLC3 else 'evaluation-results'
        return dict(
            project       = project_path,
            models        = project_path / model_folder_name,
            results       = project_path / evaluation_folder_name,
            labels        = project_path / 'labeled-data',
            training_data = project_path / 'training-datasets',
            videos        = project_path / 'videos',
        )
    
    @property
    def config(self) -> dict:
        """
        Current project configuration dictionary.
        
        Returns:
            dict: Parsed contents of config.yaml.
        """
        return _dlcloader.deeplabcut.auxiliaryfunctions.read_config(self.config_path)
    
    @property
    def name(self) -> str:
        """Project name from configuration."""
        return self.config['Task']

    @property
    def trackers(self) -> list:
        """
        Names of tracked body parts as used internally by DLC.
        
        Returns:
            list: Body part names (e.g., ['point0', 'point1']).
        """
        return self.config['bodyparts']

    @property
    def label_names(self) -> list:
        """
        Human-readable names for tracked points.
        
        Returns meaningful names from dlc_trackermap.txt if available,
        otherwise returns the internal tracker names.
        
        Returns:
            list: Display names for body parts.
        """
        trackermap = self.trackermap
        return [trackermap[tracker] if tracker in trackermap else tracker for tracker in self.trackers]

    @property
    def trackermap(self):
        """
        Load meaningful label names from dlc_trackermap.txt.
        
        This file maps internal names (point0, point1) to biological names
        (nose, left_ear, etc.) for better interpretability.
        
        Returns:
            dict: Mapping from internal names to display names.
        
        Example dlc_trackermap.txt content:
            point0 - muscle_boundary
            point1 - fascia
            point2 - bone
        """
        map_file = Path(self.paths['project']) / 'dlc_trackermap.txt'
        # Path.read_text rather than builtin open() because the module
        # defines a top-level `open` (the workflow entry point) that
        # shadows builtins.open inside this module.
        if map_file.is_file():
            text = map_file.read_text(encoding='utf-8-sig')
            trackermap = [x.split(' - ') for x in text.splitlines() if x]
            return {x[0]: x[1] for x in trackermap}
        else:
            return {}
    
    def edit_config(self, config_file=None, **kwargs):
        """
        Modify project configuration parameters.
        
        Args:
            config_file (str, optional): Path to config file. Defaults to main config.
            **kwargs: Configuration parameters to update (e.g., iteration=2, snapshotindex=5).
        
        Returns:
            Result of deeplabcut.auxiliaryfunctions.edit_config().
        """
        if config_file is None:
            config_file = self.config_path
        assert os.path.exists(config_file)
        return _dlcloader.deeplabcut.auxiliaryfunctions.edit_config(config_file, kwargs)

    @property
    def video_list(self) -> list[Path]:
        """Full paths to videos in the project."""
        return list(self.config['video_sets'].keys())
    
    @property
    def video_names(self) -> list[str]:
        """Video filenames without extensions."""
        return [Path(vname).stem for vname in self.video_list]
    
    @property
    def current_iteration(self) -> int:
        """Model iteration number currently set in config.yaml."""
        return self.config['iteration']
    
    @current_iteration.setter
    def current_iteration(self, iteration_num: int):
        """
        Set the active model iteration in config.yaml.
        
        Args:
            iteration_num: Iteration number, or 'latest' for most recent,
                or 'next' for latest+1 (if latest is trained).
        """
        if isinstance(iteration_num, str):
            assert iteration_num in ('latest', 'next')
            if iteration_num == 'latest':
                iteration_num = self.latest_iteration
            elif iteration_num == 'next':
                if self.latest_iteration_is_trained():
                    iteration_num = self.latest_iteration + 1
                else:
                    iteration_num = self.latest_iteration
        assert isinstance(iteration_num, int)
        self.edit_config(iteration=iteration_num)
    
    @property
    def latest_iteration(self) -> int:
        """Highest iteration number in dlc-models folder."""
        all_iterations = self.all_iterations
        if not all_iterations:
            return 0
        return self.all_iterations[-1]
    
    @property
    def latest_trained_iteration(self) -> int:
        """Most recent iteration that has saved model snapshots."""
        return max([iteration for iteration,snapshot in self.all_snapshots.items() if len(snapshot)], default=-1)
    
    @property
    def all_iterations(self) -> list:
        """All iteration numbers found in dlc-models, sorted ascending."""
        ret = [int(x.split('-')[-1]) for x in os.listdir(self.paths['models']) if x.startswith('iteration-') and os.path.isdir(self.paths['models'] / x)]
        ret.sort()
        return ret

    @property
    def all_snapshots(self) -> Mapping[int, list[int]]:
        """
        Training snapshots for each model iteration.
        
        Returns:
            dict: Maps iteration number to list of training iteration numbers.
                For DLC3, identifies .pt files; for DLC2, identifies .index files.
        """
        if _dlcloader.DLC3:
            ext = ".pt"
        else:
            ext = ".index"
    
        ret = {}
        for iteration_num in self.all_iterations:
            source_path = self.paths['models'] / f'iteration-{iteration_num}'
            snapshot_filenames = pyfilemanager.FileManager(source_path).add()[f'*train/snapshot*{ext}']
            snapshot_numbers = [int(Path(x).stem.split('-')[-1]) for x in snapshot_filenames if "best" not in Path(x).stem]
            snapshot_numbers.sort()
            snapshot_numbers += [int(Path(x).stem.split('-')[-1]) for x in snapshot_filenames if "best" in Path(x).stem]
            ret[iteration_num] = snapshot_numbers
        return ret
    
    def current_iteration_is_trained(self) -> bool:
        """Check if current iteration has any saved snapshots."""
        return self.iteration_is_trained(self.current_iteration)
    
    def latest_iteration_is_trained(self) -> bool:
        """Check if latest iteration has any saved snapshots."""
        return self.iteration_is_trained(self.latest_iteration)

    def iteration_is_trained(self, iteration_num: int) -> bool:
        """
        Check if a specific iteration has been trained.
        
        Args:
            iteration_num (int): Model iteration to check.
        
        Returns:
            bool: True if snapshots exist for this iteration.
        """
        if iteration_num not in self.all_snapshots:
            return False
        return len(self.all_snapshots[iteration_num]) > 0
    
    def increment_iteration(self):
        """
        Advance to next iteration if current one is trained.
        
        Returns:
            self: For method chaining.
        """
        self.current_iteration = 'next'
        return self
        
    def add_videos(self, videos: list[Path]):
        """
        Add new videos to existing project and copy their annotations.
        
        Args:
            videos: List of video file paths to add.
        
        Returns:
            self: For method chaining.
        """
        if isinstance(videos, (str, Path)):
            videos = [videos]
        _dlcloader.deeplabcut.add_new_videos(self.config_path, videos, copy_videos=True)
        self.copy_annotations(videos)
        return self
    
    def copy_annotations(self, video_name: Union[Path, list]):
        """
        Copy DUSTrack JSON files into project's video folder.
        
        Args:
            video_name: Single video path or list of video paths.
        
        Returns:
            str or list: Path(s) to copied annotation file(s), or None if not found.
        
        Note:
            Looks for files matching {video_stem}_annotations_{suffix}.json
        """
        if isinstance(video_name, list):
            copied_files = []
            for this_video_name in video_name:
                copied_file = self.copy_annotations(this_video_name)
                if copied_file is not None:
                    copied_files.append(copied_file)
            return copied_files
        v = Path(video_name)
        a_name = f'{v.stem}_annotations{"_" if self.annotation_suffix else ""}{self.annotation_suffix}.json'
        annotation_file_src = v.parent / a_name
        annotation_file_dest = Path(self.config_path).parent / 'videos' / a_name
        if os.path.exists(annotation_file_src):
            shutil.copyfile(annotation_file_src, annotation_file_dest)
            return annotation_file_dest
        return None

    def extract_frames(self, annotation_file_names=None, suffix_merged='merged', save_merged_json=False, check=False):
        """
        Extract labeled frames from videos and convert annotations to DLC format.
        
        This method:
        1. Finds all annotation JSON files for each video
        2. Merges multiple annotation files if present
        3. Extracts the annotated frames from videos
        4. Converts annotations to DLC's CSV/HDF5 format in labeled-data folder
        
        Args:
            annotation_file_names (list, optional): Specific annotation files to use.
                If None, automatically finds all matching files.
            suffix_merged (str): Suffix for merged annotation file. Defaults to 'merged'.
            save_merged_json (bool): Whether to save the merged JSON. Defaults to False.
            check (bool): Whether to run deeplabcut.check_labels(). Defaults to False.
        
        Returns:
            self: For method chaining.
        
        Note:
            Automatically excludes files with '_dlccorr' suffix (correction files).
        """
        annotation_file_names_input = annotation_file_names
        for video_file_name in self.video_list:
            coords = self.config["video_sets"][video_file_name]["crop"].split(",")
            video_stem = Path(video_file_name).stem
            output_path = self.paths['labels'] / video_stem

            if annotation_file_names_input is None:
                pattern = f'{video_stem}*_annotations*.json'
                fm = pyfilemanager.FileManager(self.paths['videos']).add()
                file_names = fnmatch.filter([Path(x).name for x in fm.all_files], pattern)
                annotation_file_names = sorted([fm[file_name][0] for file_name in file_names])
                # annotation_file_names = sorted(pyfilemanager.FileManager(self.paths['videos']).add()[f'{video_stem}*_annotations*.json'])
                # ignore the *correction* files. In theory, no training is to be done after the dlccorr files are created, but just being careful.
                annotation_file_names = [x for x in annotation_file_names if "_dlccorr" not in x]
                print(f'Loading annotations from {len(annotation_file_names)} file(s): ')
                print([Path(x).stem for x in annotation_file_names])
                print()
            
            if len(annotation_file_names) == 0:
                # there are multiple videos, but one of them does not have any labels
                continue
            
            ann = VideoAnnotation.from_multiple_files(
                fname_list = annotation_file_names,
                vname = video_file_name,
                name = suffix_merged,
                fname_merged = make_annotation_file_name(video_file_name, suffix_merged)
            )

            if save_merged_json:
                ann.save()
            _extract_frames_decord(video_file_name, ann.frames, output_path, coords)
            ann.to_dlc(
                scorer       = self.config['scorer'],
                output_path  = output_path,
                file_prefix  = f"CollectedData_{self.config['scorer']}",
                img_prefix   = 'img',
                img_suffix   = '.png',
                label_prefix = 'point',
                save         = True,
                internal_to_dlc_labels=self.internal_to_dlc_labels
                )
            
            if check:
                _dlcloader.deeplabcut.check_labels(self.config_path) # this creates an _labeled folder, which doesn't seem necessary in this case
        
        return self
    
    def get_pose_cfg_file(self, iteration_num: int=None, type_: str='train') -> Path:
        """
        Get path to pose configuration file for an iteration.
        
        Args:
            iteration_num (int, optional): Iteration number. Defaults to current.
            type_ (str): 'train' or 'test' subfolder. Defaults to 'train'.
        
        Returns:
            Path: Full path to pose_cfg.yaml (DLC2) or pytorch_config (DLC3).
        """
        if iteration_num is None:
            iteration_num = self.current_iteration
        assert type_ in ('train', 'test')
        if _dlcloader.DLC3:
            cfg_name = "pytorch_config"
        else:
            cfg_name = "pose_cfg"
        cfg_files = pyfilemanager.FileManager(self.paths['models'] / f'iteration-{iteration_num}').add()[f'*{type_}/{cfg_name}*']
        assert len(cfg_files) == 1
        return cfg_files[0]
    
    def get_best_snapshot(self, iteration_num: int=None) -> int:
        """
        Find training iteration with lowest test error.
        
        For DLC3, uses the snapshot marked as 'best' unless DLC3_USE_LAST_SNAPSHOT
        is True in config, in which case returns the last snapshot.
        For DLC2, parses CombinedEvaluation-results.csv.
        
        Args:
            iteration_num (int, optional): Model iteration. Defaults to current.
        
        Returns:
            int: Training iteration number of best snapshot.
        """
        if iteration_num is None:
            iteration_num = self.current_iteration
        
        if _dlcloader.DLC3:
            source_path = self.paths['models'] / f'iteration-{iteration_num}'
            snapshot_filenames = pyfilemanager.FileManager(source_path).add()[f'*train/snapshot*.pt']
            snapshot_numbers = [int(Path(x).stem.split('-')[-1]) for x in snapshot_filenames]
            best_snapshot_number = [int(Path(x).stem.split('-')[-1]) for x in snapshot_filenames if "best" in Path(x).stem]
            if not _config.DLC3_USE_LAST_SNAPSHOT:
                if best_snapshot_number:
                    return best_snapshot_number[0]
            return sorted(snapshot_numbers)[-1]

        eval_file_name = self.paths['results'] / f'iteration-{iteration_num}' / 'CombinedEvaluation-results.csv'
        if os.path.exists(eval_file_name):
            # pick the snapshot with the lowest training error
            df_eval = pd.read_csv(eval_file_name)
            df_eval = df_eval.rename(columns=lambda x: x.strip())
            best_snapshot = df_eval[df_eval['Test error(px)'] == min(df_eval['Test error(px)'])]['Training iterations:'].iloc[0]
        else:
            # pick the latest snapshot
            print('Could not evaluate best snapshot, setting it to latest')
            best_snapshot = self.all_snapshots[iteration_num][-1]
        return best_snapshot
    
    def get_best_snapshot_test_error(self, iteration_num: int=None) -> float:
        """
        Get test error (RMSE in pixels) at best snapshot.
        
        Args:
            iteration_num (int, optional): Model iteration. Defaults to latest trained.
        
        Returns:
            float: Test error in pixels, or -1.0 if evaluation file doesn't exist.
        """
        if iteration_num is None:
            iteration_num = self.latest_trained_iteration
        eval_file_name = self.paths['results'] / f'iteration-{iteration_num}' / 'CombinedEvaluation-results.csv'
        column_name = 'test rmse_pcutoff' if _dlcloader.DLC3 else 'Test error(px)'
        if os.path.exists(eval_file_name):
            # pick the snapshot with the lowest training error
            df_eval = pd.read_csv(eval_file_name)
            df_eval = df_eval.rename(columns=lambda x: x.strip())
            return float(min(df_eval[column_name]))
        return -1.
    
    def get_best_snapshot_idx(self, iteration_num: int=None) -> int:
        """
        Get snapshot index (not training iteration number) of best snapshot.
        
        Args:
            iteration_num (int, optional): Model iteration. Defaults to current.
        
        Returns:
            int: Index in the all_snapshots list for this iteration.
        """
        if iteration_num is None:
            iteration_num = self.current_iteration
        best_snapshot = self.get_best_snapshot(iteration_num)
        return self.all_snapshots[iteration_num].index(best_snapshot)

    def initialize_weights(self, source_iteration: int=None, source_snapshot: int=None, dest_iteration: int=None):
        """
        Initialize model weights from a previous iteration (transfer learning).
        
        Used when refining a model with additional labels. Edits the pose_cfg
        file to set init_weights parameter.
        
        Args:
            source_iteration (int, optional): Iteration to copy from. 
                Defaults to second-to-last iteration.
            source_snapshot (int, optional): Training iteration within source_iteration.
                Defaults to best snapshot.
            dest_iteration (int, optional): Iteration to initialize. 
                Defaults to latest iteration.
        
        Returns:
            self: For method chaining.
        
        Note:
            Does nothing if there's only one iteration (no source to copy from).
        """        
        all_iterations = self.all_iterations
        if source_iteration is None: # pick the last iteration
            if len(all_iterations) <= 1:
                return self
            source_iteration = all_iterations[-2]

        if source_snapshot is None:
            source_snapshot = self.get_best_snapshot(source_iteration)

        if dest_iteration is None:
            dest_iteration = all_iterations[-1]
        
        # find the correct pose_cfg file
        cfg_file = self.get_pose_cfg_file(dest_iteration)
        source_path = self.paths['models'] / f'iteration-{source_iteration}'
        ext = '.pt' if _dlcloader.DLC3 else '.index'
        init_weights_files = pyfilemanager.FileManager(source_path).add()[f'*train/snapshot-*{source_snapshot}{ext}']
        assert len(init_weights_files) == 1

        if _dlcloader.DLC3:
            self.edit_config(cfg_file, resume_training_from=init_weights_files[0].removesuffix('.index'))
        else:
            self.edit_config(cfg_file, init_weights=init_weights_files[0].removesuffix('.index'))
        return self

    def _initialize_weights_from_external_path(self, external_path):
        """Edit the latest iteration's pose_cfg to initialise weights from
        an external snapshot file.

        Sibling of :meth:`initialize_weights` for the
        ``train_iteration(refine_mode="external")`` UI path. Unlike
        :meth:`initialize_weights` (which resolves an in-project
        ``(source_iteration, source_snapshot)`` pair via ``FileManager``),
        this helper takes the path directly and writes it verbatim into
        the destination iteration's pose_cfg.

        The DLC3 ``train_iteration`` path normally bypasses this helper
        and passes ``snapshot_path=`` straight to ``train_network`` so
        pose_cfg stays clean. This helper is the DLC2 fallback (DLC2's
        ``train_network`` has no runtime override) and is also available
        on DLC3 if a caller wants the pose_cfg path explicitly.

        Args:
            external_path (str or Path): Path to an external snapshot
                file (``.pt`` on DLC3, ``.index`` on DLC2). A trailing
                ``.pt`` / ``.index`` extension is stripped to match the
                pose_cfg convention :meth:`initialize_weights` uses.

        Returns:
            self: For method chaining.
        """
        external_path = str(external_path)
        # Match initialize_weights' convention: pose_cfg expects the
        # extensionless prefix (DLC2: init_weights=.../snapshot-200;
        # DLC3: resume_training_from=.../snapshot-200).
        for ext in (".pt", ".index"):
            if external_path.endswith(ext):
                external_path = external_path[: -len(ext)]
                break

        dest_iteration = self.all_iterations[-1]
        cfg_file = self.get_pose_cfg_file(dest_iteration)
        if _dlcloader.DLC3:
            self.edit_config(cfg_file, resume_training_from=external_path)
        else:
            self.edit_config(cfg_file, init_weights=external_path)
        return self

    def create_training_dataset(self, **kwargs):
        """Call deeplabcut.create_training_dataset."""
        net_type = kwargs.pop('net_type', 'resnet_50')
        _dlcloader.deeplabcut.create_training_dataset(self.config_path, net_type=net_type, **kwargs)
        return self

    def train(self, **kwargs):
        """
        Train the neural network model.
        
        Sets custom learning rate schedule and trains with more iterations than
        DLC defaults for better convergence.
        
        Args:
            **kwargs: Passed to deeplabcut.train_network().
                - maxiters (int): Total training iterations. Default: 500000 (DLC2) or 1000 (DLC3 epochs).
                - max_snapshots_to_keep (int): Max saved checkpoints. Default: 20.
        
        Returns:
            self: For method chaining.
        
        Note:
            Custom learning rate schedule: [0.005@10k, 0.02@350k, 0.002@425k, 0.001@1M]
        """
        maxiters = kwargs.pop('maxiters', 500000)
        max_snapshots_to_keep = kwargs.pop('max_snapshots_to_keep', 20)
        cfg_file = self.get_pose_cfg_file()
        self.edit_config(cfg_file, multi_step = [[0.005, 10000], [0.02, 350000], [0.002, 425000], [0.001, 1000000]])
        _dlcloader.deeplabcut.train_network(self.config_path, maxiters=maxiters, max_snapshots_to_keep=max_snapshots_to_keep, pytorch_cfg_updates={"runner.eval_interval": 25},**kwargs)
        return self
    
    def evaluate(self, **kwargs):
        """
        Evaluate all training snapshots on test set.
        
        Temporarily sets snapshotindex to 'all' to evaluate every checkpoint,
        then restores original value.
        
        Args:
            **kwargs: Passed to deeplabcut.evaluate_network().
        
        Returns:
            self: For method chaining.
        """
        current_snapshotindex_value = self.config['snapshotindex']
        self.edit_config(snapshotindex='all')
        _dlcloader.deeplabcut.evaluate_network(self.config_path, **kwargs)
        self.edit_config(snapshotindex=current_snapshotindex_value)
        return self

    def analyze_videos(self, iteration_num=None, snapshotindex=None, create_video=True, **kwargs):
        """
        Run inference on videos and optionally create labeled output videos.
        
        Args:
            iteration_num (int, optional): Model iteration to use. Defaults to current.
            snapshotindex (int, optional): Snapshot index to use. 
                Defaults to best snapshot. Negative indices supported.
            create_video (bool): Whether to create labeled video. Defaults to True.
            **kwargs: Additional arguments for deeplabcut.analyze_videos().
                - videos: List of video paths or indices. If not provided, analyzes all videos.
        
        Returns:
            self: For method chaining.
        
        Note:
            Results saved to videos/iteration-{N}/ subfolder.
            If videos kwarg contains integers, they're treated as indices into self.video_list.
        """
        if iteration_num is None:
            iteration_num = self.current_iteration
        
        if snapshotindex is None:
            snapshotindex = self.get_best_snapshot_idx(iteration_num)
        else:
            n_snapshots = len(self.all_snapshots[iteration_num])
            if snapshotindex < 0:
                snapshotindex = snapshotindex % n_snapshots
            assert 0 <= snapshotindex < n_snapshots
        
        save_as_csv = kwargs.pop('save_as_csv', True)

        # DeepLabCut's PyTorch backend defaults to batch_size=1 when neither
        # the kwarg nor the project config sets one, which leaves an RTX-class
        # GPU heavily under-utilised. The throughput knee for ResNet-50 BU on
        # a 706x558 video on DLC 3.0.0rc14 + RTX 4090 is batchsize~4 (median
        # 154 fps; see S:/_corpus/dustrack/dlc_inference_bench_2026-05-20/).
        # Respect the project config if it sets ``batch_size`` explicitly.
        if 'batchsize' not in kwargs and self.config.get('batch_size') is None:
            kwargs['batchsize'] = 4

        if "videos" in kwargs:
            assert isinstance(kwargs["videos"], list)
            # if kwargs["videos"] is a list of integers, convert to list of video paths using self.video_list
            if all(isinstance(v, int) for v in kwargs["videos"]):
                video_indices = kwargs["videos"]
                video_list = []
                for idx in video_indices:
                    if idx < 0:
                        idx = len(self.video_list) + idx
                    assert 0 <= idx < len(self.video_list), f"Video index {idx} is out of range."
                    video_name = self.video_list[idx]
                    assert os.path.exists(video_name), f"Video {video_name} does not exist."
                    video_list.append(video_name)
                kwargs["videos"] = video_list
        else:
            kwargs["videos"] = self.video_list
        
        current_snapshotindex_value = self.config['snapshotindex']
        self.edit_config(snapshotindex=snapshotindex)

        common_params = dict(
            config     = self.config_path, 
            videos     = kwargs.pop('videos'), 
            destfolder = self.paths['videos'] / f'iteration-{iteration_num}'
            )

        _dlcloader.deeplabcut.analyze_videos(**common_params, save_as_csv=save_as_csv, **kwargs)
        if create_video:
            _dlcloader.deeplabcut.create_labeled_video(**common_params)
        
        self.edit_config(snapshotindex=current_snapshotindex_value)
        return self
    # refine can be both bool or string, if string, it is the path of the model to initialize weights from
    def process(self, iteration_num=None, maxiters=None, refine: Union[bool, str]=True, create_video=True, source_snapshot=None, **kwargs):
        """
        Automated workflow: extract frames, train, evaluate, and analyze.
        
        This is the main method for handling the full DLC pipeline. It intelligently
        decides what steps to run based on the current project state:
        - If iteration already evaluated: just analyze videos
        - If frames need extraction: extract them
        - If not trained: train the model
        - If refining: initialize weights from previous iteration
        
        Args:
            iteration_num (int or str, optional): Iteration to process. 
                Can be integer or 'latest'. Defaults to 'latest'.
            maxiters (int, optional): Training iterations. 
                Defaults: 500000 (DLC2) or 1000 epochs (DLC3).
            refine (bool): Use transfer learning from previous iteration. Defaults to True.
            create_video (bool): Create labeled output video. Defaults to True.
            source_snapshot (int, optional): Specific snapshot for weight initialization.
            **kwargs: Additional arguments.
                - videos: List of videos to analyze (can be indices or paths).
        
        Returns:
            self: For method chaining.
        
        Example:
            >>> proj = DLCProject('path/to/project')
            >>> proj.process()  # Full automated workflow
        """
        if iteration_num is None:
            iteration_num = 'latest'
        else:
            assert isinstance(iteration_num, int)

        if maxiters is None:
            if _dlcloader.DLC3:
                # TEMPORARY: dropped from 1000 → 50 to speed up the
                # datanavigator/DUSTrack test-bed iteration loop
                # (S:\_corpus\dustrack\). REVERT to 1000 before 1.1.0rc2
                # ships.
                maxiters = 50 # epochs
            else:
                maxiters = 500000

        self.current_iteration = iteration_num

        current_iteration = self.current_iteration
        latest_iteration = self.latest_iteration
        if current_iteration < latest_iteration:
            return self.evaluate().analyze_videos(create_video=create_video)

        self.extract_frames() # do this every time in case there are any updates to the manual annotations.
        
        if self.latest_iteration_is_trained():
            self.increment_iteration() # increment iteration in the config.yaml file
        
        if not os.path.exists(self.paths['training_data'] / f'iteration-{self.current_iteration}'):
            self.create_training_dataset()
        
        if isinstance(refine, bool) and refine:
            if not self.latest_iteration_is_trained() and self.current_iteration == self.latest_iteration:
                if source_snapshot is not None:
                    source_iteration = self.latest_iteration - int(not self.latest_iteration_is_trained())
                else:
                    source_iteration = None
                self.initialize_weights(source_iteration=source_iteration, source_snapshot=source_snapshot)

        if not self.current_iteration_is_trained():
            try:
                if _dlcloader.DLC3:
                    if isinstance(refine, str):
                        self.train(epochs=maxiters, snapshot_path=refine)
                    else:
                        self.train(epochs=maxiters)
                else:
                    self.train(maxiters=maxiters)
            except KeyboardInterrupt:
                pass

        analyze_videos_kwargs = {}
        if "videos" in kwargs:
            analyze_videos_kwargs["videos"] = kwargs.pop("videos")
        if "analyze_batchsize" in kwargs:
            analyze_videos_kwargs["batchsize"] = kwargs.pop("analyze_batchsize")

        return self.evaluate().analyze_videos(create_video=create_video, **analyze_videos_kwargs)

    def train_iteration(
        self,
        *,
        refine_mode: Literal["scratch", "in_project", "external"] = "scratch",
        source_iteration: int = None,
        source_snapshot: int = None,
        external_snapshot_path: str = None,
        maxiters: int = None,
        create_video: bool = False,
        videos: list = None,
        analyze_batchsize: int = None,
    ):
        """Explicit-args training driver for UI-triggered flows.

        Distinct from :meth:`process` (auto-infer for CLI ergonomics).
        Caller decides everything: refine source, training duration,
        output options. Strict validation per ``refine_mode``; no
        inference and no silent fallbacks.

        The mechanics of advancing iterations (extract_frames →
        increment_iteration if latest is trained → create_training_dataset
        if needed) mirror :meth:`process`; the only difference is *how*
        weights are initialised once the destination iteration is in
        place.

        Args:
            refine_mode: How to initialise weights for the next training
                round. ``"scratch"`` starts from random init (no pose_cfg
                edit); ``"in_project"`` copies weights from a snapshot
                in this project (requires ``source_iteration``, optional
                ``source_snapshot``); ``"external"`` initialises from an
                arbitrary snapshot file (requires
                ``external_snapshot_path``; supported on both DLC2 and
                DLC3 -- DLC3 passes the path through
                ``train_network(snapshot_path=...)``, DLC2 edits
                pose_cfg's ``init_weights`` via
                :meth:`_initialize_weights_from_external_path`).
            source_iteration: in-project iteration to copy weights from.
                Only valid with ``refine_mode="in_project"``; must point
                at a trained iteration.
            source_snapshot: specific snapshot within
                ``source_iteration``. Only valid with
                ``refine_mode="in_project"``; defaults to the best
                snapshot when ``None``.
            external_snapshot_path: path to an external snapshot file
                (``.pt`` on DLC3, ``.index`` on DLC2). Only valid with
                ``refine_mode="external"``; the file must exist at
                call time.
            maxiters: training epochs (DLC3) or iterations (DLC2).
                Defaults to the same values :meth:`process` uses (50 /
                500000) so the two methods stay consistent until the
                UI exposes the field.
            create_video: write a labeled output video after analyze.
                Defaults to ``False`` (the UI ergonomics default;
                :meth:`process` defaults to ``True`` for CLI parity).
            videos: list of videos (indices or paths) to analyze.
                Forwarded to ``analyze_videos``. ``None`` analyses every
                video in the project.
            analyze_batchsize: batchsize for ``analyze_videos``.
                Forwarded on if set; ``None`` lets ``analyze_videos``
                pick its own default (post-2026-05-20: rc14 knee at 4).

        Returns:
            self: For method chaining.

        Raises:
            ValueError: on refine_mode / argument mismatch (see
                :meth:`_validate_train_iteration_args`).
            TypeError: if ``source_iteration`` / ``source_snapshot``
                aren't ``int`` when given.
            FileNotFoundError: if ``external_snapshot_path`` is set but
                the file doesn't exist.
        """
        self._validate_train_iteration_args(
            refine_mode=refine_mode,
            source_iteration=source_iteration,
            source_snapshot=source_snapshot,
            external_snapshot_path=external_snapshot_path,
        )

        if maxiters is None:
            maxiters = 50 if _dlcloader.DLC3 else 500000  # same defaults as process()

        # Iteration advancement mechanics (mirror process()).
        self.extract_frames()  # capture any new manual annotations
        if self.latest_iteration_is_trained():
            self.increment_iteration()
        if not os.path.exists(self.paths['training_data'] / f'iteration-{self.current_iteration}'):
            self.create_training_dataset()

        # Apply refine mode.
        if refine_mode == "in_project":
            self.initialize_weights(
                source_iteration=source_iteration,
                source_snapshot=source_snapshot,
            )
        elif refine_mode == "external" and not _dlcloader.DLC3:
            # DLC2: no runtime override -- edit pose_cfg's init_weights
            # to point at the external snapshot. DLC3 handles this
            # inline at the train call below via snapshot_path=.
            self._initialize_weights_from_external_path(external_snapshot_path)
        # refine_mode == "scratch": no pose_cfg edit; pose_cfg is fresh
        # from create_training_dataset and has no init weights set.

        # Train.
        if not self.current_iteration_is_trained():
            train_kwargs = {}
            if _dlcloader.DLC3:
                train_kwargs["epochs"] = maxiters
                if refine_mode == "external":
                    train_kwargs["snapshot_path"] = external_snapshot_path
            else:
                train_kwargs["maxiters"] = maxiters
            try:
                self.train(**train_kwargs)
            except KeyboardInterrupt:
                pass

        # Evaluate + analyze.
        analyze_kwargs = {}
        if videos is not None:
            analyze_kwargs["videos"] = videos
        if analyze_batchsize is not None:
            analyze_kwargs["batchsize"] = analyze_batchsize
        return self.evaluate().analyze_videos(create_video=create_video, **analyze_kwargs)

    def _validate_train_iteration_args(
        self,
        *,
        refine_mode,
        source_iteration,
        source_snapshot,
        external_snapshot_path,
    ):
        """Strict validation for :meth:`train_iteration`. Raises on
        mismatch.

        The discriminator is ``refine_mode``; the helper enforces a
        canonical valid-combo table:

        - ``"scratch"``: every source / external arg must be ``None``.
        - ``"in_project"``: ``source_iteration`` required (``int``,
          trained); ``source_snapshot`` optional (``int`` or ``None``
          -- ``None`` lets :meth:`initialize_weights` pick the best
          snapshot); ``external_snapshot_path`` must be ``None``.
        - ``"external"``: ``external_snapshot_path`` required (string,
          file must exist); ``source_iteration`` /
          ``source_snapshot`` must be ``None``.
        """
        valid_modes = {"scratch", "in_project", "external"}
        if refine_mode not in valid_modes:
            raise ValueError(
                f"refine_mode must be one of {sorted(valid_modes)}, "
                f"got {refine_mode!r}"
            )

        if refine_mode == "scratch":
            for name, value in (
                ("source_iteration", source_iteration),
                ("source_snapshot", source_snapshot),
                ("external_snapshot_path", external_snapshot_path),
            ):
                if value is not None:
                    raise ValueError(
                        f"refine_mode='scratch' is incompatible with "
                        f"{name}={value!r}"
                    )
            return

        if refine_mode == "in_project":
            if external_snapshot_path is not None:
                raise ValueError(
                    "refine_mode='in_project' is incompatible with "
                    f"external_snapshot_path={external_snapshot_path!r}"
                )
            if source_iteration is None:
                raise ValueError(
                    "refine_mode='in_project' requires source_iteration"
                )
            if not isinstance(source_iteration, int):
                raise TypeError(
                    f"source_iteration must be int, got "
                    f"{type(source_iteration).__name__}"
                )
            if not self.iteration_is_trained(source_iteration):
                trained = [i for i, snaps in self.all_snapshots.items() if snaps]
                raise ValueError(
                    f"source_iteration={source_iteration} is not a "
                    f"trained iteration. Trained iterations: {trained}"
                )
            if source_snapshot is not None and not isinstance(source_snapshot, int):
                raise TypeError(
                    f"source_snapshot must be int or None, got "
                    f"{type(source_snapshot).__name__}"
                )
            return

        if refine_mode == "external":
            for name, value in (
                ("source_iteration", source_iteration),
                ("source_snapshot", source_snapshot),
            ):
                if value is not None:
                    raise ValueError(
                        f"refine_mode='external' is incompatible with "
                        f"{name}={value!r}"
                    )
            if external_snapshot_path is None:
                raise ValueError(
                    "refine_mode='external' requires external_snapshot_path"
                )
            if not os.path.exists(external_snapshot_path):
                raise FileNotFoundError(
                    f"External snapshot not found: {external_snapshot_path}"
                )
            return

    def annotate(self, video_index: int=0, new_annotation_suffix=None, **dustrack_kwargs):
        """
        Launch interactive annotation GUI for a video.

        Opens DUSTrack interface with existing annotation layers loaded,
        including any DLC predictions as line plot overlays.

        Args:
            video_index (int): Index of video in video_list. Defaults to 0.
                Negative indices supported.
            new_annotation_suffix (str, optional): Suffix for new annotation layer.
                Defaults to 'iteration-{N}' where N is the next iteration number.
            **dustrack_kwargs: Forwarded to the DUSTrack constructor. Notable
                pass-through options: ``fast_render=True`` (datanavigator
                1.5.0+ Tier 2 Qt-native video pane, ~3x speedup on the
                interosseous_pn24-x benchmark), ``dark_mode=True``,
                ``clahe_clip``, ``clahe_grid``, ``gamma``, ``brightness``.

        Returns:
            DUSTrack: Interactive annotation interface.

        Note:
            Creates a 'buffer' layer for temporary annotations.
            Latest DLC predictions are automatically set as overlay.
        """
        if video_index < 0:
            video_index = len(self.video_list) + video_index
        assert 0 <= video_index < len(self.video_list)

        if new_annotation_suffix is None:
            if self.latest_iteration_is_trained():
                new_iteration_num = self.latest_iteration + 1
            else:
                new_iteration_num = self.latest_iteration
            new_annotation_suffix = f'iteration-{new_iteration_num}'

        fm_annotations = VideoFileManager(self, video_index)
        annotation_names = fm_annotations.get_all_annotation_layers(new_annotation_suffix)
        annotation_names['buffer'] = fm_annotations.get_new_json('buffer')
        # fast_render default is set by DUSTrack.__init__; no need to
        # duplicate here. Callers can pass ``fast_render=False`` via
        # ``dustrack_kwargs`` to opt out.
        # Lazy import to break the dlcinterface <-> gui cycle (gui.py
        # imports DLCProject at module-load; dlcinterface.py can't
        # import gui at module-load without cycling).
        from .gui import DUSTrack
        ret = DUSTrack(self.video_list[video_index], annotation_names, height_ratios=(3,1,1), **dustrack_kwargs)
        # Wire the DUSTrack back to this project so the Train / Reduce
        # jitter buttons (and `_refresh_dlc_layers`) work on a
        # re-entered session — without this the GUI's `_dlcproject`
        # stays at its `__init__` default of None and "Train DLC model"
        # raises "DLCProject not created."
        ret._dlcproject = self
        # Single helper drives the post-load display state for both
        # the fresh-construction path (this method) and the in-place
        # refresh path (DUSTrack._refresh_dlc_layers).
        ret._normalize_dlc_layer_display()
        # Fold dlccorr (saved as ``*_annotations_dlccorr.json``, so it
        # rides the manuals block out of get_all_annotation_layers) into
        # its own group at the tail of the DLC chain. Without this, a
        # fresh open of a re-entered project would show dlccorr mixed
        # with manuals while a post-train refresh would show it grouped
        # with the DLC chain -- _restructure_annotation_order keeps the
        # two paths in lockstep.
        ret._restructure_annotation_order()
        # ``_dlcproject`` was None when DUSTrack.__init__ ran its initial
        # gate evaluation; re-run now that it's set so "Train DLC model"
        # enables and "Create DLC Project" disables.
        ret._refresh_workflow_button_state()
        ret.update()

        return ret

    def get_trajectories(self, videos=None, iteration=None):
        """
        Load tracking results as DLCData objects.
        
        Args:
            videos (list or str, optional): Videos to load. Defaults to all videos.
            iteration (int, optional): Model iteration. Defaults to current.
        
        Returns:
            dict: Maps video stem to DLCData object.
        
        Raises:
            ValueError: If a requested video is not in the project.
        """
        if iteration is None:
            iteration = self.current_iteration
        if videos is None:
            videos = self.video_list
        elif isinstance(videos, str):
            videos = [videos]

        data = {}
        for video in videos:
            if video not in self.video_list:
                raise ValueError(f"{video} does not exist in this project. It cannot be loaded.")
            data[Path(video).stem] = DLCData.from_video(video) ### Need to find a way to relate training iterations (gradient descent) with training iterations (number of times retrained)
        return data
    
    def open(self):
        """Open project folder in Windows Explorer."""
        os.system(f'explorer.exe "{str(Path(self.config_path).parent)}"')





# ---------------------------------------------------------------------
# DLC project / video path helpers used by dustrack.open dispatch AND
# by DUSTrack class methods (swap_to, add_video, create_dlc_project).
# Kept in dlcinterface to avoid a gui.py <-> open.py cycle.
# ---------------------------------------------------------------------
def _is_dlc_config_yaml(path) -> bool:
    """True iff ``path`` is a DLC ``config.yaml`` file (case-insensitive
    on the basename; the file's parent must exist but we don't structurally
    validate it as a full project here -- DLCProject construction will
    surface a clearer error if the config is malformed)."""
    p = Path(path)
    if not p.is_file():
        return False
    return p.name.lower() == "config.yaml"


def _is_dlc_project_root(folder) -> bool:
    """Cheap structural check for a DLC project folder.

    DLC's ``create_new_project`` always lays down ``config.yaml`` next to
    ``videos/`` and ``labeled-data/``; requiring all three avoids matching
    a stray ``config.yaml`` that belongs to something else. No YAML
    parsing -- pure filesystem.
    """
    f = Path(folder)
    return (
        (f / 'config.yaml').is_file()
        and (f / 'videos').is_dir()
        and (f / 'labeled-data').is_dir()
    )


def _find_dlc_config(path):
    """Resolve ``path`` to the DLC ``config.yaml`` that contains it, or None.

    Resolves four input shapes:

    - ``config.yaml`` file -> that path (only if the sibling project structure exists)
    - DLC project folder -> ``folder / 'config.yaml'``
    - Any file inside a project (notably a video under ``videos/``) -> walks up
      ancestors until a DLC-root is found
    - Anything else (a bare video outside any project, a non-existent path) -> None

    Returning None signals Phase 1 to :func:`open`. Note the walk-up stops
    at the filesystem root; in practice DLC's layout means it terminates
    after one step.
    """
    p = Path(path)
    if not p.exists():
        return None

    if p.is_file() and p.name.lower() == 'config.yaml':
        return p if _is_dlc_project_root(p.parent) else None

    if p.is_dir() and _is_dlc_project_root(p):
        return p / 'config.yaml'

    if p.is_file():
        for ancestor in p.parents:
            if _is_dlc_project_root(ancestor):
                return ancestor / 'config.yaml'

    return None


def _find_video_index(project, video_path):
    """Look up a video's index in ``project.video_list`` by filename stem.

    Stem matching (rather than full-path equality) is robust to the
    drive-letter / UNC / posix shuffling that :func:`rebase_to_config`
    already handles inside ``DLCProject``. Returns None if the video
    isn't part of the project.
    """
    target_stem = Path(video_path).stem
    for i, name in enumerate(project.video_names):
        if name == target_stem:
            return i
    return None


def _session_inside_dlc_project(dustrack) -> Optional[Path]:
    """Return the DLC project root the session sits inside, or None.

    Reuses :func:`_find_dlc_config` for the filesystem walk-up so the
    structural check (``config.yaml + videos/ + labeled-data/``) stays
    in one place. ``self._dlcproject`` is checked first as the cheap
    short-circuit: a session that was opened via ``dustrack.open(<project>)``
    or that survived a successful ``create_dlc_project`` already knows
    its project; we only fall back to walking up ``self.fname``'s
    ancestors when the attribute is unset (e.g. a video opened bare
    that happens to live inside an existing project tree).
    """
    proj = getattr(dustrack, "_dlcproject", None)
    if proj is not None:
        config_path = getattr(proj, "config_path", None)
        if config_path is not None:
            return Path(config_path).parent
    fname = getattr(dustrack, "fname", None)
    if fname is None:
        return None
    config = _find_dlc_config(fname)
    return config.parent if config is not None else None



def _resolve_multi_video_from_list(path_list: list) -> tuple:
    """Validate that every entry of ``path_list`` resolves to one
    shared DLC project, returning ``(DLCProject, list[Path])``.

    Strict-single-project contract (Roadmap *Next 1.2.0* item 3,
    1.2.0a3 cut): every video in a multi-video session must belong to
    the same DLC project. Bare-video entries, mixed projects, and
    ``config.yaml`` paths all raise ``ValueError`` so the user can fix
    the input rather than landing in an undefined state.

    The returned video-path list is the input order (the user's
    queue), NOT the project's canonical order. Bundle indexing follows
    the queue.

    Raises:
        ImportError: ``deeplabcut`` isn't installed.
        ValueError: Any entry isn't inside a DLC project, or entries
            span multiple projects, or a non-video entry sneaks in.
    """
    if not HAS_DLC:
        raise ImportError(
            "dustrack.open: multi-video sessions require deeplabcut "
            "(every video must belong to a single DLC project)."
        )
    resolved: list[Path] = []
    config_paths: set = set()
    for p in path_list:
        if not p.is_file():
            raise ValueError(
                f"dustrack.open: multi-video entry {p!s} is not a file. "
                "Multi-video sessions accept videos inside one DLC project; "
                "pass a project folder to open every video in the project."
            )
        if _is_dlc_config_yaml(p):
            raise ValueError(
                f"dustrack.open: multi-video entry {p!s} is a DLC "
                "config.yaml. To open every video in a project, pass the "
                "config.yaml (or the project folder) by itself -- not "
                "as a list entry alongside videos."
            )
        cp = _find_dlc_config(p)
        if cp is None:
            raise ValueError(
                f"dustrack.open: multi-video entry {p!s} is not inside a "
                "DLC project. Multi-video sessions require every video to "
                "belong to one shared project."
            )
        config_paths.add(Path(cp).resolve())
        resolved.append(p)
    if len(config_paths) > 1:
        raise ValueError(
            "dustrack.open: multi-video entries span multiple DLC projects "
            f"({sorted(str(c) for c in config_paths)}). All entries must "
            "belong to one shared project."
        )
    project = DLCProject(str(next(iter(config_paths))))
    return project, resolved


# ---------------------------------------------------------------------
# Late-binding proxy for the names that _ensure_dlc_loaded mutates
# on the loader module after import. from dustrack.dlcinterface import
# X snapshots the value at the time of the from statement, which
# was a problem for DLC3 / deeplabcut / VideoWriter /
# ScannerError -- those stay None / False until the lazy
# load completes. Routing attribute access through __getattr__
# returns the live value from the loader.
# ---------------------------------------------------------------------
_LAZY_NAMES = frozenset((
    'DLC3', 'deeplabcut', 'VideoWriter', 'ScannerError',
    '_DLC_LOAD_STATE', '_DLC_LOAD_THREAD',
))

# Names that relocated in the 1.2.0rc1 refactor. Exposed here as a
# lazy attribute proxy so existing ``from dustrack.dlcinterface import
# DUSTrack`` / ``from dustrack.dlcinterface import open`` paths keep
# resolving without forcing dustrack.gui / dustrack.open to load
# eagerly (which would cycle through dlcinterface at module-load
# time -- gui.py imports DLCProject from this module).
_RELOCATED_NAMES = {
    'DUSTrack': ('dustrack.gui', 'DUSTrack'),
    'open': ('dustrack._open', 'open'),
    '_open_seed_session': ('dustrack._open', '_open_seed_session'),
    '_SEED_VIDEO_PATH': ('dustrack._open', '_SEED_VIDEO_PATH'),
    '_is_dlc_config_yaml': ('dustrack._open', '_is_dlc_config_yaml'),
    '_is_dlc_project_root': ('dustrack._open', '_is_dlc_project_root'),
    '_find_dlc_config': ('dustrack._open', '_find_dlc_config'),
    '_find_video_index': ('dustrack._open', '_find_video_index'),
    '_session_inside_dlc_project': ('dustrack._open', '_session_inside_dlc_project'),
    '_attach_bundles_or_fallback': ('dustrack._open', '_attach_bundles_or_fallback'),
    '_resolve_multi_video_from_list': ('dustrack._open', '_resolve_multi_video_from_list'),
}


def __getattr__(name):  # PEP 562 module-level __getattr__
    if name in _LAZY_NAMES:
        return getattr(_dlcloader, name)
    if name in _RELOCATED_NAMES:
        import importlib as _il
        modname, attr = _RELOCATED_NAMES[name]
        return getattr(_il.import_module(modname), attr)
    raise AttributeError(f"module 'dustrack.dlcinterface' has no attribute {name!r}")

