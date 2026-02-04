"""
Main DUSTrack module, including an interface to manage DeepLabCut (DLC) projects.
"""
from __future__ import annotations

import fnmatch
import os
import shutil
from pathlib import Path
from typing import Mapping, Union
import functools

import numpy as np
import pandas as pd
import cv2 as cv
from pyfilemanager import FileManager
import pysampled
from skimage import io
from skimage.util import img_as_ubyte

import matplotlib.pyplot as plt
import datanavigator

from skimage import io, img_as_ubyte
from decord import VideoReader, cpu

from .postprocess import lk_moving_average_filter
from . import _config

try:
    import deeplabcut
    from deeplabcut.utils.auxfun_videos import VideoWriter
    from ruamel.yaml.scanner import ScannerError
    DLC3 = deeplabcut.__version__.startswith('3.')
    HAS_DLC = True
except ImportError:
    print('deeplabcut is not installed. You can still use the optical flow functions with DUSTrack.')
    HAS_DLC = False

import re
from pathlib import PureWindowsPath, PurePosixPath



EXPERIMENTER = _config.EXPERIMENTER


def enhance_ultrasound_image(image, clahe_clip=2.0, clahe_grid=8, gamma=1.0, brightness=0):
    """
    Enhance ultrasound image for better visibility.

    Args:
        image: Input image (RGB or grayscale)
        clahe_clip: CLAHE clip limit (higher = more contrast)
        clahe_grid: CLAHE tile grid size
        gamma: Gamma correction (>1 = brighter midtones, <1 = darker)
        brightness: Brightness offset (-255 to 255)

    Returns:
        Enhanced RGB image for matplotlib display.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    else:
        gray = image

    # Apply CLAHE
    clahe = cv.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    enhanced = clahe.apply(gray)

    # Apply gamma correction
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        enhanced = cv.LUT(enhanced, table)

    # Apply brightness
    if brightness != 0:
        enhanced = np.clip(enhanced.astype(np.int16) + brightness, 0, 255).astype(np.uint8)

    # Convert back to RGB for matplotlib
    return cv.cvtColor(enhanced, cv.COLOR_GRAY2RGB)


class VideoAnnotation(datanavigator.VideoAnnotation):
    """
    Enhanced VideoAnnotation with integrated post-processing.
    
    This subclass extends datanavigator.VideoAnnotation by adding the Lucas-Kanade
    moving average filter as a default post-processing method for smoothing trajectories.
    
    Attributes:
        postprocess: Function reference to lk_moving_average_filter for jitter reduction.
    """
    postprocess = lk_moving_average_filter

class DUSTrack(datanavigator.VideoPointAnnotator):
    """
    Interactive video point annotator with DeepLabCut integration.
    
    DUSTrack provides a GUI for manual annotation of points in videos, with integrated
    support for creating DeepLabCut projects, training models, and post-processing
    results using optical flow algorithms.
    
    Features:
        - Manual point annotation with event marking
        - Real-time trajectory visualization
        - DeepLabCut project creation and training
        - Optical flow-based jitter reduction
        - Multiple annotation layer management
    
    Attributes:
        _dlcproject (DLCProject): Associated DeepLabCut project instance.
        _ax_lims (dict): Stores axis limits when plot axes are frozen.
    
    Example:
        >>> # Basic usage
        >>> tracker = DUSTrack('video.mp4', "pn") # pn is the name of the annotation layer, that can be saved as {video_name}_annotations_pn.json
        >>> 
        >>> # With multiple annotation layers
        >>> tracker = DUSTrack('video.mp4', {
        ...     'manual': 'manual_labels.json',
        ...     'dlc_iter1': 'dlc_predictions.h5'
        ... })
    """
    def __init__(self, *args,
                 clahe_clip=2.0, clahe_grid=8, gamma=1.2, brightness=10,
                 dark_mode=False, enhance_enabled=True, **kwargs):
        # Store enhancement settings
        self._clahe_clip = clahe_clip
        self._clahe_grid = clahe_grid
        self._gamma = gamma
        self._brightness = brightness
        self._enhance_enabled = enhance_enabled
        self._dark_mode = dark_mode

        # Create image processor function
        def image_processor(im):
            if self._enhance_enabled:
                return enhance_ultrasound_image(
                    im, self._clahe_clip, self._clahe_grid,
                    self._gamma, self._brightness
                )
            return im

        kwargs['image_process_func'] = image_processor
        super().__init__(*args, **kwargs)

        for ann in self.annotations:
            ann.__class__ = VideoAnnotation

        self._dlcproject = None
        self._ax_lims = {'state': False, 'x': [None, None], 'y_trace_x': [None, None], 'y_trace_y': [None, None]}

        # Apply dark theme if enabled
        if dark_mode:
            self._apply_dark_theme()

        self.buttons.add(text="Keyboard shortcuts", action_func=(lambda s, ev: s.show_key_bindings(f="new", pos="center left")).__get__(self))
        self._add_dummy_button("dummy1")
        if HAS_DLC:
            self.buttons.add(text="Create DLC Project", action_func=self.create_dlc_project)
            self.buttons.add(text="Train DLC model", action_func=self.process_dlc_project)
            self.buttons.add(text="Reduce jitter", action_func=self.process_with_lk)
            self._add_dummy_button("dummy2")
        self.buttons.add(text="Trace: line", action_func=(lambda s, ev: s.ann.set_plot_type("line")).__get__(self))
        self.buttons.add(text="Trace: dot", action_func=(lambda s, ev: s.ann.set_plot_type("dot")).__get__(self))
        self.buttons.add(text="Freeze plot axes", action_func=self.freeze_plot_axes)
        self.buttons.add(text="Unfreeze plot axes", action_func=self.unfreeze_plot_axes)
        self.buttons.add(text="Replace existing from overlay", action_func=self.copy_existing_annotations_from_overlay)
        self.buttons.add(text="Toggle enhance", action_func=self._toggle_enhancement)

        self.statevariables._text._pos = datanavigator.utils._parse_pos("bottom left")
        
        if self.__class__.__name__ == "DUSTrack":
            plt.show(block=False)
            self.update()
            plt.setp(self._ax_trace_x.get_xticklabels(), visible=False)
            plt.draw()

    def _add_dummy_button(self, name="dummy"):
        """
        Add an invisible placeholder button for GUI layout spacing.

        Args:
            name (str): Internal name for the button. Defaults to "dummy".
        """
        button = self.buttons.add(text=name, action_func=lambda x, ev: None)
        button.ax.patch.set_visible(False)  # Hide the rectangular patch
        button.label.set_visible(False) # Hide the text label
        button.ax.axis('off') # Optional: Turn off the axes frame

    def _apply_dark_theme(self):
        """Apply dark theme to the GUI for better ultrasound visibility."""
        bg_color = '#1a1a1a'
        ax_color = '#2a2a2a'
        text_color = 'white'

        # Figure background
        self.figure.patch.set_facecolor(bg_color)

        # Image axis
        self._ax_image.set_facecolor(ax_color)

        # Trace axes
        for ax in [self._ax_trace_x, self._ax_trace_y]:
            ax.set_facecolor(ax_color)
            ax.tick_params(colors=text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            for spine in ax.spines.values():
                spine.set_color(text_color)

    def _toggle_enhancement(self, event=None):
        """Toggle image enhancement on/off."""
        self._enhance_enabled = not self._enhance_enabled
        self.update()

    def _increase_contrast(self, event=None):
        """Increase CLAHE contrast (clip limit)."""
        self._clahe_clip = min(self._clahe_clip + 0.5, 10.0)
        self.update()

    def _decrease_contrast(self, event=None):
        """Decrease CLAHE contrast (clip limit)."""
        self._clahe_clip = max(self._clahe_clip - 0.5, 1.0)
        self.update()

    def _increase_brightness(self, event=None):
        """Increase image brightness (gamma)."""
        self._gamma = min(self._gamma + 0.1, 3.0)
        self.update()

    def _decrease_brightness(self, event=None):
        """Decrease image brightness (gamma)."""
        self._gamma = max(self._gamma - 0.1, 0.3)
        self.update()

    def freeze_plot_axes(self, event=None):
        """
        Lock the axis limits of trajectory plots to current view.
        
        Useful for comparing tracking quality across different frames without
        automatic axis rescaling distracting from the comparison.
        
        Args:
            event: Mouse/keyboard event (unused, for button compatibility).
        """
        self._ax_lims['state'] = True
        self._ax_lims['x'] = self._ax_trace_x.get_xlim()
        self._ax_lims['y_trace_x'] = self._ax_trace_x.get_ylim()
        self._ax_lims['y_trace_y'] = self._ax_trace_y.get_ylim()
        self.update()

    def unfreeze_plot_axes(self, event=None):
        """
        Restore automatic axis scaling for trajectory plots.
        
        Args:
            event: Mouse/keyboard event (unused, for button compatibility).
        """
        self._ax_lims['state'] = False
        self._ax_lims['x'] = [None, None]
        self._ax_lims['y_trace_x'] = [None, None]
        self._ax_lims['y_trace_y'] = [None, None]
        self.update()


    def create_dlc_project(self, event=None, name=None, path=None, experimenter=_config.EXPERIMENTER) -> DLCProject:
        """
        Create a new DeepLabCut project using current annotations as training labels.
        
        This method saves the current annotation layer and initializes a new DLC project
        with the video and labels. The project is stored in self._dlcproject for subsequent
        training and analysis operations.
        
        Args:
            event: Mouse/keyboard event (unused, for button compatibility).
            name (str, optional): Project name. Defaults to "{video_name}_{annotation_layer}".
            path (str, optional): Directory for project. Defaults to video's parent directory.
            experimenter (str, optional): Experimenter name. Defaults to config value.
        
        Returns:
            DLCProject: The newly created project instance.
        
        Note:
            Project names must contain an underscore for proper DLC configuration handling.
        """
        if not HAS_DLC:
            raise ImportError('deeplabcut is not installed. Cannot create DLC project.')
        self.ann.save()
        if name is None:
            name = f"{self.name}_{self.ann.name}"
        if path is None:
            path = str(Path(self.fname).parent)
        self._dlcproject = DLCProject(
            path=path,
            videos=[self.fname],
            name=name,
            experimenter=experimenter,
            annotation_suffix=self.ann.name,
        )
        return self._dlcproject
    
    def process_dlc_project(self, event=None, *args, **kwargs):
        """
        Train the DeepLabCut model and return to annotation GUI.
        
        This method closes the current figure, trains the DLC model, evaluates it,
        analyzes videos, and reopens the annotation interface with DLC predictions
        as an overlay layer.
        
        Args:
            event: Mouse/keyboard event (unused, for button compatibility).
            *args: Additional arguments passed to DLCProject.process().
            **kwargs: Additional keyword arguments passed to DLCProject.process().
        
        Returns:
            DUSTrack: New annotation interface with DLC predictions loaded.
        
        Raises:
            ValueError: If no DLCProject has been created yet.
        """
        if not HAS_DLC:
            raise ImportError('deeplabcut is not installed. Cannot process DLC project.')
        assert self._dlcproject is not None, "DLCProject not created. Use create_dlc_project() to create it."
        plt.close(self.figure)
        if self._dlcproject is None:
            raise ValueError('DLCProject not created. Use create_dlc_project() to create it.')
        self._dlcproject.process(*args, **kwargs)
        return self._dlcproject.annotate()
    
    def process_with_lk(self, event=None, *args, **kwargs) -> VideoAnnotation:
        """
        Apply Lucas-Kanade optical flow post-processing to reduce tracking jitter.
        
        This method uses the Lucas-Kanade RSTC (Reverse Sigmoid Tracking Correction)
        algorithm to smooth trajectories. The processed annotation is saved and added
        as a new layer, with the original set as overlay for comparison.
        
        Args:
            event: Mouse/keyboard event (unused, for button compatibility).
            *args: Additional arguments passed to lk_moving_average_filter.
            **kwargs: Additional keyword arguments (e.g., window_size) passed to filter.
        
        Returns:
            VideoAnnotation: The smoothed annotation layer.
        
        See Also:
            dustrack.postprocess.lk_moving_average_filter: The filtering algorithm.
        """
        ann_processed = lk_moving_average_filter(self.ann, *args, **kwargs)
        ann_processed.save()
        self.add_annotation_layers(ann_processed)
        self.statevariables["annotation_overlay"].set_state(self.ann.name)
        self.statevariables["annotation_layer"].set_state(ann_processed.name)
        self.update()
        return ann_processed
    
    def copy_existing_annotations_from_overlay(self, event=None):
        """
        Copy overlay annotation points to the current annotation layer for selected frames.
        
        Useful when DLC predictions are more accurate than manual labels.
        Typically used with manual annotations in the primary annotation layer,
        and the model predictions in the overlay layer. Data is copied only for
        frames that exist in the primary annotation layer. Perform this action
        within a specified frame range by selecting an interval.
        
        Args:
            event: Mouse/keyboard event (unused, for button compatibility).
        
        Raises:
            ValueError: If no annotation overlay is currently selected.
        
        Note:
            Only affects the current label. Other labels remain unchanged.
        """
        overlay_name = self._current_overlay
        if overlay_name is None:
            raise ValueError('No annotation overlay selected.')
        overlay_ann = self.annotations[overlay_name]
        current_label = self._current_label
        if (self._current_layer, current_label) in self.events[0].to_dict():
            event_start, event_end = self.events[0].to_dict()[(self._current_layer, current_label)][0]
        else:
            event_start, event_end = 0, self.ann.n_frames - 1
        # if an event is specified, nudge data only in the selected interval
        for frame_num in self.ann.frames:
            if event_start <= frame_num <= event_end:
                self.ann.data[current_label][frame_num] = overlay_ann.data[current_label][frame_num]
        self.update()
    
    def update(self):
        """
        Update the display with current frame and maintain frozen axis limits if set.
        
        Returns:
            Result from parent class update() method.
        """
        ret = super().update()
        if self._ax_lims['state']:
            if self._ax_lims['x'][0] is not None:
                self._ax_trace_x.set_xlim(self._ax_lims['x'])
                self._ax_trace_y.set_xlim(self._ax_lims['x'])
            if self._ax_lims['y_trace_x'][0] is not None:
                self._ax_trace_x.set_ylim(self._ax_lims['y_trace_x'])
            if self._ax_lims['y_trace_y'][0] is not None:
                self._ax_trace_y.set_ylim(self._ax_lims['y_trace_y'])
            plt.draw()
        return ret


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
        vid_paths = FileManager(Path(file_path).parent).add()[f'*{Path(file_path).stem}*_labeled.mp4']
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
        h5_list = FileManager(vid_path.parent).add()[f'{vid_path.stem}*.h5']
        iter_num_to_fname = {int(Path(x).stem.split('_')[-1]):x for x in h5_list}
        if iter_num is None:
            # pick the highest iteration number
            iter_num = max(iter_num_to_fname)
        assert iter_num in iter_num_to_fname
        h5_file = iter_num_to_fname[iter_num]
        print(h5_file)
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
            config_path = deeplabcut.create_new_project(name, experimenter, videos, working_directory=path, copy_videos=True)
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

        # def change_ip(inp_str):
        #     x = inp_str.split('\\')
        #     if len(x[2].split('.')) == 4:
        #         x[2] = _config.NAS_IP
        #         print(f"IP address changed to {_config.NAS_IP}")
        #         return '\\'.join(x)
        #     print("IP address was not changed")
        #     return inp_str

        # if hasattr(_config, "NAS_IP") and _config.NAS_IP is not None:
        #     video_sets = self.config["video_sets"]
        #     new_video_sets = {change_ip(k):v for k,v in video_sets.items()}
        #     self.edit_config(video_sets=new_video_sets)

        # use self config path and each video path to create a new video_sets using rebase_to_config
        video_sets = self.config["video_sets"]
        new_video_sets = {rebase_to_config(self.config_path, k):v for k,v in video_sets.items()}
        self.edit_config(video_sets=new_video_sets)

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
        model_folder_name = 'dlc-models-pytorch' if DLC3 else 'dlc-models'
        evaluation_folder_name = 'evaluation-results-pytorch' if DLC3 else 'evaluation-results'
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
        return deeplabcut.auxiliaryfunctions.read_config(self.config_path)
    
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
        map_file = os.path.join(self.paths['project'], 'dlc_trackermap.txt')
        if os.path.exists(map_file):
            with open(map_file, 'r', encoding='utf-8-sig') as f:
                trackermap = [x.split(' - ') for x in f.read().splitlines() if x]
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
        return deeplabcut.auxiliaryfunctions.edit_config(config_file, kwargs)

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
        if DLC3:
            ext = ".pt"
        else:
            ext = ".index"
    
        ret = {}
        for iteration_num in self.all_iterations:
            source_path = self.paths['models'] / f'iteration-{iteration_num}'
            snapshot_filenames = FileManager(source_path).add()[f'*train/snapshot*{ext}']
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
        deeplabcut.add_new_videos(self.config_path, videos, copy_videos=True)
        self.copy_annotations(videos)
        return self
    
    def copy_annotations(self, video_name: Union[Path, list]):
        """
        Copy DUSTrack/VideoPointAnnotator JSON files into project's video folder.
        
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
        print(a_name)
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
                fm = FileManager(self.paths['videos']).add()
                file_names = fnmatch.filter([Path(x).name for x in fm.all_files], pattern)
                annotation_file_names = sorted([fm[file_name][0] for file_name in file_names])
                # annotation_file_names = sorted(FileManager(self.paths['videos']).add()[f'{video_stem}*_annotations*.json'])
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
                deeplabcut.check_labels(self.config_path) # this creates an _labeled folder, which doesn't seem necessary in this case
        
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
        if DLC3:
            cfg_name = "pytorch_config"
        else:
            cfg_name = "pose_cfg"
        cfg_files = FileManager(self.paths['models'] / f'iteration-{iteration_num}').add()[f'*{type_}/{cfg_name}*']
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
        
        if DLC3:
            source_path = self.paths['models'] / f'iteration-{iteration_num}'
            snapshot_filenames = FileManager(source_path).add()[f'*train/snapshot*.pt']
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
        column_name = 'test rmse_pcutoff' if DLC3 else 'Test error(px)'
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
        ext = '.pt' if DLC3 else '.index'
        init_weights_files = FileManager(source_path).add()[f'*train/snapshot-*{source_snapshot}{ext}']
        assert len(init_weights_files) == 1

        if DLC3:
            self.edit_config(cfg_file, resume_training_from=init_weights_files[0].removesuffix('.index'))
        else:
            self.edit_config(cfg_file, init_weights=init_weights_files[0].removesuffix('.index'))
        return self

    def create_training_dataset(self, **kwargs):
        """Call deeplabcut.create_training_dataset."""
        net_type = kwargs.pop('net_type', 'resnet_50')
        deeplabcut.create_training_dataset(self.config_path, net_type=net_type, **kwargs)
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
        deeplabcut.train_network(self.config_path, maxiters=maxiters, max_snapshots_to_keep=max_snapshots_to_keep, pytorch_cfg_updates={"runner.eval_interval": 25},**kwargs)
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
        deeplabcut.evaluate_network(self.config_path, **kwargs)
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

        deeplabcut.analyze_videos(**common_params, save_as_csv=save_as_csv, **kwargs)
        if create_video:
            deeplabcut.create_labeled_video(**common_params)
        
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
            if DLC3:
                maxiters = 1000 # epochs
            else:
                maxiters = 500000

        self.current_iteration = iteration_num

        current_iteration = self.current_iteration
        latest_iteration = self.latest_iteration
        print(f'{current_iteration=}')
        print(f'{latest_iteration=}')
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
                if DLC3:
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

    def annotate(self, video_index: int=0, new_annotation_suffix=None):
        """
        Launch interactive annotation GUI for a video.
        
        Opens DUSTrack interface with existing annotation layers loaded,
        including any DLC predictions as line plot overlays.
        
        Args:
            video_index (int): Index of video in video_list. Defaults to 0.
                Negative indices supported.
            new_annotation_suffix (str, optional): Suffix for new annotation layer.
                Defaults to 'iteration-{N}' where N is the next iteration number.
        
        Returns:
            DUSTrack: Interactive annotation interface.
        
        Note:
            Creates a 'buffer' layer for temporary annotations.
            Latest DLC predictions are automatically set as overlay.
        """
        if video_index < 0:
            video_index = len(self.video_list) + video_index
        assert 0 <= video_index < len(self.video_list)
        # print the video name, the base name of the video
        print(f"Video name: {Path(self.video_list[video_index]).stem}")

        
        if new_annotation_suffix is None:
            if self.latest_iteration_is_trained():
                new_iteration_num = self.latest_iteration + 1
            else:
                new_iteration_num = self.latest_iteration
            new_annotation_suffix = f'iteration-{new_iteration_num}'
        
        fm_annotations = VideoFileManager(self, video_index)
        annotation_names = fm_annotations.get_all_annotation_layers(new_annotation_suffix)
        annotation_names['buffer'] = fm_annotations.get_new_json('buffer')
        ret = DUSTrack(self.video_list[video_index], annotation_names, height_ratios=(3,1,1))
        
        # change dlc inference annotations to line plots
        for ann in ret.annotations:
            if 'dlc_' in ann.name:
                ann.set_plot_type('line')

        # set the overlay to the latest dlc inference annotations
        ann_names = [x.name for x in ret.annotations if 'dlc_' in x.name]
        if ann_names:
            ret.statevariables['annotation_overlay'].set_state(ann_names[-1])
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


def _extract_frames(video_file_name: str, frame_idx: list, output_path: str, coords: list):
    """
    Legacy frame extraction using DLC's VideoWriter (OpenCV-based).
    
    Note:
        This function is kept for backwards compatibility but
        _extract_frames_decord is now used by default for better
        performance, and because of discrepancy in extracted frames (seeking
        issues) when using OpenCV vs decord.
    
    Args:
        video_file_name (str): Path to video file.
        frame_idx (list): Frame numbers to extract (0-indexed).
        output_path (str): Directory to save extracted frames.
        coords (list): Crop coordinates [x, y, width, height].
    
    Returns:
        list: Paths to saved image files.
    """
    cap = VideoWriter(video_file_name)
    cap.set_bbox(*map(int, coords))
    indexlength = int(np.ceil(np.log10(len(cap))))
    output_path.mkdir(parents=True, exist_ok=True)
    img_names = []
    for index in frame_idx:
        cap.set_to_frame(index)  # extract a particular frame
        frame = cap.read_frame(crop=True)
        if frame is not None:
            img_name = output_path / f'img{str(index).zfill(indexlength)}.png'
            if not os.path.exists(img_name):
                image = img_as_ubyte(frame)
                io.imsave(img_name, image)
                print(f'{img_name.parent.stem}/{img_name.stem} saved!')
            else:
                print(f'{img_name.parent.stem}/{img_name.stem} already exists. Skipping extraction.')
            img_names.append(img_name)
        else:
            print("Frame", index, " not found!")
    cap.close()
    return img_names

def _extract_frames_decord(video_file_name: str, frame_idx: list, output_path: str, coords: list):
    """
    Extract video frames using Decord library for better performance.
    
    This is the default frame extraction method. It uses batch reading for
    better I/O efficiency compared to OpenCV sequential reading.
    
    Args:
        video_file_name (str): Path to video file.
        frame_idx (list): Frame numbers to extract (0-indexed).
        output_path (str): Directory to save extracted frames.
        coords (list): Crop coordinates. Interpreted as:
            - [x1, y1, x2, y2] if values look like absolute corners
            - [x, y, width, height] otherwise
    
    Returns:
        list: Paths to saved image files.
    
    Note:
        Skips extraction if image file already exists.
        Handles invalid frame indices gracefully.
    """
    # No need to set a bridge; default 'native' is fine and we use .asnumpy()
    vr = VideoReader(video_file_name, ctx=cpu(0), num_threads=1)  # HWC RGB uint8
    n_frames = len(vr)
    indexlength = max(1, int(np.ceil(np.log10(max(1, n_frames)))))

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    def _crop(img, coords):
        if not coords:
            return img
        x0, y0, c2, c3 = map(int, coords)
        h, w = img.shape[:2]

        # Interpret coords as [x1,y1,x2,y2] if c2/c3 look like absolute corners; else [x,y,w,h]
        if (c2 > x0) and (c3 > y0) and (c2 <= w) and (c3 <= h):
            x1, y1, x2, y2 = x0, y0, c2, c3
        else:
            x1, y1 = x0, y0
            x2, y2 = x0 + c2, y0 + c3

        # Clamp to bounds
        x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return img
        return img[y1:y2, x1:x2]

    # Keep only valid indices (and preserve order)
    valid_indices = [int(i) for i in frame_idx if isinstance(i, (int, np.integer)) and 0 <= int(i) < n_frames]

    img_names = []
    if not valid_indices:
        for idx in frame_idx:
            print(f"Frame {idx} not found!")
        return img_names

    # Batch fetch for consistent seeking & speed
    batch = vr.get_batch(valid_indices).asnumpy()  # (N, H, W, 3) RGB uint8
    for k, idx in enumerate(valid_indices):
        frame = batch[k]
        if coords:
            frame = _crop(frame, coords)

        image = frame if frame.dtype == np.uint8 else img_as_ubyte(frame)
        img_name = output_path / f'img{str(idx).zfill(indexlength)}.png'
        if not os.path.exists(img_name):
            io.imsave(str(img_name), image)
            print(f'{img_name.parent.stem}/{img_name.stem} saved!')
        else:
            print(f'{img_name.parent.stem}/{img_name.stem} already exists. Skipping extraction.')
        img_names.append(img_name)

    return img_names



def get_annotation_file_name(video_file_name: Path, annotation_suffix: str='') -> Union[str, None]:
    """
    Get full path to annotation file if it exists.
    
    Args:
        video_file_name (Path): Video file path.
        annotation_suffix (str): Annotation suffix (e.g., 'manual', 'refined').
    
    Returns:
        str or None: Full path if file exists, None otherwise.
    """
    annotation_file_name = make_annotation_file_name(video_file_name, annotation_suffix)
    if os.path.exists(annotation_file_name):
        return annotation_file_name
    return None

def make_annotation_file_name(video_file_name: Path, annotation_suffix: str='') -> str:
    """
    Construct annotation filename from video filename and suffix.
    
    Args:
        video_file_name (Path): Video file path.
        annotation_suffix (str): Annotation suffix. Empty string means no suffix.
    
    Returns:
        str: Full path to annotation file (may not exist yet).
    
    Example:
        >>> make_annotation_file_name('video.mp4', 'manual')
        'video_annotations_manual.json'
        >>> make_annotation_file_name('video.mp4', '')
        'video_annotations.json'
    """
    v = Path(video_file_name)
    annotation_file_name = v.parent / f'{v.stem}_annotations{"_" if annotation_suffix else ""}{annotation_suffix}.json'
    return annotation_file_name


class VideoFileManager(FileManager):
    """
    File manager for organizing annotation and result files for one video.
    
    Provides convenient access to all files associated with a video in a DLC project:
    - Manual annotation JSON files
    - DLC prediction HDF5 files
    - Labeled data files for training
    
    Attributes:
        project_name (str): Name of the DLC project.
        video_stem (str): Video filename without extension.
        video_fname (str): Full path to video file.
    """
    def __init__(self, d: DLCProject, video_index: int):
        """
        Initialize file manager for a specific video.
        
        Args:
            d (DLCProject): Parent DLC project.
            video_index (int): Index of video in project's video list.
        """
        if not HAS_DLC:
            raise ImportError("Install deeplabcut to use VideoFileManager.")
        
        base_dir = d.paths['project']
        super().__init__(base_dir, exclude_hidden=True)
        self.add()
        self.project_name = d.name
        self.video_stem = d.video_names[video_index]
        self.video_fname = d.video_list[video_index]
    
    @property
    def annotations(self) -> dict:
        """
        Map annotation names to file paths.
        
        Returns:
            dict: {annotation_name: file_path} for all JSON annotation files.
        """
        # name_to_path = {Path(x).stem: x for x in self.all_files}
        pattern = f'*{self.video_stem}*_annotations*.json'
        file_names = fnmatch.filter([Path(x).name for x in self.all_files], pattern)
        files = [self[file_name][0] for file_name in file_names]
        # files = [name_to_path[file_name] for file_name in file_names]
        # files = self[f'{self.video_stem}*_annotations*.json']
        return {self._get_annotation_name(fname):fname for fname in files}
    
    @property
    def annotation_files(self) -> list:
        """List of full paths to annotation JSON files."""
        return list(self.annotations.values())
    
    @property
    def annotation_names(self) -> list:
        """List of annotation layer names (without paths or extensions)."""
        return list(self.annotations.keys())
    
    @staticmethod
    def _get_annotation_name(fname):
        """
        Extract annotation name from filename.
        
        Args:
            fname (str): Full path like 'video_annotations_manual.json'.
        
        Returns:
            str: Name part after '_annotations_' (e.g., 'manual').
        """
        return Path(fname).stem.split('_annotations')[-1].removesuffix('.json').strip('_')
    
    @staticmethod
    def _get_video_name(fname):
        """Return the 'name' of the video file <video_name>_annotations_<name>.json.
        For example, C:\\video01_annotations_brachialis_praneeth.json will return video01
        """
        return Path(fname).stem.split('_annotations')[0]
    
    @property
    def dlc_traces(self) -> dict:
        """
        Map DLC trace names to HDF5 file paths.
        
        Returns:
            dict: {trace_name: file_path} for all DLC prediction files.
                Trace names format: 'dlc_iteration-{N}_{training_iter}'
        """
        fm_temp = FileManager(str(Path(self.base_dir) / "videos")).add()
        # fnames = fm_temp[f'{self.video_stem}*{self.project_name}*.h5']
        # I want both .h5 file and json file
        fnames = fm_temp[f'{self.video_stem}DLC*{self.project_name}*.h5'] + fm_temp[f'{self.video_stem}DLC*{self.project_name}*.json']
        return {self._get_dlc_trace_name(fname): fname for fname in fnames}
    
    @property
    def dlc_trace_files(self):
        """List of full paths to DLC prediction HDF5 files."""
        return list(self.dlc_traces.values())
    
    @property
    def dlc_trace_names(self):
        """List of DLC trace identifiers."""
        return list(self.dlc_traces.keys())
    
    @staticmethod
    def _get_dlc_trace_name(fname):
        """
        Extract standardized DLC trace name from filepath.
        
        Args:
            fname (str): Path like 'videos/iteration-0/video_dlc_250000.h5'.
        
        Returns:
            str: Name like 'dlc_iteration-0_250000'.
        """
        return 'dlc_' + Path(fname).parts[-2] + '_' + Path(fname).stem.split('_')[-1]

    @property
    def labeled_data(self):
        """
        Path to HDF5 file containing training labels in DLC format.
        
        Returns:
            str: Full path to CollectedData HDF5 file.
        
        Raises:
            AssertionError: If file doesn't exist or multiple files found.
        """
        fm_temp = FileManager(str(Path(self.base_dir) / "labeled-data")).add()
        ret = fm_temp[f'{self.video_stem}*CollectedData*.h5']
        assert len(ret) == 1
        return ret[0]

    def get_new_json(self, new_suffix) -> Path:
        """
        Create path for a new annotation file with given suffix.
        Used to generate the the filename for the next refinement iteration.
        
        Args:
            new_suffix (str): Suffix for new annotation layer.
        
        Returns:
            Path: Full path to new JSON file.
        
        Raises:
            ValueError: If file with this suffix already exists.
        """
        annotations_json_new = (
            Path(self.video_fname).parent / 
            f'{self.video_stem}_annotations_{new_suffix}.json'
            )
        if os.path.exists(annotations_json_new):
            raise ValueError(f'File with {new_suffix} suffix already exists!')
        return annotations_json_new
    
    def get_all_annotation_layers(self, new_annotation_suffix: str=None):
        """
        Collect all annotation sources for loading into DUSTrack.
        
        Args:
            new_annotation_suffix (str, optional): Suffix for a new layer to create.
        
        Returns:
            dict: Maps layer names to file paths, including:
                - Existing JSON annotation files
                - New empty layer (if suffix provided)
                - Labeled training data
                - DLC prediction HDF5 files
        """
        if new_annotation_suffix is None:
            new_json = {}
        else:
            new_json = {new_annotation_suffix : self.get_new_json(new_annotation_suffix)}
        
        try:
            labeled_data = dict(labeled_data=self.labeled_data)
        except AssertionError:
            labeled_data = {}
        
        return dict(
            **self.annotations, 
            **new_json,
            **labeled_data, 
            **self.dlc_traces
            )


def merge_annotations_in_folder(path, annotation_suffix='merged'):
    """
    Merge multiple annotation files for each video in a folder.
    
    Useful for combining annotations from multiple annotators or sessions.
    Creates a single merged JSON file for each video.
    
    Args:
        path (str): Directory containing videos and annotation JSON files.
        annotation_suffix (str): Suffix for merged output files. Defaults to 'merged'.
    """
    fm = FileManager(path).add_by_depth(0)
    all_names = [Path(x).name for x in fm.all_files]
    all_video_names = fnmatch.filter(all_names, '*.mp4')
    video_files = [fm[name][0] for name in all_video_names]
    for video_file in video_files:
        video_stem = Path(video_file).stem.split('_annotations')[0]
        pattern = f'{video_stem}*_annotations*.json'
        file_names = fnmatch.filter(all_names, pattern)
        annotation_file_names = sorted([fm[file_name][0] for file_name in file_names])
        if len(annotation_file_names) == 0:
            continue
        print(f'Merging {len(annotation_file_names)} files for {video_stem}:')
        print(annotation_file_names)
        print(make_annotation_file_name(video_file, annotation_suffix))
        ann = VideoAnnotation.from_multiple_files(
            fname_list = annotation_file_names,
            vname = str(Path(path) / video_file),
            name = annotation_suffix,
            fname_merged = make_annotation_file_name(video_file, annotation_suffix)
        )
        ann.save()



def rebase_to_config(config_path: str, old_path: str) -> str:
    """
    Rebase 'old_path' (some file inside the project) onto the project root
    implied by 'config_path' (points to config.yaml or the project dir).

    Keeps the correct root/anchor:
      - Posix: leading "/"
      - Windows: drive letters (e.g., "C:\\") and UNC shares ("\\\\server\\share")
    Returns separators inferred from 'config_path'.
    """
    # Choose path flavor by the config path
    is_windows_like = ("\\" in config_path) or config_path.startswith("\\\\") or bool(re.match(r"^[A-Za-z]:", config_path))
    PathCls = PureWindowsPath if is_windows_like else PurePosixPath

    # Parse the config path *as-is* to keep its anchor
    cfg = PathCls(config_path)
    # Project root is the directory that contains config.yaml; if a directory is passed, use it
    new_root = cfg.parent if cfg.name.lower() == "config.yaml" else cfg
    if not new_root.name:
        raise ValueError(f"Cannot infer project folder name from: {config_path!r}")
    project_name = new_root.name

    # Split helper that handles both slash types
    split = lambda p: [x for x in re.split(r"[\\/]+", p.strip()) if x]

    old_parts = split(old_path)

    # Find the LAST occurrence of the project folder name (exact, then case-insensitive)
    def find_idx(parts, name):
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == name:
                return i
        name_cf = name.casefold()
        for i in range(len(parts) - 1, -1, -1):
            if parts[i].casefold() == name_cf:
                return i
        return None

    idx = find_idx(old_parts, project_name)
    if idx is None:
        raise ValueError(f"Project folder {project_name!r} not found in old_path: {old_path!r}")

    tail = old_parts[idx + 1:]
    rebased = new_root / PathCls(*tail) if tail else new_root
    return str(rebased)