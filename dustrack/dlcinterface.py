from __future__ import annotations

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
from ruamel.yaml.scanner import ScannerError
from skimage import io
from skimage.util import img_as_ubyte

import deeplabcut
from deeplabcut.utils.auxfun_videos import VideoWriter

import matplotlib.pyplot as plt
import datanavigator

from .postprocess import lk_moving_average_filter
from . import _config


EXPERIMENTER = _config.EXPERIMENTER
DLC3 = deeplabcut.__version__.startswith('3.')


class VideoAnnotation(datanavigator.VideoAnnotation):
    """
    A subclass of VideoAnnotation that adds a method for applying a moving average filter to the annotations.
    """
    postprocess = lk_moving_average_filter


class DUSTrack(datanavigator.VideoPointAnnotator):
    """
    A subclass of VideoPointAnnotator that uses the DUSTrack algorithm for tracking.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for ann in self.annotations:
            ann.__class__ = VideoAnnotation
        
        self._dlcproject = None

        self.buttons.add(text="Keyboard shortcuts", action_func=(lambda s, ev: s.show_key_bindings(f="new", pos="center left")).__get__(self))
        self._add_dummy_button("dummy1")
        self.buttons.add(text="Create DLC Project", action_func=self.create_dlc_project)
        self.buttons.add(text="Train DLC model", action_func=self.process_dlc_project)
        self.buttons.add(text="Reduce jitter", action_func=self.process_with_lk)
        self._add_dummy_button("dummy2")
        self.buttons.add(text="Trace: line", action_func=(lambda s, ev: s.ann.set_plot_type("line")).__get__(self))
        self.buttons.add(text="Trace: dot", action_func=(lambda s, ev: s.ann.set_plot_type("dot")).__get__(self))

        self.statevariables._text._pos = datanavigator.utils._parse_pos("bottom left")
        
        if self.__class__.__name__ == "DUSTrack":
            plt.show(block=False)
            self.update()
            plt.setp(self._ax_trace_x.get_xticklabels(), visible=False)
            plt.draw()

    def _add_dummy_button(self, name="dummy"):
        # add a dummy button
        button = self.buttons.add(text=name, action_func=lambda x, ev: None)
        button.ax.patch.set_visible(False)  # Hide the rectangular patch
        button.label.set_visible(False) # Hide the text label
        button.ax.axis('off') # Optional: Turn off the axes frame
    
    def create_dlc_project(self, event=None, name=None, path=None, experimenter=_config.EXPERIMENTER) -> DLCProject:
        """Create a new deeplabcut project with the current annotation layer as labels."""
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
        """Process the deeplabcut project."""
        assert self._dlcproject is not None, "DLCProject not created. Use create_dlc_project() to create it."
        plt.close(self.figure)
        if self._dlcproject is None:
            raise ValueError('DLCProject not created. Use create_dlc_project() to create it.')
        self._dlcproject.process(*args, **kwargs)
        return self._dlcproject.annotate()
    
    def process_with_lk(self, event=None, *args, **kwargs) -> VideoAnnotation:
        ann_processed = lk_moving_average_filter(self.ann, *args, **kwargs)
        ann_processed.save()
        self.add_annotation_layers(ann_processed)
        self.statevariables["annotation_overlay"].set_state(self.ann.name)
        self.statevariables["annotation_layer"].set_state(ann_processed.name)
        self.update()
        return ann_processed


class DLCData(pysampled.Data):
    """
    DeepLabCut sample data class to deal with DLC results (and point-trajectory data in general)
    """
    def __setstate__(self, state):
        """For backwards compatibility."""
        super().__setstate__(state)
        if "coords" in self.meta:
            self.signal_coords = self.meta.pop("coords")
        if "label_names" in self.meta:
            self.signal_names = self.meta.pop("label_names")
    
    @classmethod
    def from_hdf(cls, file_path):
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
            d = DLCProject(r'C:\data_opr02\004_02\ml_models\dlc', name='opr02_s004_muscles', experimenter='praneeth', videos=[<video_list>])
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

        Repeat steps 4 and 5 until satisfied
    """
    def __init__(self, path, videos=[], name='test_01', experimenter=_config.EXPERIMENTER, annotation_suffix='', internal_to_dlc_labels: dict=None):
        """
        If there is no _ in the name, then the config file has issues when dealing with folders on the server. 
        """
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

        def change_ip(inp_str):
            x = inp_str.split('\\')
            if len(x[2].split('.')) == 4:
                x[2] = _config.NAS_IP
                print(f"IP address changed to {_config.NAS_IP}")
                return '\\'.join(x)
            print("IP address was not changed")
            return inp_str

        if hasattr(_config, "NAS_IP") and _config.NAS_IP is not None:
            video_sets = self.config["video_sets"]
            new_video_sets = {change_ip(k):v for k,v in video_sets.items()}
            self.edit_config(video_sets=new_video_sets)
        
        try:
            deeplabcut.auxiliaryfunctions.read_config(self.config_path)
        except ScannerError as s:
            print('Config file is corrupted. Fix it manually.')
            print('If there is no _ in the name, then the config file has issues when dealing with folders on the server. ')

    @property
    def paths(self) -> Mapping[str, Path]:
        """Full paths to the project folder and its subfolders."""
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
        """Configuration file for hte project"""
        return deeplabcut.auxiliaryfunctions.read_config(self.config_path)
    
    @property
    def name(self) -> str:
        """Name of the deeplabcut project."""
        return self.config['Task']

    @property
    def trackers(self) -> list:
        """Return the names of the points being tracked."""
        return self.config['bodyparts']

    @property
    def label_names(self) -> list:
        """Meaningful names for the points being tracked."""
        trackermap = self.trackermap
        return [trackermap[tracker] if tracker in trackermap else tracker for tracker in self.trackers]

    @property
    def trackermap(self):
        """Load meaningful names if a dlc_trackermap.txt file exists in the project folder.
        For convenience (e.g. assigning meaningful names after training) and generalization, 
        points that are being tracked are called point0 - point9.
        dlc_trackermap.txt is used to assign biologically meaningful names to these points.
        """
        map_file = os.path.join(self.paths['project'], 'dlc_trackermap.txt')
        if os.path.exists(map_file):
            with open(map_file, 'r', encoding='utf-8-sig') as f:
                trackermap = [x.split(' - ') for x in f.read().splitlines() if x]
            return {x[0]: x[1] for x in trackermap}
        else:
            return {}
    
    def edit_config(self, config_file=None, **kwargs):
        """Edit the configuration file."""
        if config_file is None:
            config_file = self.config_path
        assert os.path.exists(config_file)
        return deeplabcut.auxiliaryfunctions.edit_config(config_file, kwargs)

    @property
    def video_list(self) -> list[Path]:
        """List videos from the config file."""
        return list(self.config['video_sets'].keys())
    
    @property
    def video_names(self) -> list[str]:
        """List of video names (not full paths) for the videos listed in the config file."""
        return [Path(vname).stem for vname in self.video_list]
    
    @property
    def current_iteration(self) -> int:
        """Model iteration number specified in the configuration file."""
        return self.config['iteration']
    
    @current_iteration.setter
    def current_iteration(self, iteration_num: int):
        """Edit config.yaml file to set the current model iteration."""
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
        """Most recent model iteration, based on the sub-folders in dlc-models."""
        all_iterations = self.all_iterations
        if not all_iterations:
            return 0
        return self.all_iterations[-1]
    
    @property
    def latest_trained_iteration(self) -> int:
        """Return the most recent iteration that is trained."""
        return max([iteration for iteration,snapshot in self.all_snapshots.items() if len(snapshot)], default=-1)
    
    @property
    def all_iterations(self) -> list:
        """iterations in the dlc-models folder."""
        ret = [int(x.split('-')[-1]) for x in os.listdir(self.paths['models']) if x.startswith('iteration-') and os.path.isdir(self.paths['models'] / x)]
        ret.sort()
        return ret

    @property
    def all_snapshots(self) -> Mapping[int, list[int]]:
        """Return a dictionary mapping from model iteration number to list of training iterations in that model iteration."""
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
        """If the model iteration specified in the configuration file is trained."""
        return self.iteration_is_trained(self.current_iteration)
    
    def latest_iteration_is_trained(self) -> bool:
        """If the most recent iteration in the dlc-models folder is trained."""
        return self.iteration_is_trained(self.latest_iteration)

    def iteration_is_trained(self, iteration_num: int) -> bool:
        if iteration_num not in self.all_snapshots:
            return False
        return len(self.all_snapshots[iteration_num]) > 0
    
    def increment_iteration(self):
        """If the latest iteration is trained, increment the iteration number in the configuration file."""
        self.current_iteration = 'next'
        return self
        
    def add_videos(self, videos: list[Path]):
        """Add videos to the dlc project"""
        if isinstance(videos, (str, Path)):
            videos = [videos]
        deeplabcut.add_new_videos(self.config_path, videos, copy_videos=True)
        self.copy_annotations(videos)
        return self
    
    def copy_annotations(self, video_name: Union[Path, list]):
        """If frames were labeled using VideoPointAnnotator, then copy those files into the DLC project folder as well."""
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
        """Extract labeled data frames, and save the annotations in the labeled-data folder for each video."""
        annotation_file_names_input = annotation_file_names
        for video_file_name in self.video_list:
            coords = self.config["video_sets"][video_file_name]["crop"].split(",")
            video_stem = Path(video_file_name).stem
            output_path = self.paths['labels'] / video_stem

            if annotation_file_names_input is None:
                annotation_file_names = sorted(FileManager(self.paths['videos']).add()[f'{video_stem}*_annotations*.json'])
                # ignore the *correction* files. In theory, no training is to be done after the dlccorr files are created, but just being careful.
                annotation_file_names = [x for x in annotation_file_names if "_dlccorr" not in x]
                print(f'Loading annotations from {len(annotation_file_names)} file(s): ')
                print([Path(x).stem for x in annotation_file_names])
                print()
            
            if len(annotation_file_names) == 0:
                # there are multiple videos, but one of them doensn't have any labels
                continue
            
            ann = VideoAnnotation.from_multiple_files(
                fname_list = annotation_file_names,
                vname = video_file_name,
                name = suffix_merged,
                fname_merged = make_annotation_file_name(video_file_name, suffix_merged)
            )

            if save_merged_json:
                ann.save()
            _extract_frames(video_file_name, ann.frames, output_path, coords)
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
        """Return the full path to the pose_cfg.yaml in the dlc-models folder. Return the path in the train folder by defaults"""
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
        """Return the training iteration number with the lowest test error for a given model iteration number iteration_num."""
        if iteration_num is None:
            iteration_num = self.current_iteration
        
        if DLC3:
            source_path = self.paths['models'] / f'iteration-{iteration_num}'
            snapshot_filenames = FileManager(source_path).add()[f'*train/snapshot*.pt']
            snapshot_numbers = [int(Path(x).stem.split('-')[-1]) for x in snapshot_filenames]
            best_snapshot_number = [int(Path(x).stem.split('-')[-1]) for x in snapshot_filenames if "best" in Path(x).stem]
            if best_snapshot_number:
                return best_snapshot_number[0]
            return snapshot_numbers[-1]

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
        """Return the test error at the best snapshot."""
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
        """Return the index (not training iteration number) of the snapshot with the lowest test error."""
        if iteration_num is None:
            iteration_num = self.current_iteration
        best_snapshot = self.get_best_snapshot(iteration_num)
        return self.all_snapshots[iteration_num].index(best_snapshot)

    def initialize_weights(self, source_iteration: int=None, source_snapshot: int=None, dest_iteration: int=None):
        """Initialize the model weights, for example, from a previous model iteration when refining the model.
        This method edits the pose_cfg file to set the initial weights.

        Args:
            source_iteration (int, optional): Model iteration number to copy weights from. 
                Defaults to previous to last iteration number in the dlc-models folder.
                If there are less than two iterations, then the this method does nothing.
            source_snapshot (int, optional): Training iteration number from which to copy the weights. 
                Defaults to the best snapshot in source_iteration.
            dest_iteration (int, optional): Model iteration number to copy the weights into. 
                Defaults to the last iteration in the dlc-models folder.

        Returns:
            self: For chaining commands.
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
        init_weights_files = FileManager(source_path).add()[f'*train/snapshot-{source_snapshot}{ext}']
        assert len(init_weights_files) == 1
        self.edit_config(cfg_file, init_weights=init_weights_files[0].removesuffix('.index'))
        return self

    def create_training_dataset(self, **kwargs):
        """Call deeplabcut.create_training_dataset."""
        net_type = kwargs.pop('net_type', 'resnet_50')
        deeplabcut.create_training_dataset(self.config_path, net_type=net_type, **kwargs)
        return self

    def train(self, **kwargs):
        """Train the model. By default, it sets a different number of iterations and max learning rate from deeplabcut."""
        maxiters = kwargs.pop('maxiters', 500000)
        max_snapshots_to_keep = kwargs.pop('max_snapshots_to_keep', 20)
        cfg_file = self.get_pose_cfg_file()
        self.edit_config(cfg_file, multi_step = [[0.005, 10000], [0.02, 350000], [0.002, 425000], [0.001, 1000000]])
        deeplabcut.train_network(self.config_path, maxiters=maxiters, max_snapshots_to_keep=max_snapshots_to_keep, **kwargs)
        return self
    

    def dlc2_train(self, **kwargs):
        """Train the model. By default, it sets a different number of iterations and max learning rate from deeplabcut."""
        maxiters = kwargs.pop('maxiters', 500000)
        max_snapshots_to_keep = kwargs.pop('max_snapshots_to_keep', 20)
        multi_step = kwargs.pop('multi_step', [[0.005, 10000], [0.02, 350000], [0.002, 425000], [0.001, 1000000]])
        batch_size = kwargs.pop('batch_size', 1)
        display_iters = kwargs.pop('display_iters', 1000)
        save_iters = kwargs.pop('save_iters', 50000)
        
        cfg_file = self.get_pose_cfg_file()
        self.edit_config(cfg_file, multi_step=multi_step, batch_size=batch_size, display_iters=display_iters, save_iters=save_iters)
        deeplabcut.train_network(self.config_path, maxiters=maxiters, max_snapshots_to_keep=max_snapshots_to_keep, **kwargs)
        return self
    
    def evaluate(self, **kwargs):
        """Evaluates all the snapshots."""
        current_snapshotindex_value = self.config['snapshotindex']
        self.edit_config(snapshotindex='all')
        deeplabcut.evaluate_network(self.config_path, **kwargs)
        self.edit_config(snapshotindex=current_snapshotindex_value)
        return self

    def analyze_videos(self, iteration_num=None, snapshotindex=None, **kwargs):
        """Predict points in the videos, and create a labeled video in the videos folder."""
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
        
        current_snapshotindex_value = self.config['snapshotindex']
        self.edit_config(snapshotindex=snapshotindex)
        
        common_params = dict(
            config     = self.config_path, 
            videos     = self.video_list, 
            destfolder = self.paths['videos'] / f'iteration-{iteration_num}'
            )

        deeplabcut.analyze_videos(**common_params, save_as_csv=save_as_csv, **kwargs)
        deeplabcut.create_labeled_video(**common_params)
        
        self.edit_config(snapshotindex=current_snapshotindex_value)
        return self

    def process(self, iteration_num=None, maxiters=None, refine=True, source_snapshot=None, **kwargs):
        """Main method that tries to take the best course of action based on the state of the project."""
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
            return self.evaluate().analyze_videos()

        self.extract_frames() # do this every time in case there are any updates to the manual annotations.
        
        if self.latest_iteration_is_trained():
            self.increment_iteration() # increment iteration in the config.yaml file
        
        if not os.path.exists(self.paths['training_data'] / f'iteration-{self.current_iteration}'):
            self.create_training_dataset()
        
        if refine:
            if not self.latest_iteration_is_trained() and self.current_iteration == self.latest_iteration:
                if source_snapshot is not None:
                    source_iteration = self.latest_iteration - int(not self.latest_iteration_is_trained())
                else:
                    source_iteration = None
                self.initialize_weights(source_iteration=source_iteration, source_snapshot=source_snapshot)

        if not self.current_iteration_is_trained():
            try:
                if DLC3:
                    self.train(epochs=maxiters, **kwargs)
                else:
                    self.dlc2_train(maxiters=maxiters, **kwargs)
            except KeyboardInterrupt:
                pass

        return self.evaluate().analyze_videos()
    
    def annotate(self, video_index: int=0, new_annotation_suffix=None):
        """Annotate a video."""
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
        return DUSTrack(self.video_list[video_index], annotation_names, height_ratios=(3,1,1))

    def get_trajectories(self, videos=None, iteration=None):
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
        """Open the project folder"""
        os.system(f'explorer.exe "{str(Path(self.config_path).parent)}"')


def _extract_frames(video_file_name: str, frame_idx: list, output_path: str, coords: list):
    """Extract a set of frames. This code is borrowed from DLC."""
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

def get_annotation_file_name(video_file_name: Path, annotation_suffix: str='') -> Union[str, None]:
    """Return the full path to the annotation file if it exists give the video file name, otherwise return None."""
    annotation_file_name = make_annotation_file_name(video_file_name, annotation_suffix)
    if os.path.exists(annotation_file_name):
        return annotation_file_name
    return None

def make_annotation_file_name(video_file_name: Path, annotation_suffix: str='') -> str:
    v = Path(video_file_name)
    annotation_file_name = v.parent / f'{v.stem}_annotations{"_" if annotation_suffix else ""}{annotation_suffix}.json'
    return annotation_file_name


class VideoFileManager(FileManager):
    """Manage files associated with one video in a DLCProject."""
    def __init__(self, d: DLCProject, video_index: int):
        base_dir = d.paths['project']
        super().__init__(base_dir, exclude_hidden=True)
        self.add()
        self.project_name = d.name
        self.video_stem = d.video_names[video_index]
        self.video_fname = d.video_list[video_index]
    
    @property
    def annotations(self) -> dict:
        files = self[f'{self.video_stem}*_annotations*.json']
        return {self._get_annotation_name(fname):fname for fname in files}
    
    @property
    def annotation_files(self) -> list:
        """Return a list of the full file paths."""
        return list(self.annotations.values())
    
    @property
    def annotation_names(self) -> list:
        return list(self.annotations.keys())
    
    @staticmethod
    def _get_annotation_name(fname):
        """Return the 'name' of the annotation file *_annotations_<name>.json.
        For example, C:\\video01_annotations_brachialis_praneeth.json will return brachialis_praneeth
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
        fm_temp = FileManager(str(Path(self.base_dir) / "videos")).add()
        # fnames = fm_temp[f'{self.video_stem}*{self.project_name}*.h5']
        fnames = fm_temp[f'{self.video_stem}DLC*{self.project_name}*.h5']
        return {self._get_dlc_trace_name(fname): fname for fname in fnames}
    
    @property
    def dlc_trace_files(self):
        return list(self.dlc_traces.values())
    
    @property
    def dlc_trace_names(self):
        return list(self.dlc_traces.keys())
    
    @staticmethod
    def _get_dlc_trace_name(fname):
        """Return the 'name' of the deeplabcut trace <model_iteration>_<training_iteration>.
        For example, dlc_iteration-0_250000 
        """
        return 'dlc_' + Path(fname).parts[-2] + '_' + Path(fname).stem.split('_')[-1]

    @property
    def labeled_data(self):
        """HDF5 file containing labels in deeplabcut format, used to train and test models."""
        fm_temp = FileManager(str(Path(self.base_dir) / "labeled-data")).add()
        ret = fm_temp[f'{self.video_stem}*CollectedData*.h5']
        assert len(ret) == 1
        return ret[0]

    def get_new_json(self, new_suffix) -> Path:
        """Add a new manual labels layer after the latest trained iteration."""
        annotations_json_new = (
            Path(self.video_fname).parent / 
            f'{self.video_stem}_annotations_{new_suffix}.json'
            )
        if os.path.exists(annotations_json_new):
            raise ValueError(f'File with {new_suffix} suffix already exists!')
        return annotations_json_new
    
    def get_all_annotation_layers(self, new_annotation_suffix: str=None):
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
