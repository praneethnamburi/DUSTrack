"""File management: DLC-aware path helpers + VideoFileManager.

Brings together the path / file-name logic for one video inside a
DLC project:

* :func:`_extract_frames` / :func:`_extract_frames_decord` -- the two
  frame-extraction backends DLC uses to populate ``labeled-data/``.
* :func:`get_annotation_file_name` / :func:`make_annotation_file_name`
  -- canonical ``<stem>_annotations[_<suffix>].json`` naming.
* :class:`VideoFileManager` -- a :class:`pyfilemanager.FileManager`
  subclass that enumerates a video's annotation JSONs + DLC h5 traces +
  labeled-data folders inside a DLC project root. Single source of
  truth for the layer-name <-> filepath mapping.
* :func:`merge_annotations_in_folder` -- batch-merge multi-annotator
  JSONs in a folder.
* :func:`rebase_to_config` -- path rewriter for moving DLC projects
  between machines (Posix/Windows/UNC path flavors).

Extracted from ``dlcinterface.py`` in dustrack 1.2.0rc1.
"""
from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Union

import numpy as np
import pyfilemanager
from skimage import img_as_ubyte, io

from datanavigator import VideoReader, cpu

from . import dlcloader as _dlcloader
from ._layer_names import _is_dense_layer_name
from .dlcloader import HAS_DLC, _ensure_dlc_loaded
from .lk_filter import lk_moving_average_filter  # noqa: F401 -- kept for back-compat with callers of VideoFileManager who hot-import siblings
from .annotations import VideoAnnotation


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
    _ensure_dlc_loaded()
    cap = _dlcloader.VideoWriter(video_file_name)
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
    # No need to set a bridge; default 'native' is fine and we use .asnumpy().
    # Force pix_fmt='rgb24' so DLC's labeled-data folder gets 3-channel PNGs
    # even when the source is monochrome-encoded (dnav 1.5.0a2 would
    # otherwise auto-detect gray and write 1-channel PNGs that DLC's
    # ResNet-50 backbone can't ingest).
    vr = VideoReader(video_file_name, ctx=cpu(0), num_threads=1, pix_fmt='rgb24')  # HWC RGB uint8
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


class VideoFileManager(pyfilemanager.FileManager):
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
        pattern = f'*{self.video_stem}*_annotations*.json'
        file_names = fnmatch.filter([Path(x).name for x in self.all_files], pattern)
        files = [self[file_name][0] for file_name in file_names]
        return {self.canonical_layer_name(fname): fname for fname in files}

    @property
    def annotation_files(self) -> list:
        """List of full paths to annotation JSON files."""
        return list(self.annotations.values())

    @property
    def annotation_names(self) -> list:
        """List of annotation layer names (without paths or extensions)."""
        return list(self.annotations.keys())

    @staticmethod
    def canonical_layer_name(fname) -> str:
        """Single source of truth for DUSTrack layer names derived from a filepath.

        The DUSTrack workflow produces three categories of layer file:

        - Manual / hand-edited annotations: ``<video>_annotations[_<name>].json``.
          Returns the suffix after ``_annotations`` (or empty string if absent),
          which is what users picked when they saved.
        - DLC prediction traces: live under ``videos/iteration-{N}/`` and have
          ``DLC`` in the stem. Returns ``'dlc_iteration-{N}_<last underscore-token of stem>'``.
          This pattern also catches LK-RSTC post-processed outputs, which
          inherit the DLC source stem -- so a jitter-reduced layer gets a
          deterministic ``dlc_iteration-{N}_<window>`` name rather than the
          ``"noname"`` fallback that ``VideoAnnotation.__init__`` produces for
          paths without ``_annotations``.
        - Anything else: the file stem.

        Called by :attr:`annotations` / :attr:`dlc_traces` at fresh-load
        time AND by :meth:`DUSTrack._adopt_layer` for in-session adds, so
        the name a user sees in the layer panel is identical regardless
        of whether the layer was discovered on disk or produced live.
        """
        p = Path(fname)
        stem = p.stem
        if '_annotations' in stem:
            return stem.split('_annotations')[-1].removesuffix('.json').strip('_')
        if 'DLC' in stem:
            return 'dlc_' + p.parts[-2] + '_' + stem.split('_')[-1]
        return stem

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
        fm_temp = pyfilemanager.FileManager(str(Path(self.base_dir) / "videos")).add()
        fnames = fm_temp[f'{self.video_stem}DLC*{self.project_name}*.h5'] + fm_temp[f'{self.video_stem}DLC*{self.project_name}*.json']
        return {self.canonical_layer_name(fname): fname for fname in fnames}

    @property
    def dlc_trace_files(self):
        """List of full paths to DLC prediction HDF5 files."""
        return list(self.dlc_traces.values())

    @property
    def dlc_trace_names(self):
        """List of DLC trace identifiers."""
        return list(self.dlc_traces.keys())

    @property
    def labeled_data(self):
        """
        Path to HDF5 file containing training labels in DLC format.
        
        Returns:
            str: Full path to CollectedData HDF5 file.
        
        Raises:
            AssertionError: If file doesn't exist or multiple files found.
        """
        fm_temp = pyfilemanager.FileManager(str(Path(self.base_dir) / "labeled-data")).add()
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
    fm = pyfilemanager.FileManager(path).add_by_depth(0)
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

