"""Annotation data classes for DUSTrack.

Three closely-coupled classes lifted from ``pointtracking.py`` in
dustrack 1.2.0rc1:

* :class:`_TrackedFrameDict` -- a dict subclass that bumps a revision
  counter on every mutation so cached per-frame views (the trace pane,
  the scatter overlay) can invalidate without scanning the whole
  annotation. Critical for the dnav 1.4.0 fast-render cache invariant.

* :class:`VideoAnnotation` -- the data + IO container. Loads JSON
  manual annotations + DLC HDF5 trace files; serialises back to JSON;
  produces :class:`pysampled.Data` and DLC-shaped trace arrays;
  hosts the per-frame scatter + per-label trace artists.

* :class:`VideoAnnotations` -- an :class:`AssetContainer` of
  :class:`VideoAnnotation` keyed by layer name.

The Lucas-Kanade postprocess shortcut (``VideoAnnotation.postprocess``)
is attached in ``dustrack/__init__.py`` rather than here, so this
module stays free of LK-RSTC concerns.
"""

from __future__ import annotations

import functools
import json
import os
import weakref
from pathlib import Path
from typing import Any, Mapping
from tqdm import tqdm

import numpy as np
import pandas as pd
import pysampled
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter

from datanavigator import utils
from datanavigator.assets import AssetContainer


class _TrackedFrameDict(dict):
    """Per-label frame→location dict that bumps the parent's ``_revision`` on mutation.

    Makes the trace-display cache invariant load-bearing: any write to
    ``ann.data[label][frame]`` (direct or via :meth:`VideoAnnotation.add`)
    is observable to consumers keyed on ``_revision``. The previous
    discipline — "always route writes through :meth:`VideoAnnotation.add` /
    :meth:`VideoAnnotation.remove` / :meth:`VideoAnnotation.add_at_frame`" —
    is now enforced by the data structure itself; the two historical bypass
    sites (``check_labels_with_lk`` and DUSTrack's
    ``copy_existing_annotations_from_overlay``) would no longer have
    silently shipped a stale UI.

    Parent reference is a :class:`weakref.ref` so the per-frame dict
    does not extend the lifetime of its owning :class:`VideoAnnotation`.
    """

    def __init__(self, *args, _parent=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._parent_ref = None if _parent is None else weakref.ref(_parent)

    def _bump(self):
        if self._parent_ref is None:
            return
        parent = self._parent_ref()
        if parent is None:
            return
        # _revision is hoisted before the data setter fires, so it
        # exists by the time any mutation routes through here. The
        # getattr guard is belt-and-suspenders for partially-constructed
        # parents (e.g. classmethod factory paths).
        if hasattr(parent, "_revision"):
            parent._revision += 1

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._bump()

    def __delitem__(self, key):
        super().__delitem__(key)
        self._bump()

    def pop(self, key, *args):
        had = key in self
        result = super().pop(key, *args)
        if had:
            self._bump()
        return result

    def popitem(self):
        result = super().popitem()
        self._bump()
        return result

    def clear(self):
        if not self:
            return
        super().clear()
        self._bump()

    def update(self, *args, **kwargs):
        before = len(self)
        super().update(*args, **kwargs)
        # Bump unconditionally if anything was passed; cheap and avoids
        # missing in-place value changes for already-present keys.
        if args or kwargs:
            self._bump()
        else:
            # update() with no args is a no-op
            _ = before

    def setdefault(self, key, default=None):
        was_missing = key not in self
        result = super().setdefault(key, default)
        if was_missing:
            self._bump()
        return result

    def __reduce__(self):
        # For pickle / dill round-tripping: drop the weakref and
        # restore as a parent-less :class:`_TrackedFrameDict`. The
        # owning :class:`VideoAnnotation` (if pickled too) rewraps
        # via its property setter on restore.
        return (_TrackedFrameDict, (dict(self),))


class VideoAnnotation:
    """
    Manage one point annotation layer in a video.

    Each annotation layer can contain up to 10 labels, which are string representations of digits 0-9.
    Each label is a dictionary mapping a frame number to a 2D location on the video frame.

    Args:
        fname (str, optional): File name of the annotations (.json) file. If it
            doesn't exist, it will be created when the save method is used. If this is a
            video file, `fname` will default to `<video_name>_annotations.json`.
            This can also be a DeepLabCut `.h5` file (either labeled data OR predicted trace).
            Defaults to None.
        vname (str, optional): Name of the video being annotated. Defaults to None.
        name (str, optional): Name of the annotation (something meaningful, e.g., `<muscle_name>_<scorer>`
            such as `brachialis_praneeth`). Defaults to None.
        n_labels (int, optional): Number of labels for annotation. Defaults to 10.
        **kwargs: Additional optional parameters:
            - `palette_name` (str, default='Set2'): Color scheme to use. Defaults to 'Set2' from seaborn.
            - `ax_list` (list, default=[]): If specified, the annotation display will be initialized on these axes.
            Alternatively, use :py:meth:`VideoAnnotation.setup_display` to specify the axis list and colors.
            - `preloaded_json` (dict, optional): The result of `VideoAnnotation._load_json` (in case you prefer
            to pickle the JSON files).

    Methods:
        to_dlc(): Convert from JSON file format into a DeepLabCut DataFrame format, and optionally save the file.
    """

    def __init__(
        self,
        fname: str | None = None,
        vname: str | None = None,
        name: str | None = None,
        n_labels: int = 1,
        **kwargs,
    ) -> None:
        self.fname, vname = self._parse_inp(fname, vname, name)

        if self.fname is not None:
            self.fstem = Path(self.fname).stem
        else:
            self.fstem = None

        if name is None:
            if self.fstem is None:
                self.name = "noname"  # default name
            else:
                if "_annotations_" in self.fstem:
                    self.name = self.fstem.split("_annotations_")[-1]
                else:
                    self.name = "noname"
        else:
            assert isinstance(name, str)
            self.name = name

        # 1.2.0a2: callers (notably ``_DUSTrackBase.add_annotation_layers``)
        # can hand in an already-open reader via the ``video=`` kwarg so
        # every annotation layer of a session shares the browser's single
        # open file instead of opening it once per layer (3 av.opens per
        # ``utils.Video(...)`` × ~6 layers was the dominant cold-open
        # cost on M:). When ``video`` is supplied we trust the caller
        # and skip the ``utils.is_video`` OpenCV probe too.
        passed_video = kwargs.pop("video", None)
        if passed_video is not None:
            self.video = passed_video
        elif utils.is_video(vname):
            self.video = utils.Video(vname)
        else:
            self.video = None

        # Bumped on every mutation of self.data. Consumers (e.g.
        # _DUSTrackBase.update_frame_marker) read this to invalidate
        # caches keyed on per-label trace contents. Over-invalidates on
        # ordering-only changes (sort_labels / sort_data); under-invalidates
        # would be a correctness bug. Hoisted before the data setter so
        # _TrackedFrameDict._bump can find it during initial wrapping.
        self._revision = 0

        preloaded_json = kwargs.pop("preloaded_json", None)
        if preloaded_json is None:
            self.data = self.load(n_annotations=n_labels)
        else:
            self.data = preloaded_json

        self._original_palette = utils.get_palette(
            kwargs.pop("palette_name", "Set2"), n_colors=1000
        )  # seaborn Set 2
        self.plot_handles = {
            "ax_list_scatter": kwargs.pop("ax_list_scatter", []),
            "ax_list_trace_x": kwargs.pop("ax_list_trace_x", []),
            "ax_list_trace_y": kwargs.pop("ax_list_trace_y", []),
        }
        self._plot_type = "dot"  # line or dot
        self.setup_display()

    @property
    def data(self) -> dict[str, _TrackedFrameDict]:
        """Per-label frame→location dictionary.

        Reads behave like a normal ``dict[str, dict[int, list[float]]]``.
        Writes through ``ann.data[label][frame] = ...`` automatically
        bump :attr:`_revision`, keeping consumers' caches consistent.
        See :class:`_TrackedFrameDict`.
        """
        return self._data

    @data.setter
    def data(self, value: dict) -> None:
        self._data = self._wrap_label_dicts(value)

    def _wrap_label_dicts(self, raw: dict) -> dict[str, _TrackedFrameDict]:
        """Wrap each per-label inner dict as a :class:`_TrackedFrameDict`
        bound to ``self``.

        Idempotent: an inner dict already bound to ``self`` is reused
        as-is; a foreign-bound or bare dict is rewrapped. Inner-dict
        wrapping is what makes the cache invariant load-bearing — the
        outer-dict reassignment (``ann.data = {...}``) is already
        disciplined via the wholesale-replacement sites that bump
        :attr:`_revision` explicitly.
        """
        out = {}
        for label, frames in raw.items():
            if isinstance(frames, _TrackedFrameDict):
                parent = None if frames._parent_ref is None else frames._parent_ref()
                if parent is self:
                    out[label] = frames
                    continue
            out[label] = _TrackedFrameDict(frames, _parent=self)
        return out

    @classmethod
    def from_multiple_files(
        cls, fname_list: list[str], vname: str, name: str, fname_merged: str, **kwargs
    ) -> VideoAnnotation:
        """Merge annotations from multiple files.
        If multiple files contain an annotation label for the same frame, values from the last file will be kept.
        """
        ann_list: list[VideoAnnotation] = [
            cls(fname, vname, name, **kwargs) for fname in fname_list
        ]
        assert len({ann.video.name for ann in ann_list}) == 1

        labels = sorted(list({label for ann in ann_list for label in ann.labels}))

        ret = cls(fname=fname_merged, vname=ann_list[-1].video.fname, name=name)
        # Assemble the merged dict in one shot so the property setter
        # wraps each per-label dict (per-label assignment via
        # ``ret.data[label] = ...`` would write a bare dict into the
        # outer container and break future _revision bumps on direct
        # writes).
        ret.data = {
            label: functools.reduce(
                lambda x, y: {**x, **y}, [ann.data.get(label, {}) for ann in ann_list]
            )
            for label in labels
        }

        return ret

    @staticmethod
    def _parse_inp(fname_inp: Any, vname_inp: Any, name_inp: Any) -> Any:
        if fname_inp is None and vname_inp is None:
            fname, vname = fname_inp, vname_inp  # do nothing, empty annotation
        elif fname_inp is not None and vname_inp is None:
            if utils.is_video(fname_inp):
                vname = fname_inp
                fname = os.path.join(
                    Path(fname_inp).parent, Path(fname_inp).stem + "_annotations.json"
                )
            else:
                fname = fname_inp
                # Try to find the video in the same folder
                vname_potential = os.path.join(
                    Path(fname_inp).parent,
                    utils.removesuffix(Path(fname_inp).stem, "_annotations").split(
                        "_annotations_"
                    )[0]
                    + ".mp4",
                )
                if os.path.exists(vname_potential):
                    vname = vname_potential
                    print(f"Associating video {vname} with the annotation!")
                else:
                    vname = vname_inp
        elif fname_inp is None and vname_inp is not None:
            assert utils.is_video(vname_inp)
            vname = vname_inp
            suffix = "" if name_inp is None else f"_{name_inp}"
            fname = os.path.join(
                Path(vname_inp).parent,
                Path(vname_inp).stem + f"_annotations{suffix}.json",
            )
        elif fname_inp is not None and vname_inp is not None:
            assert utils.is_video(vname_inp)
            fname, vname = fname_inp, vname_inp  # do nothing
        return fname, vname

    def load(self, n_annotations: int = 1, **h5_kwargs) -> dict:
        """Load annotations from a json file, dlc h5 file, or initialize an annotation dictionary if a file doesn't exist.

        DUSTrack-shaped: the DeepLabCut ``.h5`` branch exists in the
        generic class because datanavigator and DUSTrack co-developed.
        The DLC paths will migrate to ``dustrack.VideoAnnotation`` in
        1.3.0 alongside the ``pointtracking.py`` split; the JSON path
        stays here.
        """
        if (self.fname is None) or (not os.path.exists(self.fname)):
            return {str(label): {} for label in range(n_annotations)}

        if Path(self.fname).suffix == ".json":
            return self._load_json(self.fname)

        assert Path(self.fname).suffix == ".h5"
        return self._load_dlc(self.fname, **h5_kwargs)

    @staticmethod
    def _load_json(json_fname: str) -> dict:
        with open(json_fname, "r") as f:
            ret = {}
            for k, v in json.load(f).items():
                ret[k] = {int(frame_num): loc for frame_num, loc in v.items()}
            return ret

    def _load_dlc(self, dlc_fname: str, **kwargs) -> dict:
        """Dispatch a DLC ``.h5`` file (path or already-loaded DataFrame) to the labeled-data or predicted-trace parser.

        DUSTrack-shaped: see :py:meth:`load` — slated to migrate to
        ``dustrack.VideoAnnotation`` in 1.3.0.
        """
        if isinstance(dlc_fname, (str, Path)):
            assert os.path.exists(dlc_fname)
            assert Path(dlc_fname).suffix == ".h5"
            df = pd.read_hdf(dlc_fname)
        else:
            assert isinstance(dlc_fname, pd.DataFrame)
            df = dlc_fname
        if isinstance(df.index, pd.MultiIndex):  # labeled data format
            return self._dlc_df_to_annotation_dict(df, **kwargs)
        return self._dlc_trace_to_annotation_dict(
            df, **kwargs
        )  # predicted points trace

    @staticmethod
    def _dlc_df_to_annotation_dict(
        df: pd.DataFrame,
        remove_label_prefix: str = "point",
        img_prefix: str = "img",
        img_suffix: str = ".png",
    ) -> dict:
        """Convert dlc labeled data dataframe to an annotation dictionary.

        DUSTrack-shaped: parses the DeepLabCut labeled-data MultiIndex
        format; slated to migrate to ``dustrack.VideoAnnotation`` in
        1.3.0.
        """
        if False in [
            utils.removeprefix(x, remove_label_prefix).isdigit()
            for x in df.columns.levels[1]
        ]:
            label_orig_to_internal = {
                x: str(xcnt) for xcnt, x in enumerate(df.columns.levels[1].tolist())
            }
        else:
            label_orig_to_internal = {
                x: utils.removeprefix(x, remove_label_prefix)
                for x in df.columns.levels[1].tolist()
            }

        frames_str = [
            utils.removesuffix(utils.removeprefix(x, img_prefix), img_suffix)
            for x in df.index.levels[-1]
        ]

        data = {label: {} for label in label_orig_to_internal.values()}
        video_stem = df.index.levels[1].values[0]
        scorer = df.columns.levels[0].values[0]
        for label_orig, label_internal in label_orig_to_internal.items():
            for frame_str in frames_str:
                coord_val = [
                    df.loc[
                        "labeled-data",
                        video_stem,
                        f"{img_prefix}{frame_str}{img_suffix}",
                    ][scorer, label_orig, coord_name]
                    for coord_name in ("x", "y")
                ]
                if np.all(np.isnan(coord_val)):
                    continue
                data[label_internal][int(frame_str)] = coord_val

        return data

    @staticmethod
    def _dlc_trace_to_annotation_dict(
        df: pd.DataFrame, remove_label_prefix: str = "point"
    ) -> dict:
        """Convert dlc labeled trace dataframe (result of analyze_videos) to an annotation dictionary.

        DUSTrack-shaped: parses the DeepLabCut predicted-trace format;
        slated to migrate to ``dustrack.VideoAnnotation`` in 1.3.0.

        Vectorised 2026-05-20 (1.2.0 cold-open optimisation): the
        pre-1.2.0 implementation called ``coords.loc[frame].values`` per
        frame inside a nested loop, ~73 k pandas-xs calls for a
        36 k-frame video at 2 labels. Profile attributed 58 % of the
        ``g.annotate()`` cold-open to this method on the
        ``interosseous_pn24-x`` config. The replacement pulls each
        label's (x, y) columns to numpy once, masks all-NaN rows, and
        builds the per-label dict in one comprehension -- same
        semantics, no pandas-xs calls. Parity-tested against the
        legacy implementation in
        ``tests/test_dlc_trace_vectorise.py``.
        """
        if False in [
            utils.removeprefix(x, remove_label_prefix).isdigit()
            for x in df.columns.levels[1]
        ]:
            label_orig_to_internal = {
                x: str(xcnt) for xcnt, x in enumerate(df.columns.levels[1].tolist())
            }
        else:
            label_orig_to_internal = {
                x: utils.removeprefix(x, remove_label_prefix)
                for x in df.columns.levels[1].tolist()
            }

        frames = df.index.to_numpy()
        scorer = df.columns.levels[0].values[0]

        data: dict = {}
        for label_orig, label_internal in label_orig_to_internal.items():
            # Column-slice once to numpy: shape (n_frames, 2).
            coords = df.loc[:, (scorer, label_orig, ["x", "y"])].to_numpy()
            # An "absent" frame in the pre-vectorised implementation
            # was an all-NaN row (the legacy ``if np.all(np.isnan(...))``
            # branch); plus the ``frame in coords.index`` guard, which
            # is trivially true here since ``coords`` was sliced from
            # ``df`` itself and inherits its full index.
            valid = ~np.all(np.isnan(coords), axis=1)
            valid_frames = frames[valid]
            valid_coords = coords[valid]
            data[label_internal] = {
                # Cast frame to a plain Python int when the index is
                # an integer numpy type so the dict keys match the
                # legacy ``frame in coords.index`` behaviour. Otherwise
                # honour whatever type the DataFrame's index carries.
                (int(f) if isinstance(f, np.integer) else f): pair.tolist()
                for f, pair in zip(valid_frames, valid_coords)
            }

        return data

    def __len__(self) -> int:
        """Number of annotations"""
        return len(self.data)

    @property
    def n_frames(self) -> int:
        """Number of frames in the video being annotated"""
        if self.video is None:
            return max(self.frames, default=-1) + 1
        return len(self.video)

    @property
    def n_annotations(self) -> int:
        """Number of points being annotated in the video."""
        return len(self)

    @property
    def labels(self) -> list[str]:
        """Labels of the annotations."""
        return list(self.data.keys())

    @property
    def palette(self) -> list[tuple]:
        """Color palette for the annotations."""
        return [self._original_palette[int(label)] for label in self.labels]

    @property
    def frames(self) -> list[int]:
        """Frame numbers in the video that have annotations."""
        ret = list(
            set([frame for label in self.labels for frame in self.get_frames(label)])
        )
        ret.sort()
        return ret

    @property
    def frames_overlapping(self) -> list[int]:
        """list of frames in the video where all the labels are annotated."""
        ret = list(
            functools.reduce(
                set.intersection, [set(self.get_frames(label)) for label in self.labels]
            )
        )
        ret.sort()
        return ret

    @property
    def plot_type(self) -> str:
        """Type of plot to use for the annotations."""
        return self._plot_type

    @plot_type.setter
    def plot_type(self, plot_type: str) -> None:
        """Set the type of plot to use for the annotations.

        Thin delegate to :meth:`set_plot_type`; the two APIs are kept
        symmetric so a caller using either ends up in the same state
        (visual style applied AND :attr:`_plot_type` recorded).
        """
        self.set_plot_type(plot_type)

    def get_frames(self, label: str) -> list[int]:
        """Return a list of frames that are annotated with the current label."""
        assert label in self.labels
        return list(self.data[label].keys())

    def reload(self) -> None:
        """Drop in-memory state and reload from disk (or start fresh).

        Inverse of :meth:`save`. If :attr:`fname` is ``None`` or the
        file doesn't exist, restores the empty
        ``{str(i): {} for i in range(n)}`` shape via the existing
        :meth:`load` fallback (see the file-missing branch). Wholesale-
        replaces ``self.data`` so the property setter rewraps every
        per-label inner dict as a :class:`_TrackedFrameDict`, and bumps
        :attr:`_revision` explicitly so per-frame caches keyed on
        ``(label_list, _revision)`` invalidate -- the outer setter
        rewraps but does not itself bump (mirrors :meth:`sort_data` and
        :meth:`remove_empty_labels`).
        """
        n = len(self.labels) or 1
        self.data = self.load(n_annotations=n)
        self._revision += 1

    def save(self, fname: str | None = None) -> None:
        """Save the annotations json file. self.fname should be a valid file path.

        Empty-but-declared labels are preserved (written as
        ``"label": {}``). 1.4.0rc2 promoted labels from a derived
        property of "which keys have data" to first-class schema; the
        previous implicit ``self.remove_empty_labels()`` call here is
        gone. Callers that want a lean export (e.g. DUSTrack pre-flight
        before DLC training) still drive :meth:`remove_empty_labels`
        explicitly.
        """
        if fname is None:
            assert self.fname is not None
            fname = self.fname
        # at the moment, saving is only supported for json files through this method
        if Path(fname).suffix != ".json":
            raise ValueError("Supply a json file name.")
        self.sort_data()
        # cast data due to json dump issues
        data = {
            label: {
                int(frame): [float(x) for x in position]
                for frame, position in label_data.items()
            }
            for label, label_data in self.data.items()
        }
        with open(fname, "w") as f:
            json.dump(data, f, indent=4)
        labels_annotations = {label: len(self.data[label]) for label in self.labels}
        print(f"Saved {fname} with labels-n_annotations \n {labels_annotations}")

    def to_json(self) -> None:
        """Alias for save method."""
        fname = self.fname
        assert Path(fname).suffix == ".h5"
        fname = str(Path(fname).with_suffix(".json"))
        self.save(fname)

    def sort_labels(self) -> None:
        """Sort labels in the data dictionary."""
        self.data = dict(sorted(self.data.items(), key=lambda item: int(item[0])))
        self._revision += 1

    def sort_data(self) -> None:
        """Sort annotations by the frame numbers."""
        self.data = {
            label: dict(sorted(self.data[label].items())) for label in self.labels
        }
        self._revision += 1

    def clip_trailing_empty_labels(self) -> None:
        """Remove trailing empty labels from the annotation dictionary."""
        n_labeled_frames = [len(self.data[label]) for label in self.labels]

        def last_nonzero_index(lst):
            for i in range(len(lst) - 1, -1, -1):
                if lst[i] != 0:
                    return i
            return 0  # or raise an exception if needed

        last_index = last_nonzero_index(n_labeled_frames)

        self.data = {
            label: self.data[label]
            for idx, label in enumerate(self.labels)
            if idx <= last_index
        }
        self._revision += 1

    def remove_empty_labels(self) -> None:
        """Remove empty labels from the annotation dictionary."""
        self.data = {
            label: self.data[label]
            for label in self.labels
            if len(self.data[label]) > 0
        }
        self._revision += 1

    def get_values_cv(self, frame_num: int) -> np.ndarray:
        """Return annotations at frame_num in a format for openCV's optical flow algorithms"""
        return np.array(self.get_at_frame(frame_num), dtype=np.float32).reshape(
            (self.n_annotations, 1, 2)
        )

    def _n_digits_in_frame_num(self) -> str:
        """Number of digits to use when constructing a string from the frame number."""
        if self.n_frames is None:
            return "6"
        return str(len(str(self.n_frames)))

    def _frame_num_as_str(self, frame_num: int) -> str:
        """Return the frame umber as a formatted string."""
        return f"{frame_num:0{self._n_digits_in_frame_num()}}"

    def add_at_frame(self, frame_num: int, values: np.ndarray) -> None:
        """Add annotations at a frame, given the annotation values."""
        assert isinstance(frame_num, int)
        values = np.array(values)
        assert values.shape == (self.n_annotations, 2)
        for label, value in zip(self.labels, values):
            self.data[label][frame_num] = list(value)
        self._revision += 1

    def get_at_frame(self, frame_num: int) -> list[list[float]]:
        """Retrieve annotations at a given frame number. If an annotation is not present, nan values will be used."""
        ret = []
        for label in self.labels:
            if frame_num in self.data[label]:
                ret.append(self.data[label][frame_num])
            else:
                ret.append([np.nan, np.nan])
        return ret

    def __getitem__(self, key: str | int) -> dict[int, list[float]] | list[list[float]]:
        """Easy access to specific annotation, or data from a frame number."""
        if key in self.labels:
            return self.data[key]
        if key in self.frames:
            return self.get_at_frame(key)
        raise ValueError(f"{key} is neither an annotation nor a frame with annotation.")

    def to_dlc(
        self,
        scorer: str = "praneeth",
        output_path: str | None = None,
        file_prefix: str | None = None,
        img_prefix: str = "img",
        img_suffix: str = ".png",
        label_prefix: str = "point",
        save: bool = True,
        internal_to_dlc_labels: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Save annotations in deeplabcut format.

        DUSTrack-shaped: emits a DeepLabCut-shaped DataFrame (and writes
        an ``.h5`` if ``save=True``); slated to migrate to
        ``dustrack.VideoAnnotation`` in 1.3.0 alongside the
        ``pointtracking.py`` split.
        """
        if (
            internal_to_dlc_labels is not None
        ):  # label_prefix is ignored if internal_to_dlc_labels are provided
            assert set(internal_to_dlc_labels) == set(self.labels)
            internal_to_dlc_labels = {
                x: internal_to_dlc_labels[x] for x in self.labels
            }  # in case of ordering mishaps
        else:
            internal_to_dlc_labels = {x: f"{label_prefix}{x}" for x in self.labels}

        annotations = self.data

        if output_path is None:
            output_path = Path(self.fname).parent
        output_path = Path(output_path)

        index_length = self._n_digits_in_frame_num()
        img_stems = [
            f"{img_prefix}{x:0{index_length}}{img_suffix}" for x in self.frames
        ]

        row_idx = pd.MultiIndex.from_tuples(
            [("labeled-data", self.video.name, img_stem) for img_stem in img_stems]
        )
        col_idx = pd.MultiIndex.from_product(
            [[scorer], [internal_to_dlc_labels[x] for x in annotations], ["x", "y"]],
            names=["scorer", "bodyparts", "coords"],
        )
        df = pd.DataFrame([], index=row_idx, columns=col_idx)
        for annotation_label_internal, annotation_dict in annotations.items():
            annotation_label_dlc = internal_to_dlc_labels[annotation_label_internal]
            for frame, xy in annotation_dict.items():
                for coord_name, coord_val in zip(("x", "y"), xy):
                    # Single-call df.loc[row, col] = val — the previous
                    # chained-assignment df.loc[row][col] = val will silently
                    # no-op under pandas 3.0 Copy-on-Write.
                    df.loc[
                        (
                            "labeled-data",
                            self.video.name,
                            f"{img_prefix}{frame:0{index_length}}{img_suffix}",
                        ),
                        (scorer, annotation_label_dlc, coord_name),
                    ] = coord_val
        df = df.apply(lambda col: pd.to_numeric(col, errors="coerce"))

        if file_prefix is None:
            file_prefix = self.fstem
        elif file_prefix == "dlc":  # usual dlc name
            file_prefix = f"CollectedData_{scorer}"
        else:
            assert isinstance(file_prefix, str)

        if save:
            labeled_data_file_prefix = str(output_path / file_prefix)
            df.to_csv(labeled_data_file_prefix + ".csv")
            df.to_hdf(labeled_data_file_prefix + ".h5", key="df_with_missing", mode="w")
        return df

    def to_trace(self, label: str) -> np.ndarray:
        """Return a 2d numpy array of n_frames x 2.

        Args:
            label (str): Annotation label.

        Returns:
            np.ndarray: 2d numpy array of number of frames x 2.
                xy position values of uannotated frames will be filled with np.nan.

        Schema-tolerant: a label absent from :attr:`data` is treated as
        all-frames-unannotated and returns the full NaN array. Lets
        cross-layer trace consumers
        (e.g. :meth:`_DUSTrackBase.update_frame_marker`, which
        iterates every annotation layer with one shared label) survive
        a layer that legitimately doesn't carry the active label --
        either a freshly created layer or a layer where the user
        hasn't placed any points for that label yet. 1.4.0rc2: prior
        to the schema-tolerant relaxation this asserted, which
        crashed the corrections-layer flow when the patch had been
        saved with one of its labels empty (save then pruned that
        label, so it disappeared from :attr:`labels` and the assert
        fired).
        """
        ret = np.full([self.n_frames, 2], np.nan)
        if label not in self.data:
            return ret
        for frame_number, frame_xy in self.data[label].items():
            ret[frame_number, :] = frame_xy
        return ret

    def to_traces(self) -> Mapping[str, np.ndarray]:
        """Return annotations as traces (numpy arrays of size n_frames x 2).

        Returns:
            Mapping[str, np.ndarray]: Dictionary mapping labels to 2d numpy arrays.
        """
        return {label: self.to_trace(label) for label in self.labels}

    def to_signal(self, label: str) -> pysampled.Data:
        """Return an annotation as pysampled.Data at the frame rate of the video.

        Args:
            label (str): Annotation label.

        Returns:
            pysampled.Data: Signal sampled at the frame rate of the video being annotated.
        """
        assert self.video is not None
        assert label in self.labels
        return pysampled.Data(self.to_trace(label), sr=self.video.get_avg_fps())

    def to_signals(self) -> Mapping[str, pysampled.Data]:
        """Return annotations as a dictionary of sampled Data signals
        sampled at the frame rate of the video being annotated.

        Returns:
            Mapping[str, pysampled.Data]: Dictionary mapping labels to pysampled.Data.
        """
        return {label: self.to_signal(label) for label in self.labels}

    def to_pysampled(self) -> pysampled.Data:
        """Return annotations as a pysampled.Data object

        Returns:
            pysampled.Data
        """
        return pysampled.Data(
            np.hstack([self.to_signal(label) for label in self.labels]),
            sr=self.video.get_avg_fps(),
            signal_names=self.labels,
            signal_coords=["x", "y"],
        )

    def add_label(
        self,
        label: str | None = None,
        color: tuple[float, float, float] | None = None,
    ) -> None:
        if label is None:  # pick the next available label
            label = f"{len(self.labels)}"
        if label in self.labels:
            print(f"Label {label} already exists in layer {self.name}.")
            return
        assert label.isdigit()

        if int(label) > len(self._original_palette):
            self._original_palette = self._original_palette * 2
        if color is None:
            color = self._original_palette[int(label)]

        # palette size should mirror the number of labels
        assert len(color) == 3
        assert all([0 <= x <= 1 for x in color])

        # Use the data setter (not self.data[label] = {}) so the new
        # per-label dict is wrapped as a _TrackedFrameDict and future
        # direct writes through ann.data[label][frame] correctly bump
        # _revision. Direct assignment to the outer dict would leave a
        # bare dict in place.
        self.data = {**self._data, label: {}}
        self.sort_labels()  # bumps _revision

        self.re_setup_display()

        print(f"Created new label {label} in layer {self.name} with color {color}.")

    def add(self, location: list[float], label: str, frame_number: int) -> None:
        """Add a point annotation (location) of a given label at a frame number."""
        assert len(location) == 2
        self.data[label][frame_number] = list(location)
        self._revision += 1

    def remove(self, label: str, frame_number: int) -> None:
        """Remove a point annotation of a given label at a frame number."""
        assert label in self.labels
        self.data[label].pop(frame_number, None)
        self._revision += 1

    # display management
    def _process_ax_list(self, ax_list, type_: str) -> list:
        """Process the list of axes (or Tier-2 scatter targets) for plots.

        The trace-axis types must still be mpl axes -- traces remain
        matplotlib in Tier 2. ``scatter`` may also be a Qt marker
        group (``QGraphicsItemGroup``) when ``fast_render=True``; the
        actual scatter rendering then routes through
        :class:`_QtScatterArtist`.
        """
        assert type_ in ("scatter", "trace_x", "trace_y")
        if ax_list is None:
            ax_list = self.plot_handles[f"ax_list_{type_}"]
        if isinstance(ax_list, plt.Axes):
            ax_list = [ax_list]
        elif not isinstance(ax_list, (list, tuple)):
            # Single Qt marker group (or any non-Axes / non-list
            # object) -> wrap in a list. The setup_display_scatter
            # branch validates the actual type below.
            ax_list = [ax_list]
        if type_ in ("trace_x", "trace_y"):
            assert all([isinstance(ax, plt.Axes) for ax in ax_list])
        self.plot_handles[f"ax_list_{type_}"] = ax_list
        return ax_list

    def setup_display_scatter(
        self,
        ax_list_scatter=None,
    ) -> None:
        """Setup scatter plot display.

        Each entry in ``ax_list_scatter`` is either:
        - a matplotlib ``Axes`` (Tier 1) -- a ``PathCollection`` is
          created via ``ax.scatter`` and stashed in ``plot_handles``;
        - a Qt ``QGraphicsItemGroup`` (Tier 2) -- a
          :class:`datanavigator._qt._QtScatterArtist` is built on the
          group and stashed instead. Both expose the same
          mpl-PathCollection-shaped API consumed downstream.
        """
        ax_list_scatter = self._process_ax_list(ax_list_scatter, "scatter")
        palette = self.palette

        # instead of len(self.labels) to keep all 10 points, some of them can be nan
        dummy_xy = [np.nan] * len(palette)
        for ax_cnt, ax in enumerate(ax_list_scatter):
            if isinstance(ax, plt.Axes):
                handle = ax.scatter(dummy_xy, dummy_xy, color=palette, picker=5)
            else:
                # Qt marker group -> _QtScatterArtist facade.
                # The marker group carries a back-reference to the
                # _QtImagePane (set by add_marker_group); passing it
                # lets the artist register for pick-adapter
                # hit-testing.
                from datanavigator._qt import _make_qt_scatter_artist_class

                scatter_cls = _make_qt_scatter_artist_class()
                image_pane = getattr(ax, "_image_pane", None)
                handle = scatter_cls(
                    ax,
                    palette,
                    picker_radius=5.0,
                    image_pane=image_pane,
                )
            self.plot_handles[f"labels_in_ax{ax_cnt}"] = handle

    def setup_display_trace(
        self,
        ax_list_trace_x: None | plt.Axes | list[plt.Axes] = None,
        ax_list_trace_y: None | plt.Axes | list[plt.Axes] = None,
    ) -> None:
        """Setup trace plot display."""
        ax_list_trace_x = self._process_ax_list(ax_list_trace_x, "trace_x")
        ax_list_trace_y = self._process_ax_list(ax_list_trace_y, "trace_y")

        if self.n_frames > 0 and len(self.frames) / self.n_frames > 0.8:
            plot_type = "-"
        else:
            plot_type = "o"

        x = np.arange(self.n_frames)
        dummy_y = np.full(self.n_frames, np.nan)
        for ax_cnt, (ax_x, ax_y) in enumerate(zip(ax_list_trace_x, ax_list_trace_y)):
            for label, x_color in zip(self.labels, self.palette):
                if ax_x.bbox.bounds == ax_y.bbox.bounds:  # if they are in the same axis
                    y_color = [1 - tc for tc in x_color]
                else:
                    y_color = x_color
                for coord, this_ax, color in zip(
                    ("x", "y"), (ax_x, ax_y), (x_color, y_color)
                ):
                    handle_name = f"trace_in_ax{coord}{ax_cnt}_label{label}"
                    (self.plot_handles[handle_name],) = this_ax.plot(
                        x, dummy_y, plot_type, color=color
                    )

            # Claim xlim only if no one has explicitly set it yet:
            # set_xlim flips autoscalex_on off, so this is a no-op for any
            # subsequent VideoAnnotation constructed on the same axes (and
            # for axes the user has panned/zoomed). Pressing `r` re-enables
            # autoscale, which restores the first-time-fit behaviour.
            if ax_x.get_autoscalex_on():
                ax_x.set_xlim(0, self.n_frames)

    def setup_display(
        self,
        ax_list_scatter: None | plt.Axes | list[plt.Axes] = None,
        ax_list_trace_x: None | plt.Axes | list[plt.Axes] = None,
        ax_list_trace_y: None | plt.Axes | list[plt.Axes] = None,
    ) -> None:
        """Setup display for scatter and trace plots."""
        self.setup_display_scatter(ax_list_scatter)
        self.setup_display_trace(ax_list_trace_x, ax_list_trace_y)
        self.set_plot_type(self.plot_type)

    def clear_display(self) -> None:
        """Clear the display."""
        for ax_cnt in range(len(self.plot_handles["ax_list_scatter"])):
            self.plot_handles[f"labels_in_ax{ax_cnt}"].remove()
        for ax_cnt in range(len(self.plot_handles["ax_list_trace_x"])):
            for label in self.labels:
                for ax_type in ("x", "y"):
                    handle_name = f"trace_in_ax{ax_type}{ax_cnt}_label{label}"
                    if handle_name in self.plot_handles:
                        self.plot_handles[handle_name].remove()
        plt.draw()

    def re_setup_display(self) -> None:
        """re-establish display elements when adding a label"""
        self.clear_display()
        self.setup_display(
            ax_list_scatter=self.plot_handles["ax_list_scatter"],
            ax_list_trace_x=self.plot_handles["ax_list_trace_x"],
            ax_list_trace_y=self.plot_handles["ax_list_trace_y"],
        )
        # Fresh plot_handles -> the cache key from the prior handles is
        # stale. Invalidate so the next update_display_trace populates
        # the new artists with their ydata.
        self.invalidate_caches()

    def invalidate_caches(self) -> None:
        """Drop per-annotation render caches so the next
        ``update_display_trace`` re-runs the full ydata sweep.

        Recovery hatch for direct ``.data`` mutations from the command
        line (which skip the ``_revision`` bump the trace cache keys
        on). Normal in-code mutations should go through ``add()`` /
        ``remove()`` / ``add_at_frame()`` instead.
        """
        self._trace_display_cache_key = None

    def update_display_scatter(self, frame_number: int, draw: bool = False) -> None:
        """Update scatter plot display."""
        for ax_cnt in range(len(self.plot_handles["ax_list_scatter"])):
            n_pts = len(self.palette)
            scatter_offsets = np.full((n_pts, 2), np.nan)
            scatter_offsets[:, :] = self.get_at_frame(frame_number)
            self.plot_handles[f"labels_in_ax{ax_cnt}"].set_offsets(scatter_offsets)
        if draw:
            plt.draw()

    def update_display_trace(
        self, label: str | None = None, draw: bool = False
    ) -> None:
        """Update trace plot display.

        Trace contents are a pure function of (label_list, self.data);
        none of that changes when only the parent's ``_current_idx``
        moves. Cache on (label_list, self._revision) so per-frame
        navigation skips the to_trace / set_ydata sweep entirely.
        """
        if label is None:
            label_list = self.labels
        else:
            assert label in self.labels
            label_list = [label]

        cache_key = (tuple(label_list), self._revision)
        if getattr(self, "_trace_display_cache_key", None) != cache_key:
            for ax_cnt in range(len(self.plot_handles["ax_list_trace_x"])):
                for label in label_list:
                    trace = self.to_trace(label)
                    for coord_cnt, coord in enumerate(("x", "y")):
                        handle_name = f"trace_in_ax{coord}{ax_cnt}_label{label}"
                        self.plot_handles[handle_name].set_ydata(trace[:, coord_cnt])
            self._trace_display_cache_key = cache_key

        if draw:
            plt.draw()

    def update_display(
        self, frame_number: int, label: str | None = None, draw: bool = False
    ) -> None:
        """Update scatter and trace plot display."""
        self.update_display_scatter(frame_number, draw=False)
        self.update_display_trace(label, draw=False)
        if draw:
            plt.draw()

    # display management - control visibility
    @property
    def _trace_handles(self) -> dict:
        """Dictionary of handle_name - handle for trace handles."""
        return {
            name: handle
            for name, handle in self.plot_handles.items()
            if name.startswith("trace_in_ax")
        }

    @property
    def _label_handles(self) -> dict:
        """Dictionary of handle_name - handle for label handles (image overlay)."""
        return {
            name: handle
            for name, handle in self.plot_handles.items()
            if name.startswith("labels_in_ax")
        }

    @property
    def _trace_or_label_handles(self) -> dict:
        return {**self._trace_handles, **self._label_handles}

    def _set_visibility(self, visibility: bool = True, draw: bool = False) -> None:
        for plot_handle in self._trace_or_label_handles.values():
            plot_handle.set_visible(visibility)
        if draw:
            plt.draw()

    def hide(self, draw: bool = True) -> None:
        """Hide all elements (scatter, traces) in this annotation."""
        self._set_visibility(False, draw)

    def show(self, draw: bool = True) -> None:
        """Show all elements (scatter, traces) in this annotation."""
        self._set_visibility(True, draw)

    def _set_trace_visibility(
        self, label: str, visibility: bool = True, draw: bool = False
    ) -> None:
        for plot_handle_name, plot_handle in self._trace_handles.items():
            if plot_handle_name.endswith(f"_label{label}"):
                plot_handle.set_visible(visibility)
        if draw:
            plt.draw()

    def show_trace(self, label: str, draw: bool = True) -> None:
        """Show trace for a specific label."""
        self._set_trace_visibility(label, True, draw)

    def hide_trace(self, label: str, draw: bool = True) -> None:
        """Hide trace for a specific label."""
        self._set_trace_visibility(label, False, draw)

    def show_one_trace(self, label: str, draw: bool = True) -> None:
        """Show only one trace for a specific label."""
        for this_label in self.labels:
            self._set_trace_visibility(this_label, this_label == label, draw)

    def set_alpha(
        self, alpha: float = 0.4, label: str | None = None, draw: bool = True
    ) -> None:
        """Set the transparency level of all (or one) the traces and labels in this annotation.

        Args:
            alpha (float, optional): alpha value between 0 and 1. Defaults to 0.4.
            label (_type_, optional): Defaults to all labels.
            draw (bool, optional): Update display if True. Defaults to True.
        """
        for handle in self._trace_or_label_handles.values():
            handle.set_alpha(alpha)
        if draw:
            plt.draw()

    def set_plot_type(self, type_: str = "line", draw: bool = True) -> None:
        """Set the plot type for traces.

        Records the choice on :attr:`_plot_type` *and* applies the
        visual style, so subsequent :meth:`re_setup_display` /
        :meth:`setup_display` calls (which read
        ``self.plot_type``) don't revert to the stale value.
        Symmetric with the :attr:`plot_type` property setter.

        Args:
            type_ (str, optional): "line" or "dot". Defaults to "line".
            draw (bool, optional): Update display if True. Defaults to True.
        """
        assert type_ in ("line", "dot")
        self._plot_type = type_
        for trace_handle in self._trace_handles.values():
            if type_ == "line":
                trace_handle.set_linestyle("-")
                trace_handle.set_marker("None")
            else:
                trace_handle.set_linestyle("None")
                trace_handle.set_marker("o")
        if draw:
            plt.draw()

    def clip_labels(self, start_frame: int, end_frame: int) -> None:
        """Remove annotations outside the clip range. Clip range includes start and end frame."""
        self.data = {
            label: {
                k: v
                for k, v in self.data[label].items()
                if start_frame <= k <= end_frame
            }
            for label in self.labels
        }
        self._revision += 1

    def keep_overlapping_continuous_frames(self) -> None:
        """Keep data from consecutive frames that have all labels."""
        x = self.frames_overlapping
        frames_to_keep = sorted(
            set([item for a, b in zip(x, x[1:]) if (b - a) == 1 for item in (a, b)])
        )
        if len(frames_to_keep) == 0:
            print(
                "You're trying to remove all frames! Saving you from yourself by aborting."
            )
            return
        self.data = {
            label: {k: v for k, v in self.data[label].items() if k in frames_to_keep}
            for label in self.labels
        }
        self._revision += 1

    def keep_overlapping_frames(self) -> None:
        """Keep data only from frames where every label is annotated.

        Sibling of :meth:`keep_overlapping_continuous_frames` without
        the consecutive-runs constraint: fully-labeled but isolated
        frames are preserved. Motivating use case is DLC training
        pre-flight, where partial frames degrade the trained model
        even though DLC tolerates per-bodypart NaN in its CSV.
        """
        frames_to_keep = set(self.frames_overlapping)
        if not frames_to_keep:
            print(
                "You're trying to remove all frames! Saving you from yourself by aborting."
            )
            return
        self.data = {
            label: {k: v for k, v in self.data[label].items() if k in frames_to_keep}
            for label in self.labels
        }
        self._revision += 1

    def get_area(
        self, labels: list[str] | str, lowpass: float | None = None
    ) -> pysampled.Data | np.ndarray:
        """Get the area in pixel squared."""

        def PolyArea(x, y):
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        if isinstance(labels, str):
            labels = list(labels)  # e.g. '023' -> ['0', '2', '3']

        for label in labels:
            assert label in self.labels

        if lowpass is None:
            traces = self.to_traces()
        else:
            traces = {
                label: signal.lowpass(lowpass)()
                for label, signal in self.to_signals().items()
            }

        trace_mat = np.asarray([traces[label] for label in labels])
        area_vals = np.array(
            [
                PolyArea(trace_mat[:, xi, 0], trace_mat[:, xi, 1])
                for xi in range(self.n_frames)
            ]
        )

        if self.video is None:  # return np.array
            return area_vals
        return pysampled.Data(area_vals, sr=self.video.get_avg_fps())

    def export_video(self, out_file_name=None, start_frame=None, end_frame=None):
        assert self.video is not None

        if start_frame is None:
            start_frame = 0

        if end_frame is None:
            end_frame = self.n_frames - 1

        if out_file_name is None:
            assert (
                self.fname is not None
            ), "Please provide a file name to save the video."
            out_file_name = str(
                Path(self.fname).parent
                / f"{Path(self.fname).stem}_sf{start_frame}_ef{end_frame}.mp4"
            )
            print(f"Saving video to {out_file_name}")

        dpi = 200
        # video_data = self.video[start_frame:end_frame+1]

        def _read_frame_rgb(i):
            # dnav 1.5.0a2 auto-detects monochrome sources and returns
            # (H, W) gray; downstream matplotlib imshow + ffmpeg encode
            # in this export path want 3-channel RGB. Replicate Y across
            # R/G/B when needed; pass through otherwise.
            arr = self.video[i].asnumpy()
            if arr.ndim == 2:
                import cv2 as _cv2

                arr = _cv2.cvtColor(arr, _cv2.COLOR_GRAY2RGB)
            return arr

        def setup(ann):
            plot_handles = {}
            first_frame = _read_frame_rgb(start_frame)
            ny, nx, _ = first_frame.shape

            figure = plt.figure(frameon=False, figsize=((nx / dpi), (ny / dpi)))
            ax = figure.add_subplot(111)
            dummy_xy = [np.nan] * len(ann.palette)
            plot_handles["im"] = ax.imshow(first_frame)
            plot_handles["scatter"] = ax.scatter(
                dummy_xy,
                dummy_xy,
                color=ann.palette,
                s=4**2,
                edgecolors=[0, 0, 0],
                linewidths=0.3,
            )
            ax.set_xlim(0, nx)
            ax.set_ylim(0, ny)
            ax.axis("off")
            ax.invert_yaxis()
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plot_handles["ax"] = ax
            plot_handles["figure"] = figure
            return plot_handles

        def update(ann, frame_number, plot_handles, scatter_points=None):
            n_pts = len(ann.palette)
            scatter_offsets = np.full((n_pts, 2), np.nan)
            if scatter_points is None:
                scatter_points = ann.get_at_frame(frame_number)
            scatter_offsets[[int(label) for label in ann.labels], :] = scatter_points
            plot_handles["im"].set_data(_read_frame_rgb(frame_number))
            plot_handles["scatter"].set_offsets(scatter_offsets)

        prev_backend = plt.get_backend()
        plt.switch_backend("agg")
        plot_handles = setup(self)

        n_frames = end_frame - start_frame + 1
        writer = FFMpegWriter(fps=self.video.get_avg_fps(), codec="h264")

        assert not os.path.exists(out_file_name)
        p1, p2 = list(self.to_signals().values())
        p1 = p1.lowpass(15)
        p2 = p2.lowpass(15)
        with writer.saving(plot_handles["figure"], out_file_name, dpi=dpi):
            for frame_number in tqdm(
                range(start_frame, end_frame + 1)
            ):  # tqdm(range(ann.n_frames)):
                scatter_points = [
                    list(p1[int(frame_number)]),
                    list(p2[int(frame_number)]),
                ]
                update(self, frame_number, plot_handles, scatter_points=scatter_points)
                writer.grab_frame()

        plt.close(plot_handles["figure"])
        plt.switch_backend(prev_backend)


class VideoAnnotations(AssetContainer):
    def add(
        self, name: str | VideoAnnotation, fname=None, vname=None, **kwargs
    ) -> VideoAnnotation:
        """Create-and-add"""
        if isinstance(name, VideoAnnotation):
            ann = name
        else:
            assert isinstance(name, str)
            ann = VideoAnnotation(fname, vname, name, **kwargs)
        return super().add(ann)

    def reorder(self, names: list[str]) -> None:
        """Permute the underlying ``_list`` so layer names follow ``names``.

        ``names`` must be a permutation of the current
        :attr:`AssetContainer.names`. Callers that own the rotation of a
        ``annotation_layer`` / ``annotation_overlay``
        :class:`StateVariable` (i.e. :class:`_DUSTrackBase` and
        subclasses) are responsible for resyncing those after this
        returns -- ``reorder`` only manages the membership order.
        """
        current = self.names
        if list(names) == current:
            return
        if set(names) != set(current) or len(names) != len(current):
            raise ValueError(
                "reorder(names) requires a permutation of existing layer names; "
                f"got {names!r} for current {current!r}"
            )
        by_name = {ann.name: ann for ann in self._list}
        self._list = [by_name[n] for n in names]
