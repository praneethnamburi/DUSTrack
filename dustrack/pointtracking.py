from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Callable, Mapping, Union

import numpy as np
import pandas as pd
import pysampled
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter

from . import utils
from .assets import AssetContainer
from .videos import VideoBrowser
from .opticalflow import lucas_kanade, lucas_kanade_rstc


class VideoPointAnnotator(VideoBrowser):
    """
    Add point annotations to videos.
    
    Use arrow keys to navigate frames.
    Select a 'label category' from 0 to 9 by pressing the corresponding number key.
    Point your mouse at a desired location in the video and press the forward slash / button to add a point annotation.
    When you're done, press 's' to save your work, which will create a '<video name>_annotations.json' file in the same folder as the video file.
    These annotations will be automagically loaded when you try to annotate this file again.

    If you're doing one label at a time, then pick the frames for the first label arbitrarily.
    For the second label onwards,

    Args:
        vid_name (Path): Path to the video.
        annotation_names (Union[list[str], Mapping[str, Path]], optional):
            list[str] - Name(s) of the annotation layer(s). The file path(s) are deduced from the name(s).
            Mapping[str, Path] - A dictionary mapping of annotation layer names to the annotation file paths.
            Defaults to '' with one layer of annotations.
        titlefunc (Callable, optional): A function used to set the title of the image axis.
            Defaults to a title function specified in :py:class:`VideoBrowser`.
    """

    def __init__(
        self,
        vid_name: Path,
        annotation_names: Union[
            list[str], Mapping[str, Path], list[VideoAnnotation]
        ] = "",
        titlefunc: Callable = None,
        image_process_func: Callable = lambda im: im,
        height_ratios: tuple = (10, 1, 1),  # depends on your screen size
    ):
        figure_handle, (
            self._ax_image,
            self._ax_trace_x,
            self._ax_trace_y,
        ) = plt.subplots(
            3, 1, gridspec_kw=dict(height_ratios=list(height_ratios)), figsize=(10, 8)
        )
        self._ax_trace_x.sharex(self._ax_trace_y)
        (self._frame_marker_x,) = self._ax_trace_x.plot(
            [], [], color="black", linewidth=1
        )
        (self._frame_marker_y,) = self._ax_trace_y.plot(
            [], [], color="black", linewidth=1
        )
        super().__init__(vid_name, titlefunc, self._ax_image, image_process_func)
        self.memoryslots.hide()
        self.memoryslots.disable()

        # annotation layers
        self.annotations = VideoAnnotations(parent=self)
        self.load_annotation_layers(annotation_names)
        if "buffer" in self.annotations.names:
            self.annotations["buffer"].set_plot_type("line")

        # frames of interest
        self.frames_of_interest = []
        (self._plot_frames_of_interest_x,) = self._ax_trace_x.plot(
            [], [], color="gray", linewidth=1, alpha=0.5
        )
        (self._plot_frames_of_interest_y,) = self._ax_trace_y.plot(
            [], [], color="gray", linewidth=1, alpha=0.5
        )

        # State variables
        self.statevariables.add("annotation_layer", self.annotations.names)
        self.statevariables.add("annotation_overlay", [None] + self.annotations.names)
        self.statevariables.add("annotation_label", self.ann.labels)
        self.statevariables.add("number_keys", ["select", "place"])
        self.statevariables.show(pos="top left")

        self.add_events()

        self.set_key_bindings()

        # set mouse click behavior
        self.cid.append(
            self.figure.canvas.mpl_connect("pick_event", self.select_label_with_mouse)
        )
        self.cid.append(
            self.figure.canvas.mpl_connect(
                "button_press_event", self.place_label_with_mouse
            )
        )
        self.cid.append(
            self.figure.canvas.mpl_connect("button_press_event", self.go_to_frame)
        )

        if self.__class__.__name__ == "VideoPointAnnotator":
            plt.show(block=False)
            self.update()
            plt.setp(self._ax_trace_x.get_xticklabels(), visible=False)
            self.figure.tight_layout()
            plt.draw()

    @classmethod
    def from_annotations(cls, annotations: list[VideoAnnotation], *args, **kwargs):
        if isinstance(annotations, VideoAnnotation):
            annotations = [annotations]
        video_names = {a.video.fname for a in annotations}
        assert len(video_names) == 1  # same video across all annotations
        return cls(video_names.pop(), annotations, *args, **kwargs)

    def load_annotation_layers(
        self, annotation_names: Union[list[str], dict[str, Path], list[VideoAnnotation]]
    ):
        """Load data from annotation files if they exist, otherwise initialize annotation layers."""
        if isinstance(annotation_names, (str, VideoAnnotation)):
            annotation_names = [annotation_names]

        if isinstance(annotation_names, list) and all(
            [isinstance(a, VideoAnnotation) for a in annotation_names]
        ):
            annotation_names = {
                a.name: a.fname for a in annotation_names
            }  # re-add because of plotting

        if "buffer" not in annotation_names:
            if isinstance(annotation_names, list):
                annotation_names.append("buffer")
            else:
                annotation_names["buffer"] = self._get_fname_annotations("buffer")

        if isinstance(annotation_names, dict):
            ann_name_fname = annotation_names
        else:
            ann_name_fname = {
                name: self._get_fname_annotations(name) for name in annotation_names
            }

        for name, fname in ann_name_fname.items():
            self.annotations.add(
                name=name,
                fname=fname,
                vname=self.fname,
                ax_list_scatter=[self._ax_image],
                ax_list_trace_x=[self._ax_trace_x],
                ax_list_trace_y=[self._ax_trace_y],
                palette_name="Set2",
            )

        # same set of non-empty labels in all the loaded annotations
        all_labels = []
        for ann in self.annotations._list:
            all_labels += [
                label for label, label_data in ann.data.items() if label_data
            ]
        all_labels = sorted(list(set(all_labels)))
        if (
            not all_labels
        ):  # when starting without any annotations, initialize a full set of empty annotations
            all_labels = [str(x) for x in range(10)]
        for ann in self.annotations._list:
            for label in all_labels:
                if label not in ann.labels:
                    ann.data[label] = {}
            for label in ann.labels:
                if label not in all_labels:
                    assert not ann.data[label]
                    del ann.data[label]
            ann.sort_labels()

    def set_key_bindings(self):
        """Set the keyboard actions."""
        self.add_key_binding("s", self.save, "Save current annotation layer")
        self.add_key_binding("t", self.add_annotation)
        self.add_key_binding("y", self.remove_annotation)

        self.add_key_binding("f", self.increment_if_unannotated)
        self.add_key_binding("g", self.increment)
        self.add_key_binding("d", self.decrement_if_unannotated)

        self.add_key_binding("-", self.previous_annotation_layer)
        self.add_key_binding("=", self.next_annotation_layer)
        self.add_key_binding("[", self.previous_annotation_overlay)
        self.add_key_binding("]", self.next_annotation_overlay)
        self.add_key_binding(";", self.previous_annotation_label)
        self.add_key_binding("'", self.next_annotation_label)
        self.add_key_binding(",", self.previous_frame_with_any_label)
        self.add_key_binding(".", self.next_frame_with_any_label)
        self.add_key_binding("alt+,", self.previous_frame_of_interest)
        self.add_key_binding("alt+.", self.next_frame_of_interest)

        self.add_key_binding(
            "j",
            (lambda s: s.pan(direction="left")).__get__(self),
            description="pan left",
        )
        self.add_key_binding(
            "k",
            (lambda s: s.pan(direction="right")).__get__(self),
            description="pan right",
        )

        self.add_key_binding("`", self.cycle_number_keys_behavior)

        self.add_key_binding("n", self.next_frame_with_current_label)
        self.add_key_binding("p", self.previous_frame_with_current_label)
        self.add_key_binding("b", self.previous_frame_with_current_label)

        self.add_key_binding("m", self.toggle_frame_of_interest)
        self.add_key_binding("c", self.copy_current_annotation_from_overlay)
        # self.add_key_binding('ctrl+alt+c', self.copy_annotations_from_overlay)
        self.add_key_binding("alt+c", self.copy_frames_of_interest_from_buffer)
        self.add_key_binding("ctrl+alt+c", self.copy_frames_in_interval_from_overlay)

        self.add_key_binding(
            "v",
            (lambda s: s.check_labels_with_lk(mode="minimal")).__get__(self),
            "Check labels with LK - minimal mode",
        )
        self.add_key_binding(
            "alt+v",
            (lambda s: s.check_labels_with_lk(mode="current")).__get__(self),
            "Check labels with LK - current label",
        )
        self.add_key_binding(
            "ctrl+alt+v",
            (lambda s: s.check_labels_with_lk(mode="all")).__get__(self),
            "Check labels with LK - all labels",
        )

        self.add_key_binding(
            "a", self.interpolate_with_lk, "Interpolate current point with LK"
        )
        self.add_key_binding(
            "ctrl+a",
            (lambda s: s.interpolate_with_lk(all_labels=True)).__get__(self),
            "Interpolate all points with LK",
        )

        self.add_key_binding(
            "ctrl+d",
            (lambda s: s.interpolate_with_lk_norstc(all_labels=True)).__get__(self),
            "Interpolate all points with LK",
        )

        self.add_key_binding(
            "alt+a", self.remove_labels_in_interval, "Clear current label in interval"
        )
        self.add_key_binding(
            "ctrl+alt+a",
            (lambda s: s.remove_labels_in_interval(all_labels=True)).__get__(self),
            "Clear all labels in interval",
        )

        self.add_key_binding(
            "alt+b",
            (lambda s: s.predict_points_with_lucas_kanade(labels="current")).__get__(
                self
            ),
            "Predict current point with lucas-kanade",
        )
        self.add_key_binding(
            "ctrl+b",
            (lambda s: s.predict_points_with_lucas_kanade(labels="all")).__get__(self),
            "Predict all points with lucas-kanade",
        )

        self.add_key_binding("alt+q", self.keep_overlapping_continuous_frames)

    def add_events(self):
        """Add an event to specify time intervals for interpolating with lucas-kanade."""
        event_name = "interp_with_lk"
        self.events.add(
            name=event_name,
            size=2,
            fname=os.path.join(
                Path(self.fname).parent,
                Path(self.fname).stem + f"_events_{event_name}.json",
            ),
            data_id_func=(lambda s: (s._current_layer, s._current_label)).__get__(self),
            data_func=round,
            color="gray",
            pick_action="overwrite",
            ax_list=[self._ax_trace_x],
            add_key="z",
            remove_key=None,
            save_key=None,
            display_type="fill",
            win_remove=(10, 10),
            show=True,
        )

    @property
    def ann(self) -> VideoAnnotation:
        """Return current annotation layer."""
        return self.annotations[self._current_layer]

    @property
    def _current_label(self) -> str:
        """Return current label '0'-'9'."""
        return self.statevariables["annotation_label"].current_state

    @property
    def _current_layer(self) -> str:
        """Return current annotation layer"""
        return self.statevariables["annotation_layer"].current_state

    @property
    def _current_overlay(self) -> Union[str, None]:
        """Return current annotation overlay layer"""
        return self.statevariables["annotation_overlay"].current_state

    def _get_fname_annotations(self, annotation_name, suffix=".json"):
        """Construct the filename corresponding to an  annotation layer named annotation_name."""
        return os.path.join(
            Path(self.fname).parent,
            Path(self.fname).stem
            + "_annotations"
            + f'{"_" if annotation_name else ""}'
            + annotation_name
            + suffix,
        )

    def __call__(self, event):
        """Callbacks for number keys."""
        super().__call__(event)
        if event.name == "key_press_event":
            if event.key in self.ann.labels:
                self.statevariables["annotation_label"].set_state(str(event.key))
                if self.statevariables["number_keys"].current_state == "place":
                    self.add_annotation(event)
                self.update()
            elif event.key in [str(x) for x in range(10)]:
                label = str(event.key)
                for ann in self.annotations._list:  # add new label to all annotations
                    if label not in ann.labels:
                        ann.add_label(label)
                self.statevariables["annotation_label"].states = self.ann.labels
                self.statevariables["annotation_label"].set_state(label)
                if self.statevariables["number_keys"].current_state == "place":
                    self.add_annotation(event)
                self.update()

    def update(self):
        """Update elements in the UI."""
        self.update_annotation_visibility(draw=False)
        # self.ann.update_display(self._current_idx, draw=False)
        # self.ann.show_one_trace(self._current_label, draw=False)
        self.statevariables.update_display(draw=False)
        self.update_frame_marker(draw=False)
        super().update()
        plt.draw()

    def update_annotation_visibility(self, draw=False):
        """Update the visibility of all annotation layers, for example, when the layer is changed."""
        for ann in self.annotations:
            if ann.name == self._current_layer:
                ann.set_alpha(1, draw=False)
                ann.show(draw=False)
                ann.show_one_trace(self._current_label, draw=False)
                ann.update_display(self._current_idx, draw=draw)
            elif ann.name == self._current_overlay:
                if ann.name != self._current_layer:
                    ann.set_alpha(0.4, draw=False)
                    ann.show(draw=False)
                    ann.show_one_trace(self._current_label, draw=False)
                    ann.update_display(self._current_idx, draw=draw)
            else:
                ann.hide(draw=draw)

    def update_frame_marker(self, draw=False):
        """Update the current frame location in the trace plots."""

        def nanlim(x):
            return [np.nanmin(x) * 0.9, np.nanmax(x) * 1.1]

        def nanlim_small(x, scale=0.6):
            nmin = np.nanmin(x) * 0.9
            nmax = np.nanmax(x) * 1.1
            m = (nmin + nmax) / 2
            return [(nmin - m) * scale + m, (nmax - m) * scale + m]

        trace_data_x, trace_data_y = np.hstack(
            [ann.to_trace(self._current_label).T for ann in self.annotations._list]
        )

        xl, yl = nanlim(trace_data_x), nanlim(trace_data_y)
        xls, yls = nanlim_small(trace_data_x), nanlim_small(trace_data_y)

        self._frame_marker_x.set_data([self._current_idx] * 2, xls)
        self._frame_marker_y.set_data([self._current_idx] * 2, yls)

        self._plot_frames_of_interest_x.set_data(
            *utils.ticks_from_times(self.frames_of_interest, xl)
        )
        self._plot_frames_of_interest_y.set_data(
            *utils.ticks_from_times(self.frames_of_interest, yl)
        )
        # self._ax_trace_x.set_xlim((0, n_frames))
        # if len(self.ann.data[self._current_label]) > 0:
        if not np.any(np.isnan(xl)):
            self._ax_trace_x.set_ylim(xl)
            self._ax_trace_y.set_ylim(yl)
        if draw:
            plt.draw()

    def copy_annotations_from_overlay(self):
        """Copy annotations from the overlay layer into the current layer."""
        ann_overlay = self.annotations[self._current_overlay]
        frame_number = self._current_idx
        for label in self.ann.labels:
            if label in ann_overlay.labels:
                location = ann_overlay.data[label].get(frame_number, None)
                if location is not None:
                    self.ann.add(location, label, frame_number)
        self.update()

    def copy_current_annotation_from_overlay(self):
        """Copy annotations from the overlay layer into the current layer."""
        ann_overlay = self.annotations[self._current_overlay]
        frame_number = self._current_idx
        label = self._current_label
        if label in ann_overlay.labels:
            location = ann_overlay.data[label].get(frame_number, None)
            if location is not None:
                self.ann.add(location, label, frame_number)
        self.update()

    def copy_frames_of_interest_from_buffer(self):
        """copy annotations at frames of interest from buffer into the current layer.
        If there is no buffer, then copy from the overlay layer.
        """
        if "buffer" in self.annotations.names:
            source_ann = self.annotations["buffer"]
        else:
            source_ann = self.annotations[self._current_overlay]

        for frame_number in self.frames_of_interest:
            for label in self.ann.labels:
                if label in source_ann.labels:
                    location = source_ann.data[label].get(frame_number, None)
                    if location is not None:
                        self.ann.add(location, label, frame_number)
        self.update()

    def copy_frames_in_interval_from_overlay(self):
        """For the current label only."""
        start_frame, end_frame = self.get_selected_interval()
        ann_overlay = self.annotations[self._current_overlay]
        label = self._current_label
        if label in ann_overlay.labels:
            for frame_number in range(start_frame, end_frame + 1):
                location = ann_overlay.data[label].get(frame_number, None)
                if location is not None:
                    self.ann.add(location, label, frame_number)
        self.update()

    def _add_annotation(self, location, frame_number=None, label=None):
        """Core function for adding annotations. Allows more control."""
        if frame_number is None:
            frame_number = self._current_idx
        if label is None:
            label = self._current_label
        self.ann.add(location, label, frame_number)

    def add_annotation(self, event):
        """Add annotation at frame. If it exists, it'll get overwritten."""
        if event.inaxes == self._ax_image:
            self._add_annotation([float(event.xdata), float(event.ydata)])
        self.update()

    def remove_annotation(self, event=None):
        """remove annotation at the current frame if it exists"""
        self.ann.remove(self._current_label, self._current_idx)
        self.update()

    def _get_nearest_annotated_frame(self) -> int:
        """Return the nearest frame (in either direction) number with the current label in the current annotation layer."""
        d = {
            abs(x - self._current_idx): x
            for x in self.ann.get_frames(self._current_label)
        }
        return d[min(d)]

    def previous_frame_with_current_label(self, event=None):
        """Go to the previous frame with the current label in the current annotation layer."""
        self._current_idx = max(
            [
                x
                for x in self.ann.get_frames(self._current_label)
                if x < self._current_idx
            ],
            default=self._current_idx,
        )
        self.update()

    def next_frame_with_current_label(self, event=None):
        """Go to the next frame with the current label in the current annotation layer."""
        self._current_idx = min(
            [
                x
                for x in self.ann.get_frames(self._current_label)
                if x > self._current_idx
            ],
            default=self._current_idx,
        )
        self.update()

    def previous_frame_with_any_label(self, event=None):
        """Go to the previous frame with any label in the current annotation layer."""
        self._current_idx = max(
            [x for x in self.ann.frames if x < self._current_idx],
            default=self._current_idx,
        )
        self.update()

    def next_frame_with_any_label(self, event=None):
        """Go to the next frame with any label in the current annotation layer."""
        self._current_idx = min(
            [x for x in self.ann.frames if x > self._current_idx],
            default=self._current_idx,
        )
        self.update()

    def previous_frame_of_interest(self, event=None):
        """Go to the previous frame of interest."""
        self._current_idx = max(
            [x for x in self.frames_of_interest if x < self._current_idx],
            default=self._current_idx,
        )
        self.update()

    def next_frame_of_interest(self, event=None):
        self._current_idx = min(
            [x for x in self.frames_of_interest if x > self._current_idx],
            default=self._current_idx,
        )
        self.update()

    def _update_statevariable_annotation_label(self):
        """Update the annotation_label state variable.
        Used when the annotation_layer is changed,
        and each layer has a different set of labels.
        """
        x = self.statevariables["annotation_label"]
        current_state = x.current_state
        x.states = self.ann.labels
        if current_state not in self.ann.labels:
            x.set_state(0)
        else:
            x.set_state(current_state)

    def previous_annotation_layer(self):
        """Go to the previous annotation layer."""
        self.statevariables["annotation_layer"].cycle_back()
        self._update_statevariable_annotation_label()
        self.update()

    def next_annotation_layer(self):
        """Go to the next annotation layer"""
        self.statevariables["annotation_layer"].cycle()
        self._update_statevariable_annotation_label()
        self.update()

    def previous_annotation_overlay(self):
        """Go to the previous annotation overlay layer."""
        self.statevariables["annotation_overlay"].cycle_back()
        self.update()

    def next_annotation_overlay(self):
        """Go to the next annotation overlay layer."""
        self.statevariables["annotation_overlay"].cycle()
        self.update()

    def previous_annotation_label(self):
        """Set current annotation label to the previous one."""
        self.statevariables["annotation_label"].cycle_back()
        self.update()

    def next_annotation_label(self):
        """Set current annotation label to the next one."""
        self.statevariables["annotation_label"].cycle()
        self.update()

    def cycle_number_keys_behavior(self):
        """Number keys can be used to either select labels, or place a specific label.
        Toggle between these two behaviors.
        """
        self.statevariables["number_keys"].cycle()
        self.update()

    def increment_if_unannotated(self, event=None):
        """Advance the frame if the current frame doesn't have any annotations.
        Useful to pause at annotated frames when adding a new label.
        """
        if self._current_idx not in self.ann.frames:
            self.increment()

    def decrement_if_unannotated(self, event=None):
        """Go to the previous frame if the current frame doesn't have any annotations.
        Useful to pause at annotated frames when adding a new label.
        """
        if self._current_idx not in self.ann.frames:
            self.decrement()

    def save(self):
        """Save current annotation layer json file."""
        self.ann.save()

    def select_label_with_mouse(self, event):
        """Select a label by clicking on it with the left mousebutton."""
        if event.mouseevent.button.name == "LEFT" and len(event.ind == 1):
            self.statevariables["annotation_label"].set_state(str(event.ind[0]))
            print(
                f'Picked {self._current_label} with index {self.statevariables["annotation_label"].current_state} at frame {self._current_idx}'
            )
            self.update()

    def place_label_with_mouse(self, event):
        """Place the selected label with the right mousebutton."""
        if event.inaxes == self._ax and event.button.name == "RIGHT":
            self.add_annotation(event)

    def go_to_frame(self, event):
        if (
            event.inaxes in (self._ax_trace_x, self._ax_trace_y)
            and event.button.name == "RIGHT"
        ):
            self._current_idx = int(event.xdata)
            self.update()

    def toggle_frame_of_interest(self, event):
        """Mark/unmark the current frame as a frame of interest"""
        if event.inaxes in (self._ax_trace_x, self._ax_trace_y):
            frame_number = self._current_idx
            if frame_number in self.frames_of_interest:
                self.frames_of_interest.remove(frame_number)
            else:
                self.frames_of_interest.append(frame_number)
            self.frames_of_interest.sort()
            self.update()

    def keep_overlapping_continuous_frames(self):
        self.ann.keep_overlapping_continuous_frames()
        self.update()

    def predict_points_with_lucas_kanade(
        self, labels="all", start_frame=None, mode="full"
    ):
        """Compute the location of labels at the current frame using Lucas-Kanade algorithm."""
        if labels == "all":
            labels = self.ann.labels
        elif labels == "current":
            labels = [self._current_label]
        elif isinstance(labels, str):  # specify one label
            assert labels in self.ann.labels
            labels = [labels]
        else:  # specify a list of labels
            assert all([label in self.ann.labels for label in labels])

        if start_frame is None:
            start_frame = self._get_nearest_annotated_frame()

        end_frame = self._current_idx  # always predict at the current location

        video = self.data
        init_loc = [self.ann.data[label][start_frame] for label in labels]
        tracked_loc = lucas_kanade(video, start_frame, end_frame, init_loc, mode=mode)
        end_loc_all = tracked_loc[-1]
        for label, end_loc in zip(labels, end_loc_all):
            if end_frame in self.ann.get_frames(label):
                print(f"Updating location for {label} at {end_frame}.")
                print(
                    f"To revert, use v._add_annotation({self.ann.data[label][end_frame]}, frame_number={end_frame}, label='{label}'); v.update()"
                )
            self._add_annotation(end_loc, label=label)
        self.update()
        return tracked_loc

    def get_selected_interval(self):
        start_frame, end_frame = (
            self.events["interp_with_lk"]
            ._data[(self._current_layer, self._current_label)]
            .get_times()[-1]
        )
        return start_frame, end_frame

    def remove_labels_in_interval(self, all_labels=False):
        video = self.data
        if self._current_overlay is None:
            return

        if all_labels:
            label_list = self.annotations[self._current_overlay].labels
        else:
            label_list = [self._current_label]

        start_frame, end_frame = self.get_selected_interval()
        ann_overlay = self.annotations[self._current_overlay]
        for frame_count, frame_number in enumerate(range(start_frame, end_frame + 1)):
            for label in label_list:
                self.ann.remove(label, frame_number)
        self.update()

    def interpolate_with_lk(self, all_labels=False):
        video = self.data
        if self._current_overlay is None:
            return

        if all_labels:
            label_list = self.annotations[self._current_overlay].labels
        else:
            label_list = [self._current_label]

        start_frame, end_frame = self.get_selected_interval()
        ann_overlay = self.annotations[self._current_overlay]
        start_points = [ann_overlay.data[label][start_frame] for label in label_list]
        end_points = [ann_overlay.data[label][end_frame] for label in label_list]
        rstc_path = lucas_kanade_rstc(
            video, start_frame, end_frame, start_points, end_points
        )
        for frame_count, frame_number in enumerate(range(start_frame, end_frame + 1)):
            for label_count, label in enumerate(label_list):
                location = list(rstc_path[frame_count, label_count, :])
                self._add_annotation(location, frame_number, label)
        self.update()

    def interpolate_with_lk_norstc(self, all_labels=False):
        video = self.data
        if self._current_overlay is None:
            return

        if all_labels:
            label_list = self.annotations[self._current_overlay].labels
        else:
            label_list = [self._current_label]

        start_frame, end_frame = self.get_selected_interval()
        ann_overlay = self.annotations[self._current_overlay]
        start_points = [ann_overlay.data[label][start_frame] for label in label_list]
        end_points = [ann_overlay.data[label][end_frame] for label in label_list]
        rstc_path = lucas_kanade(video, start_frame, end_frame, start_points)
        for frame_count, frame_number in enumerate(range(start_frame, end_frame + 1)):
            for label_count, label in enumerate(label_list):
                location = list(rstc_path[frame_count, label_count, :])
                self._add_annotation(location, frame_number, label)
        self.update()

    def check_labels_with_lk(self, mode="minimal"):
        """Interpolate between all labeled frames.
        This only makes sense for sparse-labeled annotations.
        Use this when doing first-time annotations (as opposed to refinement).

        I am testing if refining the start labels using this strategy,
        and augmenting the training data will improve deeplabcut tracking!

        Args:
            mode (str, optional):
                "all"     - LK-interpolation for all labels across all labeled frames
                "current" - LK-interpolation for current label across all labeled frames
                "minimal" - LK-interpolation for current label between labeled frames near the current frame
                Defaults to "minimal".
        """
        assert mode in ("current", "all", "minimal")

        source_ann = self.ann
        if "buffer" in self.annotations.names:
            target_ann = self.annotations["buffer"]

        video = source_ann.video
        if mode == "all":
            label_list = source_ann.labels
        else:
            label_list = [self._current_label]

        rstc_paths = {
            label: np.full((source_ann.n_frames, 2), np.nan) for label in label_list
        }
        for label in label_list:
            frames = utils._List(sorted(list(source_ann[label].keys())))
            if mode == "minimal":
                c = self._current_idx
                n1 = c if c in frames else frames.next(c)
                n2 = frames.next(n1)
                p1 = frames.previous(c)
                p2 = frames.previous(p1)
                frames = sorted(list(set([p2, p1, n1, n2])))
            for start_frame, end_frame in zip(frames, frames[1:]):
                start_points = [source_ann[label][start_frame]]
                end_points = [source_ann[label][end_frame]]
                rstc_path = lucas_kanade_rstc(
                    video, start_frame, end_frame, start_points, end_points
                )
                for frame_count, frame_num in enumerate(
                    range(start_frame, end_frame + 1)
                ):
                    target_ann.data[label][frame_num] = list(
                        rstc_path[frame_count, 0, :]
                    )
                rstc_paths[label][start_frame : end_frame + 1, :] = np.squeeze(
                    rstc_path
                )
        self.update()

    def render(self, start_frame, end_frame):
        out_vid_name = Path(self.ann.fname).with_suffix(".mp4")
        assert not os.path.exists(out_vid_name)
        fps = self.ann.video.get_avg_fps()
        codec = "h264"
        writer = FFMpegWriter(fps=fps, codec=codec)
        with writer.saving(self.figure, out_vid_name, dpi=300):
            for idx in range(start_frame, end_frame + 1):
                self._current_idx = idx
                self.update()
                writer.grab_frame()


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
        self, fname: str = None, vname: str = None, name: str = None, **kwargs
    ):
        self.fname, vname = self._parse_inp(fname, vname)

        if self.fname is not None:
            self.fstem = Path(fname).stem
        else:
            self.fstem = None

        if name is None:
            if self.fstem is None:
                self.name = "video_annotation"
            else:
                self.name = self.fstem.split("_annotations_")[-1]
        else:
            assert isinstance(name, str)
            self.name = name

        if utils.is_video(vname):
            self.video = utils.Video(vname)
        else:
            self.video = None

        preloaded_json = kwargs.pop("preloaded_json", None)
        if preloaded_json is None:
            self.data = self.load()
        else:
            self.data = preloaded_json

        self.palette = utils.get_palette(
            kwargs.pop("palette_name", "Set2"), n_colors=10
        )  # seaborn Set 2
        self.plot_handles = {
            "ax_list_scatter": kwargs.pop("ax_list_scatter", []),
            "ax_list_trace_x": kwargs.pop("ax_list_trace_x", []),
            "ax_list_trace_y": kwargs.pop("ax_list_trace_y", []),
        }
        self.setup_display()

    @classmethod
    def from_multiple_files(cls, fname_list, vname, name, fname_merged, **kwargs):
        """Merge annotations from multiple files.
        If multiple files contain an annotation label for the same frame, values from the last file will be kept.
        """
        ann_list = [cls(fname, vname, name, **kwargs) for fname in fname_list]
        assert len({ann.video.name for ann in ann_list}) == 1

        labels = sorted(list({label for ann in ann_list for label in ann.labels}))

        ret = cls(fname=fname_merged, vname=ann_list[-1].video.fname, name=name)
        ret.data = {label: {} for label in labels}
        for label in labels:
            ret.data[label] = functools.reduce(
                lambda x, y: {**x, **y}, [ann.data.get(label, {}) for ann in ann_list]
            )
        ret.palette = ann_list[-1].palette

        return ret

    @staticmethod
    def _parse_inp(fname_inp, vname_inp):
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
                    utils.removesuffix(Path(fname_inp), "_annotations")
                    .split("_annotations_")[0]
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
            fname = os.path.join(
                Path(vname_inp).parent, Path(vname_inp).stem + "_annotations.json"
            )
        elif fname_inp is not None and vname_inp is not None:
            assert utils.is_video(vname_inp)
            fname, vname = fname_inp, vname_inp  # do nothing
        return fname, vname

    def load(self, n_annotations=10, **h5_kwargs):
        """Load annotations from a json file, dlc h5 file, or initialize an annotation dictionary if a file doesn't exist."""
        if (self.fname is None) or (not os.path.exists(self.fname)):
            return {str(label): {} for label in range(n_annotations)}

        if Path(self.fname).suffix == ".json":
            return self._load_json(self.fname)

        assert Path(self.fname).suffix == ".h5"
        return self._load_dlc(self.fname, **h5_kwargs)

    @staticmethod
    def _load_json(json_fname):
        with open(json_fname, "r") as f:
            ret = {}
            for k, v in json.load(f).items():
                if v:
                    ret[k] = {int(frame_num): loc for frame_num, loc in v.items()}
            return ret

    def _load_dlc(self, dlc_fname, **kwargs):
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
        df, remove_label_prefix="point", img_prefix="img", img_suffix=".png"
    ):
        """Convert dlc labeled data dataframe to an annotation dictionary"""
        if False in [
            utils.removeprefix(x, remove_label_prefix).isdigit() for x in df.columns.levels[1]
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
    def _dlc_trace_to_annotation_dict(df, remove_label_prefix="point"):
        """Convery dlc labeled trace dataframe (result of analyze_videos) to an annotation dictionary."""
        if False in [
            utils.removeprefix(x, remove_label_prefix).isdigit() for x in df.columns.levels[1]
        ]:
            label_orig_to_internal = {
                x: str(xcnt) for xcnt, x in enumerate(df.columns.levels[1].tolist())
            }
        else:
            label_orig_to_internal = {
                x: utils.removeprefix(x, remove_label_prefix)
                for x in df.columns.levels[1].tolist()
            }
        frames = df.index.values
        print(label_orig_to_internal)

        data = {label: {} for label in label_orig_to_internal.values()}
        scorer = df.columns.levels[0].values[0]
        for label_orig, label_internal in label_orig_to_internal.items():
            for frame in frames:
                coord_val = [
                    df.loc[frame][scorer, label_orig, coord_name]
                    for coord_name in ("x", "y")
                ]
                if np.all(np.isnan(coord_val)):
                    continue
                data[label_internal][frame] = coord_val

        return data

    def __len__(self):
        """Number of annotations"""
        return len(self.data)

    @property
    def n_frames(self):
        """Number of frames in the video being annotated"""
        if self.video is None:
            return max(self.frames, default=-1) + 1
        return len(self.video)

    @property
    def n_annotations(self):
        """Number of points being annotated in the video."""
        return len(self)

    @property
    def labels(self):
        """Labels of the annotations."""
        return list(self.data.keys())

    @property
    def frames(self):
        """Frame numbers in the video that have annotations."""
        ret = list(
            set([frame for label in self.labels for frame in self.get_frames(label)])
        )
        ret.sort()
        return ret

    @property
    def frames_overlapping(self):
        """List of frames in the video where all the labels are annotated."""
        ret = list(
            functools.reduce(
                set.intersection, [set(self.get_frames(label)) for label in self.labels]
            )
        )
        ret.sort()
        return ret

    def get_frames(self, label):
        """Return a list of frames that are annotated with the current label."""
        assert label in self.labels
        return list(self.data[label].keys())

    def save(self, fname=None):
        """Save the annotations json file. self.fname should be a valid file path."""
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

    def sort_labels(self):
        """Sort labels in the data dictionary."""
        self.data = dict(sorted(self.data.items()))

    def sort_data(self):
        """Sort annotations by the frame numbers."""
        self.data = {
            label: dict(sorted(self.data[label].items())) for label in self.labels
        }

    def get_values_cv(self, frame_num: int):
        """Return annotations at frame_num in a format for openCV's optical flow algorithms"""
        return np.array(self.get_at_frame(frame_num), dtype=np.float32).reshape(
            (self.n_annotations, 1, 2)
        )

    def _n_digits_in_frame_num(self):
        """Number of digits to use when constructing a string from the frame number."""
        if self.n_frames is None:
            return "6"
        return str(len(str(self.n_frames)))

    def _frame_num_as_str(self, frame_num: int):
        """Return the frame umber as a formatted string."""
        return f"{frame_num:0{self._n_digits_in_frame_num()}}"

    def add_at_frame(self, frame_num: int, values: np.ndarray):
        """Add annotations at a frame, given the annotation values."""
        assert isinstance(frame_num, int)
        values = np.array(values)
        assert values.shape == (self.n_annotations, 2)
        for label, value in zip(self.labels, values):
            self.data[label][frame_num] = list(value)

    def get_at_frame(self, frame_num: int):
        """Retrieve annotations at a given frame number. If an annotation is not present, nan values will be used."""
        ret = []
        for label in self.labels:
            if frame_num in self.data[label]:
                ret.append(self.data[label][frame_num])
            else:
                ret.append([np.nan, np.nan])
        return ret

    def __getitem__(self, key):
        """Easy access to specific annotation, or data from a frame number."""
        if key in self.labels:
            return self.data[key]
        if key in self.frames:
            return self.get_at_frame(key)
        raise ValueError(f"{key} is neither an annotation nor a frame with annotation.")

    def to_dlc(
        self,
        scorer: str = "praneeth",
        output_path: str = None,
        file_prefix: str = None,
        img_prefix: str = "img",
        img_suffix: str = ".png",
        label_prefix: str = "point",
        save: bool = True,
        internal_to_dlc_labels: dict = None,
    ) -> pd.DataFrame:
        """Save annotations in deeplabcut format."""
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
                    df.loc[
                        "labeled-data",
                        self.video.name,
                        f"{img_prefix}{frame:0{index_length}}{img_suffix}",
                    ][scorer, annotation_label_dlc, coord_name] = coord_val
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
        """
        assert label in self.labels
        ret = np.full([self.n_frames, 2], np.nan)
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

    def add_label(self, label=None, color=None):
        if label is None:  # pick the next available label
            assert len(self.labels) < 10
            label = f"{len(self.labels)}"
        assert label not in self.labels
        assert label in [
            str(x) for x in range(10)
        ]  # for now, might remove this limitation in the future

        if color is None:
            color = self.palette[int(label)]
        else:
            assert len(color == 3)
            assert all([0 <= x <= 1 for x in color])
            self.palette[len(self.labels)] = tuple(color)

        self.data[label] = {}
        self.sort_labels()

        print(f"Created new label {label}")

    def add(self, location, label, frame_number):
        """Add a point annotation (location) of a given label at a frame number."""
        assert len(location) == 2
        self.data[label][frame_number] = list(location)

    def remove(self, label, frame_number):
        assert label in self.labels
        self.data[label].pop(frame_number, None)

    # display management
    def _process_ax_list(self, ax_list, type_):
        assert type_ in ("scatter", "trace_x", "trace_y")
        if ax_list is None:
            ax_list = self.plot_handles[f"ax_list_{type_}"]
        if isinstance(ax_list, plt.Axes):
            ax_list = [ax_list]
        assert all([isinstance(ax, plt.Axes) for ax in ax_list])
        self.plot_handles[f"ax_list_{type_}"] = ax_list
        return ax_list

    def _process_palette(self, palette):
        if palette is None:
            return self.palette

        if isinstance(palette, str):
            palette = utils.get_palette(palette, 10)

        self.palette = palette
        return palette

    def setup_display_scatter(self, ax_list_scatter=None, palette=None):
        ax_list_scatter = self._process_ax_list(ax_list_scatter, "scatter")
        palette = self._process_palette(palette)

        dummy_xy = [np.nan] * len(
            palette
        )  # instead of len(self.labels) to keep all 10 points, some of them can be nan
        for ax_cnt, ax in enumerate(ax_list_scatter):
            self.plot_handles[f"labels_in_ax{ax_cnt}"] = ax.scatter(
                dummy_xy, dummy_xy, color=palette, picker=5
            )

    def setup_display_trace(
        self, ax_list_trace_x=None, ax_list_trace_y=None, palette=None
    ):
        ax_list_trace_x = self._process_ax_list(ax_list_trace_x, "trace_x")
        ax_list_trace_y = self._process_ax_list(ax_list_trace_y, "trace_y")
        palette = self._process_palette(palette)

        if self.n_frames > 0 and len(self.frames) / self.n_frames > 0.8:
            plot_type = "-"
        else:
            plot_type = "o"

        x = np.arange(self.n_frames)
        dummy_y = np.full(self.n_frames, np.nan)
        for ax_cnt, (ax_x, ax_y) in enumerate(zip(ax_list_trace_x, ax_list_trace_y)):
            for label_cnt, x_color in enumerate(
                self.palette
            ):  # create plots for all 10 traces
                label = str(label_cnt)
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

            ax_x.set_xlim(0, self.n_frames)

    def setup_display(self, ax_list_scatter=None, ax_list_trace=None, palette=None):
        self.setup_display_scatter(ax_list_scatter, palette)
        self.setup_display_trace(ax_list_trace, palette)

    def update_display_scatter(self, frame_number, draw=False):
        for ax_cnt in range(len(self.plot_handles["ax_list_scatter"])):
            n_pts = len(self.palette)
            scatter_offsets = np.full((n_pts, 2), np.nan)
            scatter_offsets[
                [int(label) for label in self.labels], :
            ] = self.get_at_frame(frame_number)
            self.plot_handles[f"labels_in_ax{ax_cnt}"].set_offsets(scatter_offsets)
        if draw:
            plt.draw()

    def update_display_trace(self, label=None, draw=False):
        if label is None:
            label_list = self.labels
        else:
            assert label in self.labels
            label_list = [label]

        for ax_cnt in range(len(self.plot_handles["ax_list_trace_x"])):
            for label in label_list:
                trace = self.to_trace(label)
                for coord_cnt, coord in enumerate(("x", "y")):
                    handle_name = f"trace_in_ax{coord}{ax_cnt}_label{label}"
                    self.plot_handles[handle_name].set_ydata(trace[:, coord_cnt])

        if draw:
            plt.draw()

    def update_display(self, frame_number, label=None, draw=False):
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

    def _set_visibility(self, visibility: bool = True, draw=False):
        for plot_handle in self._trace_or_label_handles.values():
            plot_handle.set_visible(visibility)
        if draw:
            plt.draw()

    def hide(self, draw=True):
        """Hide all elements (scatter, traces) in this annotation."""
        self._set_visibility(False, draw)

    def show(self, draw=True):
        """Show all elements (scatter, traces) in this annotation."""
        self._set_visibility(True, draw)

    def _set_trace_visibility(self, label: str, visibility: bool = True, draw=False):
        for plot_handle_name, plot_handle in self._trace_handles.items():
            if plot_handle_name.endswith(f"_label{label}"):
                plot_handle.set_visible(visibility)
        if draw:
            plt.draw()

    def show_trace(self, label, draw=True):
        self._set_trace_visibility(label, True, draw)

    def hide_trace(self, label, draw=True):
        self._set_trace_visibility(label, False, draw)

    def show_one_trace(self, label, draw=True):
        for this_label in self.labels:
            self._set_trace_visibility(this_label, this_label == label, draw)

    def set_alpha(self, alpha=0.4, label=None, draw=True):
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

    def set_plot_type(self, type_="line", draw=True):
        assert type_ in ("line", "dot")
        for trace_handle in self._trace_handles.values():
            if type_ == "line":
                trace_handle.set_linestyle("-")
                trace_handle.set_marker("None")
            else:
                trace_handle.set_linestyle("None")
                trace_handle.set_marker("o")
        if draw:
            plt.draw()

    def clip_labels(self, start_frame: int, end_frame: int):
        """Remove annotations outside the clip range. Clip range includes start and end frame."""
        for label in self.labels:
            self.data[label] = {
                k: v
                for k, v in self.data[label].items()
                if start_frame <= k <= end_frame
            }

    def keep_overlapping_continuous_frames(self):
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
        for label in self.labels:
            self.data[label] = {
                k: v for k, v in self.data[label].items() if k in frames_to_keep
            }

    def get_area(
        self, labels: list | str, lowpass: float = None
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


class VideoAnnotations(AssetContainer):
    def add(self, name: Union[str, VideoAnnotation], fname=None, vname=None, **kwargs):
        """Create-and-add"""
        if isinstance(name, VideoAnnotation):
            ann = name
        else:
            assert isinstance(name, str)
            ann = VideoAnnotation(fname, vname, name, **kwargs)
        return super().add(ann)
