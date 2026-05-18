"""
Point-tracking UI, annotation containers, and DeepLabCut HDF5 interop.

The DLC-specific paths in :py:class:`VideoAnnotation` are tagged
``DUSTrack-shaped`` in their docstrings and are slated to migrate to
``dustrack.VideoAnnotation`` in 1.3.0 alongside the ``pointtracking.py``
split.
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Callable, Mapping, Any
from tqdm import tqdm

import numpy as np
import pandas as pd
import pysampled
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter

from datanavigator import utils
from datanavigator.assets import AssetContainer
from datanavigator.videos import VideoBrowser
from datanavigator.opticalflow import lucas_kanade, lucas_kanade_rstc


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
        annotation_names (list[str] | Mapping[str, Path], optional):
            list[str] - Name(s) of the annotation layer(s). The file path(s) are deduced from the name(s).
            Mapping[str, Path] - A dictionary mapping of annotation layer names to the annotation file paths.
            Defaults to '' with one layer of annotations.
        titlefunc (Callable, optional): A function used to set the title of the image axis.
            Defaults to a title function specified in :py:class:`VideoBrowser`.
    """

    def __init__(
        self,
        vid_name: Path,
        annotation_names: list[str] | Mapping[str, Path] | list[VideoAnnotation] = "",
        n_labels: int = 10,
        titlefunc: Callable = None,
        image_process_func: Callable = lambda im: im,
        height_ratios: tuple = (10, 1, 1),  # depends on your screen size
        fast_render: bool = False,
    ) -> None:
        if fast_render:
            # Tier 2: image lives in a Qt widget above the canvas, so
            # the mpl figure only needs the trace rows. A smaller
            # figure (matched to the trace region) keeps the canvas
            # widget small so its per-frame raster + upload cost stays
            # near probe 13's ~5 ms prediction instead of inheriting
            # the full 12x8 figure's raster cost. Skip
            # constrained_layout (probe-14 finding) -- the layout
            # solver re-runs on every canvas.draw() and adds ~30 ms
            # for trace-only figures, dwarfing the actual raster cost.
            figure_handle = plt.figure(constrained_layout=False, figsize=(12, 3))
            gs = figure_handle.add_gridspec(2, 1, hspace=0.05)
            figure_handle.subplots_adjust(
                left=0.06, right=0.99, top=0.97, bottom=0.12, hspace=0.05,
            )
            self._ax_image = None  # set after super().__init__ to the Qt pane
            self._ax_trace_x = figure_handle.add_subplot(gs[0, 0])
            self._ax_trace_y = figure_handle.add_subplot(gs[1, 0])
        else:
            figure_handle = plt.figure(constrained_layout=True, figsize=(12, 8))
            # rc2 (Commit 2): state-variables moved out of the figure
            # into the QDockWidget left column, so the gridspec drops
            # its dedicated left column. Image now spans full width.
            # mpl fallback (non-Qt backend, e.g. Agg in tests) gets the
            # state-variables text overlay floating on the figure.
            gs = figure_handle.add_gridspec(
                3, 1, height_ratios=list(height_ratios)
            )
            self._ax_image = figure_handle.add_subplot(gs[0, 0])
            self._ax_trace_x = figure_handle.add_subplot(gs[1, 0])
            self._ax_trace_y = figure_handle.add_subplot(gs[2, 0])
        self._ax_trace_x.sharex(self._ax_trace_y)
        (self._frame_marker_x,) = self._ax_trace_x.plot(
            [], [], color="black", linewidth=1
        )
        (self._frame_marker_y,) = self._ax_trace_y.plot(
            [], [], color="black", linewidth=1
        )
        # Tier 1 passes the image axis; Tier 2 passes the figure so
        # VideoBrowser can build the Qt image pane on it.
        ax_or_fig = figure_handle if fast_render else self._ax_image
        super().__init__(
            vid_name, titlefunc, ax_or_fig, image_process_func,
            fast_render=fast_render,
        )
        if fast_render:
            # super() has built and stashed self._image_pane; mirror it
            # as self._ax_image so existing identity checks
            # (event.inaxes == self._ax_image) still fire.
            self._ax_image = self._image_pane
        self.memoryslots.hide()
        self.memoryslots.disable()

        # annotation layers
        self.annotations = VideoAnnotations(parent=self)
        self.add_annotation_layers(annotation_names, n_labels)
        if "buffer" in self.annotations.names:
            self.annotations["buffer"].plot_type = "line"

        # frames of interest
        self.frames_of_interest = []
        (self._plot_frames_of_interest_x,) = self._ax_trace_x.plot(
            [], [], color="gray", linewidth=1, alpha=0.5
        )
        (self._plot_frames_of_interest_y,) = self._ax_trace_y.plot(
            [], [], color="gray", linewidth=1, alpha=0.5
        )

        # State variables. rc2: each variable advertises a control
        # surface via the `widget=` kwarg, read by the Qt sidebar to
        # render QComboBox / QButtonGroup / QLabel. Hint is ignored on
        # non-Qt backends (Agg falls back to TextView).
        self.statevariables.add(
            "annotation_layer", self.annotations.names, widget="dropdown",
        )
        self.statevariables.add(
            "annotation_overlay", [None] + self.annotations.names,
            widget="dropdown",
        )
        self.statevariables.add(
            "annotation_label", self.ann.labels, widget="dropdown",
        )
        self.statevariables.add(
            "label_range",
            [f"{x*10}-{x*10+9}" for x in range(100)],  # up to 1000 labels
            widget="dropdown",
        )
        first_label = self.ann.labels[0]
        self.statevariables["label_range"].set_state(int(first_label) // 10)
        self.update_annotation_label_states()
        self.statevariables.add("number_keys", ["select", "place"], widget="toggle")
        # rc2: single show() call regardless of fast_render. Inside,
        # StateVariables.show() tries the Qt-native dock widget first
        # (mounts under the buttons column for both tiers) and falls
        # back to TextView on non-Qt backends. The pre-rc2
        # _ax_statevar gridspec slot is gone.
        self._ax_statevar = None
        self.statevariables.show(pos="bottom left")

        self.add_events()

        self.set_key_bindings()

        self.buttons.add(text="Refresh UI", action_func=self.refresh)

        # set mouse click behavior
        if self._fast_render:
            # Image pane is Qt-native: pick + place_label events on
            # the image come through the _QtPickAdapter, not mpl's
            # canvas signal chain. go_to_frame stays on mpl because
            # the trace axes still live in matplotlib.
            adapter = self._image_pane.install_pick_adapter()
            adapter.connect_pick(self.select_label_with_mouse)
            adapter.connect_button_press(self.place_label_with_mouse)
            self.cid.append(
                self.figure.canvas.mpl_connect("button_press_event", self.go_to_frame)
            )
        else:
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
            plt.draw()

    @classmethod
    def from_annotations(
        cls, annotations: list[VideoAnnotation], *args, **kwargs
    ) -> VideoPointAnnotator:
        if isinstance(annotations, VideoAnnotation):
            annotations = [annotations]
        video_names = {a.video.fname for a in annotations}
        assert len(video_names) == 1  # same video across all annotations
        return cls(video_names.pop(), annotations, *args, **kwargs)

    def add_annotation_layers(
        self, 
        annotation_names: list[str] | dict[str, Path] | list[VideoAnnotation],
        n_labels: int = 10
    ) -> None:
        """Load data from annotation files if they exist, otherwise initialize annotation layers."""
        if isinstance(annotation_names, (str, VideoAnnotation)):
            annotation_names = [annotation_names]

        if isinstance(annotation_names, list) and all(
            [isinstance(a, VideoAnnotation) for a in annotation_names]
        ):
            annotation_names = {
                a.name: a.fname for a in annotation_names
            }  # re-add because of plotting

        if "buffer" not in annotation_names and "buffer" not in self.annotations.names:
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
            # Tier 1: scatter target is the mpl image axis. Tier 2:
            # scatter target is a fresh Qt marker group on the image
            # pane, wrapped by _QtScatterArtist downstream. The
            # VideoAnnotation factory branches on the type of the
            # element in ax_list_scatter.
            if self._fast_render:
                ax_list_scatter = [self._image_pane.add_marker_group()]
            else:
                ax_list_scatter = [self._ax_image]
            self.annotations.add(
                name=name,
                fname=fname,
                vname=self.fname,
                ax_list_scatter=ax_list_scatter,
                ax_list_trace_x=[self._ax_trace_x],
                ax_list_trace_y=[self._ax_trace_y],
                palette_name="Set2",
                n_labels=n_labels,
            )

        # same set of non-empty labels in all the loaded annotations
        all_labels = []
        for ann in self.annotations._list:
            all_labels += [
                label for label, label_data in ann.data.items() if label_data
            ]
        all_labels = sorted(list(set(all_labels)))
        if not all_labels:
            # when starting without any annotations, initialize a full set of empty annotations
            all_labels = [str(x) for x in range(n_labels)]
        for ann in self.annotations._list:
            for label in all_labels:
                if label not in ann.labels:
                    ann.add_label(label)
            for label in ann.labels:
                if label not in all_labels:
                    assert not ann.data[label]
                    del ann.data[label]
            ann.sort_labels()
            ann.re_setup_display()

        if "annotation_layer" in self.statevariables.names:
            self.statevariables["annotation_layer"].states = self.annotations.names
        if "annotation_overlay" in self.statevariables.names:
            self.statevariables["annotation_overlay"].states = [
                None
            ] + self.annotations.names
    

    def set_key_bindings(self) -> None:
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
        self.add_key_binding("alt+c", self.copy_frames_of_interest_from_overlay)
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

        self.add_key_binding("w", self.increment_label_range)
        self.add_key_binding("q", self.decrement_label_range)

        self.remove_key_binding("e")  # remove the "Extract clip" feature from VideoBrowser

        self.add_key_binding(
            "f5", self.refresh, "Refresh UI from current annotation data"
        )

        if self._fast_render:
            # Overwrite the inherited 'r' binding (GenericBrowser.reset_axes)
            # so a single keystroke resets BOTH the trace axes AND the
            # Tier-2 image-pane zoom/pan, matching the pre-1.5.0
            # matplotlib UX where 'r' was the catch-all "give me my view
            # back" key. Tier 1 keeps the inherited binding (no image
            # zoom to reset).
            self.add_key_binding(
                "r", self._reset_view_all, "Reset image zoom + trace axes"
            )

    def _reset_view_all(self, event: Any | None = None) -> None:
        """Reset both the Qt image pane (zoom/pan -> fit) and the
        matplotlib trace axes (`reset_axes("both")`)."""
        self._image_pane.reset_view()
        self.reset_axes(axis="both", event=event)

    def refresh(self, event: Any | None = None) -> None:
        """Force-refresh the UI from the current annotation ``.data``.

        Drops the trace-display cache on every annotation layer and the
        frame-marker cache, then calls ``update()`` so the next draw
        re-reads every value from the backing dicts.

        Recovery path for the rare case where ``.data`` was mutated
        directly (e.g. from an IPython prompt:
        ``v.ann.data["0"][42] = [x, y]``) bypassing the public
        ``add()`` / ``remove()`` / ``add_at_frame()`` API and therefore
        skipping the ``_revision`` bump that
        :meth:`VideoAnnotation.update_display_trace` and
        :meth:`update_frame_marker` cache on. In normal in-code
        mutation paths the public API bumps ``_revision`` and the
        caches invalidate automatically; calling ``refresh()`` is
        always safe but rarely necessary.
        """
        for ann in self.annotations:
            ann.invalidate_caches()
        self._frame_marker_cache = None
        self.update()

    def add_events(self) -> None:
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
        """Return current label."""
        return str(int(self.statevariables["annotation_label"].current_state) + int(self.statevariables["label_range"]._current_state_idx) * 10)

    @property
    def _current_layer(self) -> str:
        """Return current annotation layer"""
        return self.statevariables["annotation_layer"].current_state

    @property
    def _current_overlay(self) -> str | None:
        """Return current annotation overlay layer"""
        return self.statevariables["annotation_overlay"].current_state

    def _get_fname_annotations(
        self, annotation_name: str, suffix: str = ".json"
    ) -> str:
        """Construct the filename corresponding to an  annotation layer named annotation_name."""
        return os.path.join(
            Path(self.fname).parent,
            Path(self.fname).stem
            + "_annotations"
            + f'{"_" if annotation_name else ""}'
            + annotation_name
            + suffix,
        )

    def set_image_background_color(self, color: Any) -> None:
        """Set the background color of the image region.

        Tier 1 routes to ``self._ax_image.set_facecolor`` (matplotlib);
        Tier 2 routes to the Qt image pane's
        ``set_background_color``. The single-method surface lets
        consumers (DUSTrack's ``_apply_dark_theme``) issue one call
        regardless of which tier is active.
        """
        if self._fast_render:
            self._image_pane.set_background_color(color)
        else:
            self._ax_image.set_facecolor(color)

    def _patch_event_for_image_pane(self, event: Any) -> None:
        """When a key press fires with cursor over the Qt image pane,
        patch the event so ``event.inaxes == self._ax_image`` and
        ``xdata`` / ``ydata`` reflect the scene-space cursor position.

        Without this, Tier 2 silently drops the "press T over the
        video to add a label here" workflow because mpl reports
        ``inaxes=None`` whenever the cursor is outside its canvas
        widget -- which is *always* the case when hovering the Qt
        image pane.
        """
        if not getattr(self, "_fast_render", False):
            return
        if event.name != "key_press_event":
            return
        if getattr(event, "inaxes", None) is not None:
            return
        pane = self._image_pane
        if pane is None or not pane.underMouse():
            return
        try:
            from qtpy.QtGui import QCursor
        except ImportError:
            return
        view = pane._view
        viewport = view.viewport()
        local = viewport.mapFromGlobal(QCursor.pos())
        scene_pos = view.mapToScene(local)
        event.inaxes = pane
        event.xdata = float(scene_pos.x())
        event.ydata = float(scene_pos.y())

    def __call__(self, event: Any) -> None:
        """Callbacks for number keys."""
        self._patch_event_for_image_pane(event)
        super().__call__(event)
        # if a number key is pressed
        if event.name == "key_press_event" and str(event.key).isdigit() and int(event.key) in range(10):
            key_str = str(event.key)
            key_int = int(event.key)
            label = str(key_int + int(self.statevariables["label_range"]._current_state_idx) * 10)
            
            # if the label already exists
            if label in self.ann.labels:
                self.statevariables["annotation_label"].set_state(key_str)
                if self.statevariables["number_keys"].current_state == "place":
                    self.add_annotation(event)
            else:                
                # add a new label if the label doesn't exist
                for ann in self.annotations._list:  # add new label to all annotations
                    if label not in ann.labels:
                        ann.add_label(label)
                self.update_annotation_label_states()
                self.statevariables["annotation_label"].set_state(key_str)
                if self.statevariables["number_keys"].current_state == "place":
                    self.add_annotation(event)

            self.update()

    def update(self) -> None:
        """Update elements in the UI."""
        self.update_annotation_visibility(draw=False)
        self.statevariables.update_display(draw=False)
        self.update_frame_marker(draw=False)
        super().update()
        plt.draw()

    def update_annotation_visibility(self, draw: bool = False) -> None:
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

    def update_frame_marker(self, draw: bool = False) -> None:
        """Update the current frame location in the trace plots.

        The trace ylim / FOI tick positions are a pure function of the
        active label + per-annotation contents + frames_of_interest, none
        of which change when only ``_current_idx`` moves. Cache them
        keyed on a cheap tuple of (label, annotation revisions,
        frames_of_interest) so per-frame navigation only does the
        frame-marker set_data calls.
        """

        def nanlim(x, default):
            if np.all(np.isnan(x)):
                return default
            return [np.nanmin(x) * 0.9, np.nanmax(x) * 1.1]

        def nanlim_small(x, default, scale=0.6):
            if np.all(np.isnan(x)):
                nmin, nmax = default
            else:
                nmin = np.nanmin(x) * 0.9
                nmax = np.nanmax(x) * 1.1
            m = (nmin + nmax) / 2
            return [(nmin - m) * scale + m, (nmax - m) * scale + m]

        cache_key = (
            self._current_label,
            tuple(ann._revision for ann in self.annotations._list),
            tuple(self.frames_of_interest),
        )
        cached = getattr(self, "_frame_marker_cache", None)
        if cached is None or cached[0] != cache_key:
            trace_data_x, trace_data_y = np.hstack(
                [ann.to_trace(self._current_label).T for ann in self.annotations._list]
            )

            default_x, default_y = self._ax_trace_x.get_ylim(), self._ax_trace_y.get_ylim()
            xl, yl = nanlim(trace_data_x, default_x), nanlim(trace_data_y, default_y)
            xls, yls = nanlim_small(trace_data_x, default_x), nanlim_small(trace_data_y, default_y)

            self._plot_frames_of_interest_x.set_data(
                *utils.ticks_from_times(self.frames_of_interest, xl)
            )
            self._plot_frames_of_interest_y.set_data(
                *utils.ticks_from_times(self.frames_of_interest, yl)
            )
            if not np.any(np.isnan(xl)):
                self._ax_trace_x.set_ylim(xl)
                self._ax_trace_y.set_ylim(yl)

            self._frame_marker_cache = (cache_key, xls, yls)
        else:
            _, xls, yls = cached

        self._frame_marker_x.set_data([self._current_idx] * 2, xls)
        self._frame_marker_y.set_data([self._current_idx] * 2, yls)

        if draw:
            plt.draw()

    def copy_annotations_from_overlay(self) -> None:
        """Copy annotations from the overlay layer into the current layer in the current frame."""
        ann_overlay = self.annotations[self._current_overlay]
        frame_number = self._current_idx
        for label in self.ann.labels:
            if label in ann_overlay.labels:
                location = ann_overlay.data[label].get(frame_number, None)
                if location is not None:
                    self.ann.add(location, label, frame_number)
        self.update()

    def copy_current_annotation_from_overlay(self) -> None:
        """Copy annotations from the overlay layer into the current layer."""
        ann_overlay = self.annotations[self._current_overlay]
        frame_number = self._current_idx
        label = self._current_label
        if label in ann_overlay.labels:
            location = ann_overlay.data[label].get(frame_number, None)
            if location is not None:
                self.ann.add(location, label, frame_number)
        self.update()

    def copy_frames_of_interest_from_overlay(self) -> None:
        """copy annotations at frames of interest from buffer into the current layer.
        If there is no buffer, then copy from the overlay layer.
        """
        source_ann = self.annotations[self._current_overlay]

        for frame_number in self.frames_of_interest:
            for label in self.ann.labels:
                if label in source_ann.labels:
                    location = source_ann.data[label].get(frame_number, None)
                    if location is not None:
                        self.ann.add(location, label, frame_number)
        self.update()

    def copy_frames_in_interval_from_overlay(self) -> None:
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

    def _add_annotation(
        self,
        location: list[float],
        frame_number: int | None = None,
        label: str | None = None,
    ) -> None:
        """Core function for adding annotations. Allows more control."""
        if frame_number is None:
            frame_number = self._current_idx
        if label is None:
            label = self._current_label
        self.ann.add(location, label, frame_number)

    def add_annotation(self, event: Any) -> None:
        """Add annotation at frame. If it exists, it'll get overwritten."""
        if event.inaxes == self._ax_image:
            self._add_annotation([float(event.xdata), float(event.ydata)])
        self.update()

    def remove_annotation(self, event: Any | None = None) -> None:
        """remove annotation at the current frame if it exists"""
        self.ann.remove(self._current_label, self._current_idx)
        self.update()

    def _get_nearest_annotated_frame(self, label: str | None = None) -> int:
        """Return the nearest frame (in either direction) number with the current label in the current annotation layer."""
        if label is None:
            label = self._current_label
        d = {abs(x - self._current_idx): x for x in self.ann.get_frames(label)}
        if not d:
            raise ValueError(
                f"No frames with label {label} in annotation layer {self._current_layer}."
            )
        return d[min(d)]

    def previous_frame_with_current_label(self, event: Any | None = None) -> None:
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

    def next_frame_with_current_label(self, event: Any | None = None) -> None:
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

    def previous_frame_with_any_label(self, event: Any | None = None) -> None:
        """Go to the previous frame with any label in the current annotation layer."""
        self._current_idx = max(
            [x for x in self.ann.frames if x < self._current_idx],
            default=self._current_idx,
        )
        self.update()

    def next_frame_with_any_label(self, event: Any | None = None) -> None:
        """Go to the next frame with any label in the current annotation layer."""
        self._current_idx = min(
            [x for x in self.ann.frames if x > self._current_idx],
            default=self._current_idx,
        )
        self.update()

    def previous_frame_of_interest(self, event: Any | None = None) -> None:
        """Go to the previous frame of interest."""
        self._current_idx = max(
            [x for x in self.frames_of_interest if x < self._current_idx],
            default=self._current_idx,
        )
        self.update()

    def next_frame_of_interest(self, event: Any | None = None) -> None:
        self._current_idx = min(
            [x for x in self.frames_of_interest if x > self._current_idx],
            default=self._current_idx,
        )
        self.update()

    def previous_annotation_layer(self) -> None:
        """Go to the previous annotation layer."""
        self.statevariables["annotation_layer"].cycle_back()
        self.update()

    def next_annotation_layer(self) -> None:
        """Go to the next annotation layer"""
        self.statevariables["annotation_layer"].cycle()
        self.update()

    def previous_annotation_overlay(self) -> None:
        """Go to the previous annotation overlay layer."""
        self.statevariables["annotation_overlay"].cycle_back()
        self.update()

    def next_annotation_overlay(self) -> None:
        """Go to the next annotation overlay layer."""
        self.statevariables["annotation_overlay"].cycle()
        self.update()

    def previous_annotation_label(self) -> None:
        """Set current annotation label to the previous one."""
        self.statevariables["annotation_label"].cycle_back()
        self.update()

    def next_annotation_label(self) -> None:
        """Set current annotation label to the next one."""
        self.statevariables["annotation_label"].cycle()
        self.update()
    
    def update_annotation_label_states(self) -> None:
        """Update the states of the annotation label state variable."""
        # find labels in the current range
        label_states = [str(int(x) % 10) for x in self.ann.labels if int(x) in range(int(self.statevariables["label_range"]._current_state_idx) * 10, (int(self.statevariables["label_range"]._current_state_idx) + 1) * 10)]
        self.statevariables["annotation_label"].states = label_states

    def cycle_number_keys_behavior(self) -> None:
        """Number keys can be used to either select labels, or place a specific label.
        Toggle between these two behaviors.
        """
        self.statevariables["number_keys"].cycle()
        self.update()

    def increment_label_range(self) -> None:
        """Increment the label range by 10."""
        self.statevariables["label_range"].cycle()
        # if the current label is not present, add it to the list of labels in all the layers!
        current_label = self._current_label
        if self._current_label not in self.ann.labels:
            for ann in self.annotations._list:
                ann.add_label(self._current_label)
        self.update_annotation_label_states()
        self.statevariables["annotation_label"].set_state(str(int(current_label) % 10))
        self.update()
    
    def decrement_label_range(self) -> None:
        """Increment the label range by 10."""
        self.statevariables["label_range"].cycle_back()
        # if the current label is not present, add it to the list of labels
        current_label = self._current_label
        if self._current_label not in self.ann.labels:
            for ann in self.annotations._list:
                ann.add_label(self._current_label)
        self.update_annotation_label_states()
        self.statevariables["annotation_label"].set_state(str(int(current_label) % 10))
        self.update()

    def increment_if_unannotated(self, event: Any | None = None) -> None:
        """Advance the frame if the current frame doesn't have any annotations.
        Useful to pause at annotated frames when adding a new label.
        """
        if self._current_idx not in self.ann.frames:
            self.increment()

    def decrement_if_unannotated(self, event: Any | None = None) -> None:
        """Go to the previous frame if the current frame doesn't have any annotations.
        Useful to pause at annotated frames when adding a new label.
        """
        if self._current_idx not in self.ann.frames:
            self.decrement()

    def save(self) -> None:
        """Save current annotation layer json file."""
        self.ann.save()

    def select_label_with_mouse(self, event: Any) -> None:
        """Select a label by clicking on it with the left mousebutton."""
        if event.mouseevent.button.name == "LEFT":
            picked_label = self.ann.labels[int(event.ind[0])]
            self.statevariables["label_range"]._current_state_idx = int(picked_label) // 10
            self.update_annotation_label_states()
            self.statevariables["annotation_label"].set_state(str(int(picked_label) % 10))
            print(
                f'Picked label {picked_label} at frame {self._current_idx}'
            )
            self.update()

    def place_label_with_mouse(self, event: Any) -> None:
        """Place the selected label with the right mousebutton."""
        if event.inaxes == self._ax and event.button.name == "RIGHT":
            self.add_annotation(event)

    def go_to_frame(self, event: Any) -> None:
        if (
            event.inaxes in (self._ax_trace_x, self._ax_trace_y)
            and event.button.name == "RIGHT"
        ):
            self._current_idx = round(event.xdata)
            self.update()

    def toggle_frame_of_interest(self, event: Any) -> None:
        """Mark/unmark the current frame as a frame of interest"""
        if event.inaxes in (self._ax_trace_x, self._ax_trace_y):
            frame_number = self._current_idx
            if frame_number in self.frames_of_interest:
                self.frames_of_interest.remove(frame_number)
            else:
                self.frames_of_interest.append(frame_number)
            self.frames_of_interest.sort()
            self.update()

    def keep_overlapping_continuous_frames(self) -> None:
        self.ann.keep_overlapping_continuous_frames()
        self.update()

    def predict_points_with_lucas_kanade(
        self,
        labels: str | list[str] = "all",
        start_frame: int | None = None,
        mode: str = "full",
    ) -> Any:
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

        assert (
            len(set([self._get_nearest_annotated_frame(label) for label in labels]))
            == 1
        ), "There must be a unique annotated frame across all labels."
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

    def get_selected_interval(self) -> tuple[int, int]:
        start_frame, end_frame = (
            self.events["interp_with_lk"]
            ._data[(self._current_layer, self._current_label)]
            .get_times()[-1]
        )
        return start_frame, end_frame

    def remove_labels_in_interval(self, all_labels: bool = False) -> None:
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

    def interpolate_with_lk(self, all_labels: bool = False) -> None:
        """Interpolate with lk-rstc between selected interval.
        Use data in the overlay layer as start and end points.
        Add interpolated points to the current layer.
        """
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

    def interpolate_with_lk_norstc(self, all_labels: bool = False) -> None:
        """Infer the motion of the current label using lucas-kanade algorithm in the selected interval."""
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
        # end_points = [ann_overlay.data[label][end_frame] for label in label_list]
        rstc_path = lucas_kanade(video, start_frame, end_frame, start_points)
        for frame_count, frame_number in enumerate(range(start_frame, end_frame + 1)):
            for label_count, label in enumerate(label_list):
                location = list(rstc_path[frame_count, label_count, :])
                self._add_annotation(location, frame_number, label)
        self.update()

    def check_labels_with_lk(self, mode: str = "minimal") -> None:
        """Interpolate between all labeled frames.
        This only makes sense for sparse-labeled annotations.
        Use this when doing first-time annotations (as opposed to refinement).

        I am testing if refining the start labels using this strategy,
        and augmenting the training data will improve deeplabcut tracking!

        Args:
            mode (str, optional):
                "all"     - LK-interpolation for all labels across all labeled frames
                "current" - LK-interpolation for current label across all labeled frames
                "minimal" - LK-interpolation for current label between labeled frames near the current frame (two previous to two next)
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
                    target_ann.add(
                        list(rstc_path[frame_count, 0, :]), label, frame_num
                    )
                rstc_paths[label][start_frame : end_frame + 1, :] = np.squeeze(
                    rstc_path
                )
        self.update()

    def render(
        self, start_frame: int, end_frame: int, out_vid_name: str | None = None
    ) -> None:
        """Render the video with annotations."""
        if out_vid_name is None:
            out_vid_name = str(Path(self.ann.fname).with_suffix(".mp4"))
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
        n_labels: int = 10,
        **kwargs
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

        if utils.is_video(vname):
            self.video = utils.Video(vname)
        else:
            self.video = None

        preloaded_json = kwargs.pop("preloaded_json", None)
        if preloaded_json is None:
            self.data = self.load(n_annotations=n_labels)
        else:
            self.data = preloaded_json

        self._original_palette = utils.get_palette(
            kwargs.pop("palette_name", "Set2"),
            n_colors=1000
        )  # seaborn Set 2
        self.plot_handles = {
            "ax_list_scatter": kwargs.pop("ax_list_scatter", []),
            "ax_list_trace_x": kwargs.pop("ax_list_trace_x", []),
            "ax_list_trace_y": kwargs.pop("ax_list_trace_y", []),
        }
        self._plot_type = "dot" # line or dot
        # Bumped on every mutation of self.data. Consumers (e.g.
        # VideoPointAnnotator.update_frame_marker) read this to invalidate
        # caches keyed on per-label trace contents. Over-invalidates on
        # ordering-only changes (sort_labels / sort_data); under-invalidates
        # would be a correctness bug.
        self._revision = 0
        self.setup_display()

    @classmethod
    def from_multiple_files(
        cls, fname_list: list[str], vname: str, name: str, fname_merged: str, **kwargs
    ) -> VideoAnnotation:
        """Merge annotations from multiple files.
        If multiple files contain an annotation label for the same frame, values from the last file will be kept.
        """
        ann_list: list[VideoAnnotation] = [cls(fname, vname, name, **kwargs) for fname in fname_list]
        assert len({ann.video.name for ann in ann_list}) == 1

        labels = sorted(list({label for ann in ann_list for label in ann.labels}))

        ret = cls(fname=fname_merged, vname=ann_list[-1].video.fname, name=name)
        ret.data = {label: {} for label in labels}
        for label in labels:
            ret.data[label] = functools.reduce(
                lambda x, y: {**x, **y}, [ann.data.get(label, {}) for ann in ann_list]
            )

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

    def load(self, n_annotations: int = 10, **h5_kwargs) -> dict:
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
        frames = df.index.values

        data = {label: {} for label in label_orig_to_internal.values()}
        scorer = df.columns.levels[0].values[0]
        for label_orig, label_internal in label_orig_to_internal.items():
            coords = df.loc[:, (scorer, label_orig, ["x", "y"])]
            for frame in frames:
                if frame in coords.index:
                    coord_val = coords.loc[frame].values
                    if np.all(np.isnan(coord_val)):
                        continue
                    data[label_internal][frame] = coord_val.tolist()

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
        """Set the type of plot to use for the annotations."""
        assert plot_type in ("line", "dot")
        self._plot_type = plot_type
        self.set_plot_type(self._plot_type)

    def get_frames(self, label: str) -> list[int]:
        """Return a list of frames that are annotated with the current label."""
        assert label in self.labels
        return list(self.data[label].keys())

    def save(self, fname: str | None = None) -> None:
        """Save the annotations json file. self.fname should be a valid file path."""
        if fname is None:
            assert self.fname is not None
            fname = self.fname
        # at the moment, saving is only supported for json files through this method
        if Path(fname).suffix != ".json":
            raise ValueError("Supply a json file name.")
        self.sort_data()
        self.remove_empty_labels()
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

        self.data = {label: self.data[label] for idx, label in enumerate(self.labels) if idx <= last_index}
        self._revision += 1

    def remove_empty_labels(self) -> None:
        """Remove empty labels from the annotation dictionary."""
        self.data = {label: self.data[label] for label in self.labels if len(self.data[label]) > 0}
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

    def __getitem__(
        self, key: str | int
    ) -> dict[int, list[float]] | list[list[float]]:
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
            self._original_palette = self._original_palette*2
        if color is None:
            color = self._original_palette[int(label)]

        # palette size should mirror the number of labels
        assert len(color) == 3
        assert all([0 <= x <= 1 for x in color])

        self.data[label] = {}
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
    def _process_ax_list(
        self, ax_list, type_: str
    ) -> list:
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
        self, ax_list_scatter=None,
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
                handle = ax.scatter(
                    dummy_xy, dummy_xy, color=palette, picker=5
                )
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
                    ax, palette, picker_radius=5.0, image_pane=image_pane,
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

        Args:
            type_ (str, optional): "line" or "dot". Defaults to "line".
            draw (bool, optional): Update display if True. Defaults to True.
        """
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

    def clip_labels(self, start_frame: int, end_frame: int) -> None:
        """Remove annotations outside the clip range. Clip range includes start and end frame."""
        for label in self.labels:
            self.data[label] = {
                k: v
                for k, v in self.data[label].items()
                if start_frame <= k <= end_frame
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
        for label in self.labels:
            self.data[label] = {
                k: v for k, v in self.data[label].items() if k in frames_to_keep
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
            assert self.fname is not None, "Please provide a file name to save the video."
            out_file_name = str(Path(self.fname).parent / f"{Path(self.fname).stem}_sf{start_frame}_ef{end_frame}.mp4")
            print(f"Saving video to {out_file_name}")

        dpi = 200
        # video_data = self.video[start_frame:end_frame+1]

        def setup(ann):
            plot_handles = {}
            first_frame = self.video[start_frame].asnumpy()
            ny, nx, _ = first_frame.shape

            figure = plt.figure(frameon=False, figsize=((nx / dpi), (ny / dpi)))
            ax = figure.add_subplot(111)
            dummy_xy = [np.nan]*len(ann.palette)
            plot_handles["im"] = ax.imshow(first_frame)
            plot_handles["scatter"] = ax.scatter(dummy_xy, dummy_xy, color=ann.palette, s=4**2, edgecolors=[0, 0, 0], linewidths=0.3)
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
            plot_handles["im"].set_data(self.video[frame_number].asnumpy())
            plot_handles['scatter'].set_offsets(scatter_offsets)
        
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
            for frame_number in tqdm(range(start_frame, end_frame+1)): # tqdm(range(ann.n_frames)):
                scatter_points = [list(p1[int(frame_number)]), list(p2[int(frame_number)])]
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


if __name__ == "__main__":
    import sys
    sys.path.append(r"C:\dev\datanavigator")
    import datanavigator
    video_fname = datanavigator.get_example_video()
    v = VideoPointAnnotator(video_fname, "pn2", n_labels=2)
    for ann in v.annotations:
        ann.add_label("3")
    v.update()
    plt.show(block=True)
