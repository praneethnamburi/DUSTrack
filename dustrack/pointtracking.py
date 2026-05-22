"""
Point-tracking UI, annotation containers, and DeepLabCut HDF5 interop.

Moved here from ``datanavigator.pointtracking`` in datanavigator 1.5.0a1 /
dustrack 1.2.0a1 -- see CHANGELOG. The browsers / asset managers / event
plumbing still live in datanavigator; this module sits on top of them.
The full pre-relocation history is preserved -- ``git log --follow
dustrack/pointtracking.py`` traces it back through datanavigator's
rc1-rc2 perf work, the label-aware y-refit, the _TrackedFrameDict
mutation guard, etc.

The DLC-specific paths in :py:class:`VideoAnnotation` are tagged
``DUSTrack-shaped`` in their docstrings (kept as a greppable marker for
the DLC-aware code paths, even though the whole module now lives in DUSTrack).
"""
from __future__ import annotations

import functools
import json
import os
import weakref
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
from dustrack.lk_opticalflow import lucas_kanade, lucas_kanade_rstc


class _DUSTrackBase(VideoBrowser):
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
        n_labels: int = 1,
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

        # Label-aware y-refit (1.4.0rc2): wire after bootstrap set_state
        # above so the init-time label_range pin doesn't trigger a
        # spurious first fit. Both annotation_label and label_range
        # participate in :py:attr:`_current_label`, so both feed the
        # hook; :meth:`_on_active_label_change` de-dupes via
        # ``_last_active_label`` so the second callback fired by an
        # increment_label_range / decrement_label_range / digit-key
        # path is a no-op when the derived label hasn't actually
        # changed. Initial value: the current label at registration
        # time, so a user-driven set_state to the same label is a no-op.
        self._last_active_label = self._current_label
        self.statevariables["annotation_label"].add_on_change(
            self._on_active_label_change
        )
        self.statevariables["label_range"].add_on_change(
            self._on_active_label_change
        )
        # rc2: single show() call regardless of fast_render. Inside,
        # StateVariables.show() tries the Qt-native dock widget first
        # (mounts under the buttons column for both tiers) and falls
        # back to TextView on non-Qt backends. The pre-rc2
        # _ax_statevar gridspec slot is gone.
        self._ax_statevar = None
        self.statevariables.show(pos="bottom left")

        self.add_events()

        self.set_key_bindings()

        self._add_default_buttons()

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

        if self.__class__.__name__ == "_DUSTrackBase":
            plt.show(block=False)
            self.update()
            plt.setp(self._ax_trace_x.get_xticklabels(), visible=False)
            plt.draw()

    @classmethod
    def from_annotations(
        cls, annotations: list[VideoAnnotation], *args, **kwargs
    ) -> _DUSTrackBase:
        if isinstance(annotations, VideoAnnotation):
            annotations = [annotations]
        video_names = {a.video.fname for a in annotations}
        assert len(video_names) == 1  # same video across all annotations
        return cls(video_names.pop(), annotations, *args, **kwargs)

    def add_annotation_layers(
        self,
        annotation_names: list[str] | dict[str, Path] | list[VideoAnnotation],
        n_labels: int = 1
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
                video=self.data,
                ax_list_scatter=ax_list_scatter,
                ax_list_trace_x=[self._ax_trace_x],
                ax_list_trace_y=[self._ax_trace_y],
                palette_name="Set2",
                n_labels=n_labels,
            )

        # same set of labels in all the loaded annotations -- union of
        # every layer's declared labels, including empty-but-declared
        # ones. Pre-1.4.0rc2 this filtered to labels-with-data only,
        # because `VideoAnnotation.save` pruned empties on the way out
        # (the historical "10 default labels, drop what the user didn't
        # use" contract). 1.4.0rc2 makes labels first-class schema --
        # save no longer prunes, so an empty label on disk is a real
        # declaration that should round-trip.
        all_labels = sorted(
            {label for ann in self.annotations._list for label in ann.labels}
        )
        if not all_labels:
            # when starting without any annotations, initialize a full set of empty annotations
            all_labels = [str(x) for x in range(n_labels)]
        for ann in self.annotations._list:
            for label in all_labels:
                if label not in ann.labels:
                    ann.add_label(label)
            ann.sort_labels()
            ann.re_setup_display()

        self._refresh_annotation_state_lists()

    def _refresh_annotation_state_lists(self) -> None:
        """Resync the ``annotation_layer`` / ``annotation_overlay``
        statevariables' rotations to match the current
        ``self.annotations.names``.

        Single source of truth for the dropdown-states refresh shared by
        :meth:`add_annotation_layers` (extending the rotation) and
        :meth:`remove_annotation_layer` (shrinking it). On shrink, also
        clamps each statevariable's ``_current_state_idx`` so the
        position is never out-of-bounds after the underlying
        ``states`` list shortens; the caller is responsible for picking
        a sensible new selection *before* calling this method, the
        clamp here is the last-resort safety net.
        """
        if "annotation_layer" in self.statevariables.names:
            sv = self.statevariables["annotation_layer"]
            sv.states = self.annotations.names
            if sv._current_state_idx >= len(sv.states):
                sv._current_state_idx = max(0, len(sv.states) - 1)
        if "annotation_overlay" in self.statevariables.names:
            sv = self.statevariables["annotation_overlay"]
            sv.states = [None] + self.annotations.names
            if sv._current_state_idx >= len(sv.states):
                sv._current_state_idx = 0  # None

    def remove_annotation_layer(self, name: str) -> None:
        """Remove an annotation layer from the active session.

        Tears down the layer's plot artists (scatter + per-label trace
        lines on x/y trace axes) via :meth:`VideoAnnotation.clear_display`,
        drops the layer from :attr:`annotations` via
        :meth:`AssetContainer.remove`, then resyncs the
        ``annotation_layer`` / ``annotation_overlay`` statevariables
        through :meth:`_refresh_annotation_state_lists` so the dropdown
        rotations + current selections stay valid.

        Pre-flight:
        - ``name`` must be an existing layer name.
        - Refuses if it would leave the container empty (callers needing
          a "reset to single empty layer" semantic should use
          :meth:`VideoAnnotation.reload` on the surviving layer instead).

        Active-layer handoff: if ``name`` is currently the primary
        layer (``annotation_layer.current_state``), the previous layer
        in the rotation becomes the new primary (or the first one if
        the removed layer was at index 0). If ``name`` is currently the
        overlay, the overlay clears to ``None``.

        Note: ``"buffer"`` is *not* excluded here -- the dnav layer
        treats every named layer the same. Consumers (DUSTrack)
        enforce buffer-exclusion at the UI layer.
        """
        assert name in self.annotations.names, (
            f"layer {name!r} not in {self.annotations.names!r}"
        )
        if len(self.annotations) <= 1:
            raise ValueError(
                f"refusing to remove the only remaining annotation layer "
                f"{name!r}; use VideoAnnotation.reload() to reset its "
                f"contents instead."
            )

        # Pick the new primary / overlay selections *before* mutating
        # the container so we can name them by string rather than by
        # post-removal index.
        layer_sv = self.statevariables["annotation_layer"]
        overlay_sv = self.statevariables["annotation_overlay"]
        current_primary = layer_sv.current_state
        current_overlay = overlay_sv.current_state

        names = list(self.annotations.names)
        removed_idx = names.index(name)
        survivors = [n for n in names if n != name]
        if current_primary == name:
            # Prefer the previous-in-rotation layer; if the removed
            # layer was at index 0, fall through to survivors[0].
            new_primary = survivors[max(removed_idx - 1, 0)]
        else:
            new_primary = current_primary

        # Tear down the artists this layer owns. clear_display() walks
        # ax_list_scatter + ax_list_trace_x and calls .remove() on each
        # handle the layer registered in setup_display.
        ann = self.annotations[name]
        ann.clear_display()
        self.annotations.remove(name)

        layer_sv.states = survivors
        layer_sv.set_state(new_primary)
        if current_overlay == name:
            overlay_sv.states = [None] + survivors
            overlay_sv.set_state(0)  # None
        else:
            # Just resync the rotation; current selection stays.
            overlay_sv.states = [None] + survivors
            if current_overlay is not None:
                overlay_sv.set_state(current_overlay)

    def set_key_bindings(self) -> None:
        """Set the keyboard actions.

        Groups mirror the 5-step workflow in ``docs/source/resources/
        keyboard_shortcuts.png``: layer selection → label selection →
        frame navigation → edit → refine. ``self._section_order`` pins
        the cheatsheet's section order to the workflow order (section 3
        first, then 1, 2, 4, 5a, 5b, 5c) regardless of when each binding
        was originally registered. Bindings not on the PNG (save,
        refresh, reset view, pan, keep-overlapping, toggle-num-keys
        mode) fall through to the "Other" section.
        """
        sec1 = "1. Select annotation layer"
        sec2 = "2. Select annotation number (#)"
        sec3 = "3. Navigate to the desired video frame"
        sec4 = "4. Edit annotation"
        sec5a = "5a. LK-RSTC based label augmentation"
        sec5b = "5b. Refine labels in a selected interval"
        sec5c = "5c. Copy annotations between layers"
        self._section_order = (sec1, sec2, sec3, sec4, sec5a, sec5b, sec5c)

        # 1. Select annotation layer
        self.add_key_binding("=", self.next_annotation_layer,
            "Next annotation layer (primary)", group=sec1)
        self.add_key_binding("-", self.previous_annotation_layer,
            "Previous annotation layer (primary)", group=sec1)
        self.add_key_binding("]", self.next_annotation_overlay,
            "Next annotation layer (overlay)", group=sec1)
        self.add_key_binding("[", self.previous_annotation_overlay,
            "Previous annotation layer (overlay)", group=sec1)

        # 2. Select annotation number (#)
        self.add_key_binding("'", self.next_annotation_label,
            "Next annotation label (#)", group=sec2)
        self.add_key_binding(";", self.previous_annotation_label,
            "Previous annotation label (#)", group=sec2)
        self.add_key_binding("w", self.increment_label_range,
            "Next annotation # range", group=sec2)
        self.add_key_binding("q", self.decrement_label_range,
            "Previous annotation # range", group=sec2)

        # 3. Navigate to the desired video frame
        self.add_key_binding("g", self.increment,
            "Next video frame", group=sec3)
        self.add_key_binding("f", self.increment_if_unannotated,
            "Next video frame if unannotated", group=sec3)
        self.add_key_binding("d", self.decrement_if_unannotated,
            "Previous video frame if unannotated", group=sec3)
        self.add_key_binding(",", self.previous_frame_with_any_label,
            "Previous frame with any annotation", group=sec3)
        self.add_key_binding(".", self.next_frame_with_any_label,
            "Next frame with any annotation", group=sec3)
        self.add_key_binding("alt+,", self.previous_frame_of_interest,
            "Previous frame of interest", group=sec3)
        self.add_key_binding("alt+.", self.next_frame_of_interest,
            "Next frame of interest", group=sec3)
        self.add_key_binding("n", self.next_frame_with_current_label,
            "Next frame with current annotation", group=sec3)
        self.add_key_binding("p", self.previous_frame_with_current_label,
            "Previous frame with current annotation", group=sec3)
        self.add_key_binding("b", self.previous_frame_with_current_label,
            "Previous frame with current annotation (alias of p)", group=sec3)

        # 4. Edit annotation
        self.add_key_binding("t", self.add_annotation,
            "Add annotation (hover on image)", group=sec4)
        self.add_key_binding("y", self.remove_annotation,
            "Remove annotation (hover near it on image)", group=sec4)

        # 5a. LK-RSTC based label augmentation
        # Sequence: v, alt+v, ctrl+alt+v, alt+b, ctrl+b
        self.add_key_binding(
            "v",
            (lambda s: s.check_labels_with_lk(mode="minimal")).__get__(self),
            "Check labels with LK - minimal mode",
            group=sec5a,
        )
        self.add_key_binding(
            "alt+v",
            (lambda s: s.check_labels_with_lk(mode="current")).__get__(self),
            "Check labels with LK - current label",
            group=sec5a,
        )
        self.add_key_binding(
            "ctrl+alt+v",
            (lambda s: s.check_labels_with_lk(mode="all")).__get__(self),
            "Check labels with LK - all labels",
            group=sec5a,
        )
        self.add_key_binding(
            "alt+b",
            (lambda s: s.predict_labels_with_lucas_kanade(labels="current")).__get__(
                self
            ),
            "Predict current label at current frame with LK (primary layer)",
            group=sec5a,
        )
        self.add_key_binding(
            "ctrl+b",
            (lambda s: s.predict_labels_with_lucas_kanade(labels="all")).__get__(self),
            "Predict all labels at current frame with LK (primary layer)",
            group=sec5a,
        )

        # 5b. Refine labels in a selected interval
        # Sequence: z, a, ctrl+a, alt+a, ctrl+alt+a, ctrl+d
        #
        # ``z`` is registered by ``add_events()`` (which runs in __init__
        # before set_key_bindings) as the add-event hotkey for the
        # interp_with_lk interval picker -- with an auto-generated
        # description and no group. Re-register here with the proper
        # group + description; remove_key_binding first so it lands at
        # the end of the dict (rather than its earlier insertion slot),
        # ensuring it leads section 5b in dict-iteration order.
        z_binding = self._keypressdict.get("z")
        if z_binding is not None:
            self.remove_key_binding("z")
            self.add_key_binding(
                "z", z_binding.callback,
                "Select interval (press once at start, once at end)",
                group=sec5b,
            )
        self.add_key_binding(
            "a", self.interpolate_with_lk,
            "Interpolate current label with LK-RSTC (primary layer)",
            group=sec5b,
        )
        self.add_key_binding(
            "ctrl+a",
            (lambda s: s.interpolate_with_lk(all_labels=True)).__get__(self),
            "Interpolate all labels with LK-RSTC (primary layer)",
            group=sec5b,
        )
        self.add_key_binding(
            "alt+a", self.remove_labels_in_interval,
            "Clear current label in selected interval (primary layer)",
            group=sec5b,
        )
        self.add_key_binding(
            "ctrl+alt+a",
            (lambda s: s.remove_labels_in_interval(all_labels=True)).__get__(self),
            "Clear all labels in selected interval (primary layer)",
            group=sec5b,
        )
        self.add_key_binding(
            "ctrl+d",
            (lambda s: s.interpolate_with_lk_norstc(all_labels=True)).__get__(self),
            "Interpolate all labels with LK (no RSTC, primary layer)",
            group=sec5b,
        )

        # 5c. Copy annotations between layers
        # Sequence: m, alt+c, c, ctrl+alt+c
        self.add_key_binding("m", self.toggle_frame_of_interest,
            "Toggle (mark / unmark) current frame as a frame of interest",
            group=sec5c)
        self.add_key_binding("alt+c", self.copy_frames_of_interest_from_overlay,
            "Copy annotations at frames of interest from overlay",
            group=sec5c)
        self.add_key_binding("c", self.copy_current_annotation_from_overlay,
            "Copy current annotation at current frame from overlay",
            group=sec5c)
        self.add_key_binding("ctrl+alt+c", self.copy_frames_in_interval_from_overlay,
            "Copy annotations in selected interval from overlay",
            group=sec5c)

        # Bindings not depicted on the docs PNG -- fall through to "Other".
        self.add_key_binding("s", self.save, "Save current annotation layer")
        self.add_key_binding("`", self.cycle_number_keys_behavior,
            "Toggle num-keys mode (select / place)")
        self.add_key_binding("alt+q", self.keep_overlapping_continuous_frames,
            "Keep only consecutive frames where every label is annotated")
        self.add_key_binding(
            "f5", self.refresh, "Refresh UI from current annotation data",
        )

        # Pan keys: ``/`` and ``l`` were registered by
        # set_default_keybindings at very early dict positions; remove +
        # re-add here alongside the new ``j`` / ``k`` so all four pan
        # keys sit as one contiguous block in the cheatsheet's Other
        # section.
        self.remove_key_binding("/")
        self.remove_key_binding("l")
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
        self.add_key_binding(
            "/",
            (lambda s: s.pan(direction="right")).__get__(self),
            description="pan right (alias of k)",
        )
        self.add_key_binding(
            "l",
            (lambda s: s.pan(direction="up")).__get__(self),
            description="pan up",
        )

        self.remove_key_binding("e")  # remove the "Extract clip" feature from VideoBrowser

        if self._fast_render:
            # Overwrite the inherited 'r' binding (GenericBrowser.reset_axes)
            # so a single keystroke resets the pane under the cursor (image
            # OR traces, not both), with a fall-through to the pre-rc2
            # "reset everything" behaviour when the cursor is undetectable
            # — preserving the catch-all muscle memory. Tier 1 keeps the
            # inherited binding (no image zoom to reset).
            self.add_key_binding(
                "r", self._reset_view_all,
                "Reset view under cursor (traces: full-video x, active label y)",
            )
            self.add_key_binding(
                "alt+r", self._reset_view_to_data_extent,
                "Reset view under cursor (traces: data-extent x, all-labels y)",
            )

    def _reset_view_all(self, event: Any | None = None) -> None:
        """Cursor-aware ``r`` dispatch (Tier 2 only); traces use full-video x.

        Three branches, keyed on ``event.inaxes`` after
        :meth:`_patch_event_for_image_pane` has patched the event in
        :meth:`__call__`:

        - Cursor over the Tier 2 image pane (``inaxes == self._ax_image``,
          which mirrors ``self._image_pane``) → image-pane zoom/pan reset
          only; trace axes untouched.
        - Cursor over a trace axis (``inaxes in (self._ax_trace_x,
          self._ax_trace_y)``) → trace x set to ``(0, ann.n_frames)`` and
          y fit to the **active label** (active + overlay layers, if both
          carry it; see :meth:`_fit_y_to_active_label`). Image pane
          untouched. Setting x to the full video range (rather than
          autoscaling to the annotation data extent) keeps frames
          *outside* the current annotation envelope visible, which is the
          usual case when extending annotations to a new region. See
          ``_reset_view_to_data_extent`` for the autoscale-x +
          union-y sibling, bound to ``alt+r``.
        - Cursor anywhere else / event undetectable (``event is None`` or
          ``event.inaxes is None`` or some unrelated mpl axis) → reset
          everything: image pane reset AND the same trace treatment
          (x = full-video, y = active-label fit). Preserves muscle memory
          for users who hit ``r`` while hovering a button or off-figure.
        """
        inaxes = getattr(event, "inaxes", None) if event is not None else None
        if inaxes is self._ax_image:
            self._image_pane.reset_view()
            return
        if inaxes is self._ax_trace_x or inaxes is self._ax_trace_y:
            self._reset_traces_to_full_video(event=event)
            self.update()
            return
        self._image_pane.reset_view()
        self._reset_traces_to_full_video(event=event)
        self.update()

    def _reset_view_to_data_extent(self, event: Any | None = None) -> None:
        """Cursor-aware ``alt+r`` dispatch (Tier 2 only); union autoscale.

        Same dispatch structure as :meth:`_reset_view_all`, but the trace
        branch + fallback autoscale both x and y to the **union** of all
        visible data (active layer + overlay's full label set), not just
        the active label. Use when you want to confirm no label has gone
        catastrophically offscreen, or when you genuinely want to see
        several labels at the same scale. Also useful for shrinking the
        trace view to the annotated region when the full video is much
        longer than the annotated window.
        """
        inaxes = getattr(event, "inaxes", None) if event is not None else None
        if inaxes is self._ax_image:
            self._image_pane.reset_view()
            return
        if inaxes is self._ax_trace_x or inaxes is self._ax_trace_y:
            self.reset_axes(
                axis="both",
                event=event,
                axes=[self._ax_trace_x, self._ax_trace_y],
            )
            self.update()
            return
        self._image_pane.reset_view()
        self.reset_axes(axis="both", event=event)
        self.update()

    def _reset_traces_to_full_video(self, event: Any | None = None) -> None:
        """Trace pair: x = ``(0, ann.n_frames)``, y fit to active label.

        Helper for the ``r`` dispatch trace + fallback branches. The y
        branch fits only the active label (across the active layer +
        overlay layer, if both carry it) rather than autoscaling to the
        union of every label's data extent. Multi-label workflow:
        pressing ``r`` over a trace gives a comfortable view of what the
        user is actively editing instead of compressing it into the
        envelope of every other label. Single-label sessions: collapses
        to "fit to all data" -- the helper walks exactly one label.

        For the union-autoscale behaviour, ``alt+r`` is the sibling
        (:meth:`_reset_view_to_data_extent`).
        """
        self._ax_trace_x.set_xlim(0, self.ann.n_frames)
        self._fit_y_to_active_label(event=event)

    def _fit_y_to_active_label(self, event: Any | None = None) -> None:
        """One-shot y-refit on both trace axes, scoped to the active label.

        Computes y-extent from the active layer's active-label trace
        (and the overlay layer's same-named label, if an overlay is set
        and contains the label) with a 5% margin, then ``set_ylim`` on
        both trace axes. No-op if neither layer has data for the active
        label (helper short-circuits before touching ylim).

        Mirrors the "first cache miss with real data fits y" bootstrap
        in :meth:`update_frame_marker`: a deliberate one-shot fit that
        leaves the Manual y-policy guard satisfied (``set_ylim`` flips
        ``autoscaley_on=False`` as a side effect, so subsequent
        mutations / label switches / FOI toggles don't disturb the
        view). Pair with the label-switch hook
        (:meth:`_on_active_label_change`) and the ``r`` / ``alt+r``
        dispatch.
        """
        label = self._current_label
        layers = [self.ann]
        overlay_name = self._current_overlay
        if overlay_name is not None and overlay_name != self._current_layer:
            overlay = self.annotations[overlay_name]
            if label in overlay.labels:
                layers.append(overlay)

        y_x_vals: list[np.ndarray] = []
        y_y_vals: list[np.ndarray] = []
        for ann in layers:
            if label not in ann.labels:
                continue
            trace = ann.to_trace(label)  # (n_frames, 2)
            col_x, col_y = trace[:, 0], trace[:, 1]
            col_x = col_x[~np.isnan(col_x)]
            col_y = col_y[~np.isnan(col_y)]
            if col_x.size:
                y_x_vals.append(col_x)
            if col_y.size:
                y_y_vals.append(col_y)

        def _apply(ax: plt.Axes, parts: list[np.ndarray]) -> None:
            if not parts:
                return
            cat = np.concatenate(parts)
            lo, hi = float(cat.min()), float(cat.max())
            if not np.isfinite(lo) or not np.isfinite(hi):
                return
            if hi == lo:
                pad = max(abs(lo) * 0.05, 0.5)
            else:
                pad = (hi - lo) * 0.05
            ax.set_ylim(lo - pad, hi + pad)

        _apply(self._ax_trace_x, y_x_vals)
        _apply(self._ax_trace_y, y_y_vals)

    def _on_active_label_change(self) -> None:
        """StateVariable on_change hook: refit y only on real label change.

        Wired (in :meth:`__init__`) to both ``annotation_label`` and
        ``label_range`` statevariables. Each callback compares the
        derived ``_current_label`` to the previously-seen value cached
        in ``_last_active_label``; if it changed, dispatches the
        one-shot y-refit via :meth:`_fit_y_to_active_label`. No-op when
        the label hasn't actually changed (e.g. ``increment_label_range``
        fires set_state on ``annotation_label`` after ``cycle()`` on
        ``label_range``; the second callback sees the same
        ``_current_label`` and short-circuits).

        Switching primary layer or overlay does NOT route through here
        -- the layer-flip comparison workflow keeps its current y window.
        """
        new_label = self._current_label
        if new_label == self._last_active_label:
            return
        self._last_active_label = new_label
        self._fit_y_to_active_label()

    def _add_default_buttons(self) -> None:
        """Install the default action buttons appended after ``__init__``.

        Subclasses with a hand-curated button order (e.g. DUSTrack's
        rc2 sidebar) override this to suppress the default placement
        and add the same buttons at the desired position. Currently
        installs only ``Refresh UI``; new defaults belong here.
        """
        self.buttons.add(text="Refresh UI", action_func=self.refresh)

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
            ax_list=[self._ax_trace_x, self._ax_trace_y],
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
            # Manual y-policy: refit ylim only while autoscale is still
            # claimed (mpl-default + no prior set_ylim). After the first
            # cache miss with real data, set_ylim flips autoscaley_on off
            # and subsequent mutations / label switches / FOI toggles
            # leave the user's view alone. Pressing `r` re-enables
            # autoscale via reset_axes, restoring a one-shot refit.
            #
            # `have_*_data` guards against the all-NaN case: nanlim()
            # falls back to the current ylim then, and calling
            # set_ylim(current_ylim) would still flip autoscale off as a
            # side effect, locking the view at the mpl default before
            # any real data lands. Only consume the autoscale claim when
            # there is actually data to fit.
            have_x_data = not np.all(np.isnan(trace_data_x))
            have_y_data = not np.all(np.isnan(trace_data_y))
            if have_x_data and self._ax_trace_x.get_autoscaley_on():
                self._ax_trace_x.set_ylim(xl)
            if have_y_data and self._ax_trace_y.get_autoscaley_on():
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

    def keep_overlapping_frames(self) -> None:
        self.ann.keep_overlapping_frames()
        self.update()

    def predict_labels_with_lucas_kanade(
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
        self._ensure_target_has_labels(self.ann, label_list)
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
        self._ensure_target_has_labels(self.ann, label_list)
        for frame_count, frame_number in enumerate(range(start_frame, end_frame + 1)):
            for label_count, label in enumerate(label_list):
                location = list(rstc_path[frame_count, label_count, :])
                self._add_annotation(location, frame_number, label)
        self.update()

    @staticmethod
    def _ensure_target_has_labels(target_ann, labels) -> None:
        """Declare each label on ``target_ann`` if not already present.

        Pre-1.4.0rc2 every annotation layer carried 10 empty default
        labels, so cross-layer copy paths (the LK family below, plus
        :meth:`check_labels_with_lk`) could blindly ``add(label, ...)``
        against any label in 0-9. The 1.4.0rc2 first-class-label
        schema dropped that bootstrap to a single ``"0"``; cross-layer
        copies now declare missing labels on the target explicitly.
        """
        for label in labels:
            if label not in target_ann.labels:
                target_ann.add_label(label)

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

        # 1.4.0rc2: with labels first-class and n_labels=1 by default,
        # the target layer (typically "buffer") may not yet hold every
        # label the source carries. Pre-rc2 the 10-default-empty
        # bootstrap made this implicit. Declare here so target_ann.add
        # below doesn't KeyError on a missing label slot.
        self._ensure_target_has_labels(target_ann, label_list)

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


# ---------------------------------------------------------------------
# Re-exports for back-compat
# ---------------------------------------------------------------------
# VideoAnnotation, VideoAnnotations and _TrackedFrameDict moved to
# dustrack.annotations in 1.2.0rc1. Re-imported here so existing
# ``from dustrack.pointtracking import VideoAnnotation`` paths keep
# resolving (the pickle-compat sys.modules alias in __init__.py also
# routes ``dustrack.pointtracking`` to the new home for pickles that
# may have the old path baked in).
from .annotations import (  # noqa: E402,F401
    VideoAnnotation,
    VideoAnnotations,
    _TrackedFrameDict,
)
