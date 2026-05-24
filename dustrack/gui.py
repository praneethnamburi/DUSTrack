"""The interactive DUSTrack GUI class.

:class:`DUSTrack` is the user-facing point-tracking widget; it inherits
from :class:`datanavigator.videos.VideoBrowser` directly (the
intermediate ``_DUSTrackBase`` parent collapsed into ``DUSTrack`` in
the 1.2.0rc1 follow-up). It provides:

* Manual point annotation: add / remove / interpolate / copy across
  layers with keyboard shortcuts (formerly the ``_DUSTrackBase`` API)
* Ultrasound image enhancement (CLAHE + gamma + brightness sliders)
* DLC project lifecycle buttons (Create / Train / Reduce-jitter /
  Apply manual corrections / Save annotation as...)
* Workflow-button enable/disable state machine
* Save-on-close + Train pre-flight guards
* Multi-video swap with bg hydration (Roadmap *Next* item 3)
* Bundle nav widget + Alt+Left/Right key bindings
* DLC layer refresh + adopt + name normalization

Extracted from ``dlcinterface.py`` in dustrack 1.2.0rc1. The
companion :class:`DLCProject` (DLC project lifecycle) stays in
``dlcinterface.py`` -- the modules cross-reference each other so
:meth:`DLCProject.annotate` lazy-imports ``DUSTrack`` to avoid an
import cycle.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import datanavigator as dnav
from datanavigator import utils
from datanavigator.videos import VideoBrowser

from .lk_filter import lk_moving_average_filter
from .lk_opticalflow import lucas_kanade, lucas_kanade_rstc
from .annotations import VideoAnnotation, VideoAnnotations
from . import _config
from ._bundle import _BundleState
from ._layer_names import (
    _dlc_bodyparts_to_layer_labels,
    _is_dense_layer_name,
    get_fname_annotations,
    is_manual_annotation_layer,
    is_manual_layer_name,
)
from . import _bundle as _bundle_helpers
from . import _bundle_swap
from . import _close_guard
from . import _nav_widget
from . import _preflight
from . import _preflight_modal
from . import _seed_bundle_modal
from . import _blip_modal
from . import _train_modal
from . import blip as _blip
from . import _view_state
from . import _workflow_gates
from ._qt_styling import _make_group_styler, _pin_qt_palette
from ._image_enhance import (
    _CLAHE_CLIP_MIN,
    _GAMMA_MIN,
    _apply_gamma_only,
    _make_enhance_widget_class,
    enhance_ultrasound_image,
)
from .dlcloader import HAS_DLC
from ._overlays import (
    _CREATE_PROJECT_PHASES,
    _JITTER_PHASES,
    _PROGRESS_PATTERNS,
    _SEED_PROJECT_PHASES,
    _TRAINING_PHASES,
    _make_confirm_overlay_class,
    _make_progress_overlay_class,
    _QueueWriter,
    _Tee,
)
from ._file_management import VideoFileManager, make_annotation_file_name
from .dlcinterface import DLCProject, _find_video_index
from .seed import import_seed_bundle_into_project


class DUSTrack(VideoBrowser):
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
        >>> # Most users should call :func:`dustrack.open` instead of
        >>> # constructing DUSTrack directly -- it dispatches to either
        >>> # this class (bare video) or :meth:`DLCProject.annotate`
        >>> # (resume in a DLC project) based on the path you pass.
        >>> # Direct construction below is for advanced / scripted use.
        >>>
        >>> # Basic usage -- annotation_names defaults to "iteration-0",
        >>> # the canonical seed name for the DLC pipeline (the next
        >>> # DLC training round lands as iteration-1).
        >>> tracker = DUSTrack('video.mp4')
        >>>
        >>> # Explicit layer name -- saved as {video_name}_annotations_pn.json
        >>> tracker = DUSTrack('video.mp4', "pn")
        >>>
        >>> # With multiple annotation layers
        >>> tracker = DUSTrack('video.mp4', {
        ...     'manual': 'manual_labels.json',
        ...     'dlc_iter1': 'dlc_predictions.h5'
        ... })
    """

    # Name of the spliced-corrections layer produced by
    # :meth:`apply_manual_corrections`. Parallels the magic ``"buffer"``
    # layer name :meth:`DLCProject.annotate` creates. The string matches
    # the ``_dlccorr`` substring filter in :func:`_extract_frames` so any
    # file written under this name is automatically excluded from DLC
    # training input -- this is a terminal output, not a refinement
    # source.
    CORRECTIONS_LAYER_NAME = "dlccorr"

    def __init__(
        self,
        vid_name,
        annotation_names="iteration-0",
        *args,
        clahe_clip=1.0,
        clahe_grid=8,
        gamma=1.0,
        brightness=0,
        dark_mode=False,
        n_labels: int = 1,
        titlefunc: Optional[Callable] = None,
        height_ratios: tuple = (10, 1, 1),
        fast_render: bool = True,
        **kwargs,
    ):
        # Store enhancement settings. Defaults now correspond to "no
        # enhancement" so the EnhanceWidget sliders start at their
        # leftmost position and DUSTrack opens with the raw frame;
        # the user dials enhancement in via the sliders. Replaced the
        # pre-2026-05-19 "Toggle enhance" button: with slider-driven
        # bypass at min, a separate toggle is redundant. ``brightness``
        # default dropped from ``+10`` to ``0`` so nudging either
        # slider one tick off zero doesn't visually jump a +10 offset
        # in alongside CLAHE/gamma -- the transition off the bypass
        # is now purely the user-driven slider values.
        self._clahe_clip = clahe_clip
        self._clahe_grid = clahe_grid
        self._gamma = gamma
        self._brightness = brightness
        self._dark_mode = dark_mode
        # Construction-time enhance defaults snapshot. Used by
        # :meth:`_set_enhance_state` to reset the sliders on a
        # first-visit swap to a bundle that has no saved
        # ``enhance_state`` yet, rather than carrying the leaving
        # bundle's slider positions into the arriving bundle.
        self._initial_enhance_state = {
            "clahe_clip": float(clahe_clip),
            "gamma": float(gamma),
            "brightness": float(brightness),
        }

        # Create image processor function with per-slider bypass to
        # smooth the transition at the left end of each slider:
        #
        # - Both at min -> raw image, no cvtColor, no CLAHE.
        # - clip at min, gamma > min -> gamma LUT applied per-channel
        #   on the RGB frame; no CLAHE, no cvtColor roundtrip. At
        #   gamma=1.0+epsilon the LUT is near-identity so the
        #   transition off bypass is visually continuous.
        # - clip > min -> full CLAHE pipeline (which inherently
        #   includes the RGB->gray->RGB roundtrip and the gamma
        #   branch inside enhance_ultrasound_image). The clip-slider
        #   transition off zero is still a step (CLAHE startup is
        #   not a fade-in operation) but only happens once now,
        #   instead of being triggered by either slider.
        def image_processor(im):
            skip_clahe = self._clahe_clip <= _CLAHE_CLIP_MIN
            skip_gamma = self._gamma <= _GAMMA_MIN
            if skip_clahe and skip_gamma:
                out = im
            elif skip_clahe:
                out = _apply_gamma_only(im, self._gamma)
            else:
                out = enhance_ultrasound_image(
                    im,
                    self._clahe_clip,
                    self._clahe_grid,
                    self._gamma,
                    self._brightness,
                )
            # dnav 1.5.0a2 auto-detects monochrome sources and returns
            # (H, W) gray frames. Coerce to 3-channel RGB here so the
            # downstream image pane (matplotlib or Qt-native) sees the
            # same shape regardless of source pix_fmt. The hot perf
            # paths (LK / postprocess / opticalflow) short-circuit on
            # 2D upstream of this, so this coerce only fires once per
            # interactive frame navigation -- negligible.
            if out.ndim == 2:
                out = cv.cvtColor(out, cv.COLOR_GRAY2RGB)
            return out

        # Pin the QApplication palette before any widget is built so
        # appearance is reproducible across Qt bindings + OS themes.
        # See :func:`_pin_qt_palette` for the rationale. DUSTrack
        # defaults to datanavigator 1.5.0+ Tier 2 (Qt-native video
        # pane, ~3x speedup on real videos); pass ``fast_render=False``
        # in headless / mpl-only contexts (tests, scripted renders).
        _pin_qt_palette(dark_mode)

        # === Inlined from the former _DUSTrackBase.__init__ ===========
        # 1.2.0rc1 follow-up: the two-class split (_DUSTrackBase +
        # DUSTrack subclass) collapsed into a single class. The body
        # below is the annotation / trace / state-variable / key-binding
        # bootstrap that used to live in the parent; everything below
        # the inline section is DUSTrack-specific (sidebar, enhance
        # widget, DLC gates, bundles, nav).
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
                left=0.06,
                right=0.99,
                top=0.97,
                bottom=0.12,
                hspace=0.05,
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
            gs = figure_handle.add_gridspec(3, 1, height_ratios=list(height_ratios))
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
            vid_name,
            titlefunc,
            ax_or_fig,
            image_processor,
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
            "annotation_layer",
            self.annotations.names,
            widget="dropdown",
        )
        self.statevariables.add(
            "annotation_overlay",
            [None] + self.annotations.names,
            widget="dropdown",
        )
        self.statevariables.add(
            "annotation_label",
            self.ann.labels,
            widget="dropdown",
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
        self.statevariables["label_range"].add_on_change(self._on_active_label_change)
        # rc2: single show() call regardless of fast_render. Inside,
        # StateVariables.show() tries the Qt-native dock widget first
        # (mounts under the buttons column for both tiers) and falls
        # back to TextView on non-Qt backends. The pre-rc2
        # _ax_statevar gridspec slot is gone.
        self._ax_statevar = None
        self.statevariables.show(pos="bottom left")

        self.add_events()
        self.set_key_bindings()
        # Pre-merge _DUSTrackBase.__init__ called ``self._add_default_buttons()``
        # here which appended ``Refresh UI``. After the 1.2.0rc1 merge,
        # the DUSTrack sidebar block below installs ``Refresh UI`` next
        # to ``Keyboard shortcuts`` as a styled utility pair, so the
        # default-buttons hook is gone.

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
                self.figure.canvas.mpl_connect(
                    "pick_event", self.select_label_with_mouse
                )
            )
            self.cid.append(
                self.figure.canvas.mpl_connect(
                    "button_press_event", self.place_label_with_mouse
                )
            )
            self.cid.append(
                self.figure.canvas.mpl_connect("button_press_event", self.go_to_frame)
            )

        # === End of inlined _DUSTrackBase.__init__ body ===============

        for ann in self.annotations:
            ann.__class__ = VideoAnnotation

        self._dlcproject = None
        self._ax_lims = {
            "state": False,
            "x": [None, None],
            "y_trace_x": [None, None],
            "y_trace_y": [None, None],
        }

        # Apply dark theme if enabled
        if dark_mode:
            self._apply_dark_theme()

        # rc2 sidebar order: workflow | display | niche-op | utilities | swap.
        # Double separators mark major group boundaries; a *pair* of
        # double separators sets off Swap layers as a distinct trailing
        # action. dnav's _QtStatevarsWidget appends its own trailing
        # double separator below the statevars section, so we don't add
        # one after Swap layers.
        #
        # rc2 styling: per-group palette is plumbed through dnav's
        # ``Buttons.register_style`` / ``style_tag=`` API -- one styler
        # registered per palette key, then each ``add()`` / ``add_multi``
        # declares its tag inline. The pre-rc2 ``_btns_*`` collection
        # lists + ``_style_sidebar_buttons`` batch pass are gone; styling
        # auto-applies at add-time inside ``_finalize_button``. The
        # statevars widget palette still wants direct palette
        # manipulation (QSS on a parent flattens child QComboBoxes), so
        # that one piece lives in ``_paint_statevars_widget`` below.
        for _group, _spec in self._SIDEBAR_PALETTE.items():
            self.buttons.register_style(_group, _make_group_styler(_spec))

        # --- Workflow group: end-to-end annotation pipeline -------------
        if HAS_DLC:
            self.buttons.add(
                text="Create DLC Project",
                action_func=self.create_dlc_project,
                style_tag="workflow",
            )
            self.buttons.add(
                text="Train DLC model",
                action_func=self.process_dlc_project,
                style_tag="workflow",
            )
            self.buttons.add(
                text="Apply manual corrections",
                action_func=self.apply_manual_corrections,
                style_tag="workflow",
            )
            self.buttons.add(
                text="Reduce jitter",
                action_func=self.process_with_lk,
                style_tag="workflow",
            )
            self.buttons.add(
                text="Detect blip outliers",
                action_func=self.detect_blips_workflow,
                style_tag="workflow",
            )
        self.buttons.add(
            text="Save annotation as...",
            action_func=self.save_annotation_as,
            style_tag="workflow",
        )
        self.buttons.add_separator(style="double")

        # --- Display / trace controls -----------------------------------
        # Image enhancement is driven by the EnhanceWidget sliders
        # (mounted below statevars by _add_enhance_widget). Sliders at
        # min = bypass; no separate Toggle enhance button.
        self.buttons.add_multi(
            dict(
                text="Trace: line",
                action_func=(lambda s, ev: s.ann.set_plot_type("line")).__get__(self),
                style_tag="display",
            ),
            dict(
                text="Trace: dot",
                action_func=(lambda s, ev: s.ann.set_plot_type("dot")).__get__(self),
                style_tag="display",
            ),
        )
        self.buttons.add_multi(
            dict(
                text="Freeze plot axes",
                action_func=self.freeze_plot_axes,
                style_tag="display",
            ),
            dict(
                text="Unfreeze plot axes",
                action_func=self.unfreeze_plot_axes,
                style_tag="display",
            ),
        )
        self.buttons.add(
            text="Refresh UI", action_func=self.refresh, style_tag="display"
        )
        self.buttons.add(
            text="Keyboard shortcuts",
            action_func=(lambda s, ev: s.show_key_bindings()).__get__(self),
            style_tag="display",
        )
        self.buttons.add_separator(style="double")

        # Niche group -- layer-mutating affordances that compound with
        # the user's edit history. Surfaced as buttons (vs keyboard-only)
        # because their destructive scope warrants the explicit click +
        # confirm-modal cadence.
        self.buttons.add(
            text="Decimate annotations",
            action_func=(lambda s, ev: s.decimate_annotations_in_interval()).__get__(
                self
            ),
            style_tag="niche",
        )
        self.buttons.add(
            text="Discard unsaved annotations",
            action_func=self.discard_unsaved_annotations,
            style_tag="niche",
        )
        self.buttons.add(
            text="Replace existing from overlay",
            action_func=self.copy_existing_annotations_from_overlay,
            style_tag="niche",
        )
        self.buttons.add(
            text="Remove layer",
            action_func=self.remove_current_layer,
            style_tag="niche",
        )
        self.buttons.add_separator(style="double")

        self.buttons.add(
            text="Swap annotation layers",
            action_func=self.swap_active_and_overlay,
            style_tag="swap",
        )

        # Statevars widget palette -- can't ride the Buttons styling
        # path because QSS on a QWidget parent flattens child
        # QComboBoxes (Windows-native dropdown rendering goes away).
        self._paint_statevars_widget()

        # Enhance two-slider widget: Qt-only; mpl fallback gets the
        # constructor defaults baked in via __init__ kwargs. Same
        # convention as the rest of DUSTrack's Qt-specific UI
        # (confirm modals, progress overlay, dock sidebar).
        self._enhance_widget = None
        self._add_enhance_widget()

        # rc2 safeguard: intercept window close so the user is asked
        # before losing in-memory annotation diffs. The Train pre-flight
        # already catches this path; this hook covers every other way
        # the window can close (X button, alt-F4, plt.close()).
        self._install_close_guard()

        self.statevariables._text._pos = dnav.utils._parse_pos("bottom left")

        # Workflow-button gating: re-evaluate enabled state whenever the
        # active or overlay dropdown changes. The single refresh helper
        # is also called once below (initial state) and at every other
        # site that mutates _dlcproject / fname / annotation state.
        # ``add_on_change`` is a dnav 1.4.0rc2+ surface; this class
        # already floors on >=1.4.0, so the attribute is guaranteed.
        for _sv_name in ("annotation_layer", "annotation_overlay"):
            if _sv_name in self.statevariables.names:
                self.statevariables[_sv_name].add_on_change(
                    self._refresh_workflow_button_state
                )
        self._refresh_workflow_button_state()

        # Re-evaluate the gates once the lazy DLC import finishes. The
        # bg loader (kicked off in ``dustrack.open()``) flips state
        # from ``pending`` -> ``done`` / ``missing`` on its own thread,
        # so the callback hops to the Qt thread via a one-shot
        # ``QTimer.singleShot(0, ...)`` post into the main event loop
        # before touching the button widgets. ``QTimer.singleShot`` is
        # safe to call from any thread when given a callable -- it
        # posts a deferred event onto the parent's thread. Polling
        # alternative would mirror the ``_run_with_overlay`` 200 ms
        # tick; one-shot post is cheaper and matches the "fire when
        # ready" shape of the callback API.
        self._install_dlc_load_gate_refresh()

        # Multi-video (1.2.0a3) bundle scaffolding. ``_init_bundles``
        # called by :func:`dustrack.open` populates these post-
        # construction; constructions that bypass ``open()`` (test
        # harnesses, advanced callers) get the safe single-bundle
        # default through the lazy accessor below. The nav widget +
        # key bindings are mounted unconditionally so they survive
        # later bundle population without needing a re-mount pass.
        self._bundles: list[_BundleState] = []
        self._active_index: int = 0
        self._video_queue: list = []  # back-compat: tail of bundle fnames
        # 1.2.0a3 seed-modal flag. Set to True by
        # :func:`_open_seed_session` when the tracker is constructed
        # as the modal host; the close-guard + history writer read
        # this to short-circuit save-prompts / recent-sessions writes
        # for the synthetic seed asset. Normal construction paths
        # leave it False.
        self._is_seed_session: bool = False
        self._nav_widget = None
        self._nav_prev_btn = None
        self._nav_next_btn = None
        self._nav_combo = None
        self._nav_combo_signature = None
        self._add_nav_widget()
        self._add_video_nav_key_bindings()
        self._refresh_nav_buttons()

        # Tools menu (Batch process...). Safe no-op on mpl fallback;
        # the helper bails when ``_find_qt_window()`` returns None.
        self._install_tools_menu()

        if self.__class__.__name__ == "DUSTrack":
            plt.show(block=False)
            self.update()
            plt.setp(self._ax_trace_x.get_xticklabels(), visible=False)
            plt.draw()

    # rc2 sidebar palette (pastel analogous band; cool -> warm -> neutral).
    # Each group gets a coordinated bg/fg/border/hover/pressed triplet so
    # group transitions read at a glance without saturation-clash. Swap
    # layers + the statevars widget share the same pale silver `#e0e4e8`
    # -- visual pair at the bottom of the column. Single dark-slate
    # text color (`#2c3e50`) is the same for every group so the eye
    # isn't asked to retune contrast row-to-row.
    _SIDEBAR_STATEVARS_BG = "#e0e4e8"
    _SIDEBAR_TEXT_COLOR = "#2c3e50"
    _SIDEBAR_PALETTE = {
        "workflow": {  # powder blue -- primary pipeline, coolest end
            "bg": "#cfdef3",
            "fg": "#2c3e50",
            "border": "#a8c0dd",
            "hover": "#bccfea",
            "pressed": "#a8c0dd",
        },
        "display": {  # pale mint -- cool green, analogous step from blue
            "bg": "#d4ebd4",
            "fg": "#2c3e50",
            "border": "#aed4ae",
            "hover": "#c1dfc1",
            "pressed": "#aed4ae",
        },
        "niche": {  # pale apricot -- warm shift, "use sparingly"
            "bg": "#f5d9c0",
            "fg": "#2c3e50",
            "border": "#d9b88a",
            "hover": "#eaca9f",
            "pressed": "#d9b88a",
        },
        "swap": {  # pale silver -- matches statevars
            "bg": "#e0e4e8",
            "fg": "#2c3e50",
            "border": "#c0c5cb",
            "hover": "#d0d4d9",
            "pressed": "#c0c5cb",
        },
    }

    def _add_enhance_widget(self) -> None:
        """Mount the two-slider :class:`EnhanceWidget` below the
        statevars widget in the rc2 left-column dock.

        Inserted into ``_dnav_left_column.outer_layout`` immediately
        after the statevars slot so the visual stack is
        ``buttons | statevars | enhance | stretch``. No-op on the
        mpl fallback path (no Qt main window / left column).
        """
        qt_window = self._find_qt_window()
        if qt_window is None:
            return
        col = getattr(qt_window, "_dnav_left_column", None)
        if col is None:
            return
        EnhanceWidget = _make_enhance_widget_class()
        widget = EnhanceWidget(self, parent=col.host)
        # statevars slot is at col.statevars_slot_index; insert right
        # after it (or right after the buttons widget if statevars
        # never got built).
        insert_at = (
            (col.statevars_slot_index + 1)
            if col.statevars_widget is not None
            else col.statevars_slot_index
        )
        col.outer_layout.insertWidget(insert_at, widget)
        self._enhance_widget = widget

    def _paint_statevars_widget(self) -> None:
        r"""Paint the statevars widget bg/fg to match the rc2 sidebar palette.

        Lives outside the ``Buttons.register_style`` machinery because
        the statevars widget needs *palette* manipulation, not QSS:
        QSS on a ``QWidget`` selector cascades into child
        ``QComboBox``\ es and replaces the native Windows dropdown
        paint with a flat CSS box, which is the opposite of what we
        want. ``QPalette.setColor`` respects native widget styling
        and only the parent bg/fg shifts. No-op when the Qt main
        window / left column can't be located (mpl-fallback path).
        """
        qt_window = self._find_qt_window()
        if qt_window is None:
            return
        col = getattr(qt_window, "_dnav_left_column", None)
        if col is None or col.statevars_widget is None:
            return
        from qtpy.QtGui import QColor

        sv = col.statevars_widget
        pal = sv.palette()
        pal.setColor(sv.backgroundRole(), QColor(self._SIDEBAR_STATEVARS_BG))
        pal.setColor(sv.foregroundRole(), QColor(self._SIDEBAR_TEXT_COLOR))
        sv.setPalette(pal)
        sv.setAutoFillBackground(True)
        # Clear any stylesheet that may have been set on a prior pass
        # so the palette change actually takes effect (QSS wins over
        # palette).
        sv.setStyleSheet("")

    def _install_dlc_load_gate_refresh(self) -> None:
        """Arm a QTimer that re-evaluates gates after the lazy DLC import.
        See :func:`._workflow_gates.install_dlc_load_gate_refresh`.
        """
        _workflow_gates.install_dlc_load_gate_refresh(self)

    def _refresh_workflow_button_state(self) -> None:
        """Apply current gate decisions to the Workflow-group buttons.
        See :func:`._workflow_gates.refresh_workflow_button_state`.
        """
        _workflow_gates.refresh_workflow_button_state(self)

    def _evaluate_workflow_gates(self) -> dict:
        """Compute ``{button_label: (enabled, tooltip)}`` for the gated buttons.
        See :func:`._workflow_gates.evaluate_workflow_gates`.
        """
        return _workflow_gates.evaluate_workflow_gates(self)

    def _apply_dark_theme(self):
        """Apply dark theme to the GUI for better ultrasound visibility."""
        bg_color = "#1a1a1a"
        ax_color = "#2a2a2a"
        text_color = "white"

        # Figure background
        self.figure.patch.set_facecolor(bg_color)

        # Image axis (routes to either mpl set_facecolor (Tier 1) or
        # the Qt image pane background brush (Tier 2 / fast_render)).
        self.set_image_background_color(ax_color)

        # Trace axes
        for ax in [self._ax_trace_x, self._ax_trace_y]:
            ax.set_facecolor(ax_color)
            ax.tick_params(colors=text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            for spine in ax.spines.values():
                spine.set_color(text_color)

    def discard_unsaved_annotations(self, event=None):
        """Drop in-memory edits on the active layer and reload from disk.

        Inverse of ``save``: confirms via :class:`ConfirmOverlay`, then
        calls :meth:`VideoAnnotation.reload` on ``self.ann`` and
        triggers ``self.update()`` so the trace pane + scatter
        re-render. The confirm body branches on whether the layer's
        backing file exists:

        - File exists -> "Reload from disk" semantic.
        - File missing -> "Reset to empty" semantic.

        Refuses on the ``dlccorr`` splice and any layer matching
        :func:`_is_dense_layer_name` -- those are regenerated from
        the active + overlay layers (via ``apply_manual_corrections``
        or ``process_with_lk``), not authored by hand, so "discard
        and reload" has no meaningful semantic. Points the user at
        ``Remove layer`` instead.
        """
        qt_window = self._find_qt_window()
        if qt_window is None:
            # mpl fallback: no overlay; just reload silently. Same
            # convention as the rest of DUSTrack's Qt-specific UI
            # (confirm modals, progress overlay, dock sidebar).
            self.ann.reload()
            self.update()
            return

        ConfirmOverlay = _make_confirm_overlay_class()
        layer_name = self._current_layer

        if _is_dense_layer_name(layer_name):
            ConfirmOverlay(
                qt_window,
                title="Cannot discard derived layer",
                message=(
                    f"Layer {layer_name!r} is regenerated from other "
                    "layers (DLC prediction / manual-correction splice "
                    "/ jitter-reduced output) -- there is no "
                    "hand-authored state to discard.\n\n"
                    "Use Remove layer to drop it from the session "
                    "instead."
                ),
                buttons=[("OK", "neutral")],
                default="OK",
                severity="info",
            ).exec_()
            return

        fname = getattr(self.ann, "fname", None)
        file_exists = bool(fname) and Path(fname).exists()
        if file_exists:
            body = (
                f"Reload layer {layer_name!r} from disk?\n\n"
                f"File: {fname}\n\n"
                "In-memory edits since the last save will be lost."
            )
        else:
            body = (
                f"Reset layer {layer_name!r} to empty?\n\n"
                "No file exists on disk for this layer yet. "
                "All in-memory annotations will be lost."
            )
        result = ConfirmOverlay(
            qt_window,
            title="Discard unsaved annotations",
            message=body,
            buttons=[
                ("Discard", "destructive"),
                ("Cancel", "neutral"),
            ],
            default="Cancel",
            severity="destructive",
        ).exec_()
        if result == "Discard":
            self.ann.reload()
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
        self._ax_lims["state"] = True
        self._ax_lims["x"] = self._ax_trace_x.get_xlim()
        self._ax_lims["y_trace_x"] = self._ax_trace_x.get_ylim()
        self._ax_lims["y_trace_y"] = self._ax_trace_y.get_ylim()
        self.update()

    def unfreeze_plot_axes(self, event=None):
        """
        Restore automatic axis scaling for trajectory plots.

        Args:
            event: Mouse/keyboard event (unused, for button compatibility).
        """
        self._ax_lims["state"] = False
        self._ax_lims["x"] = [None, None]
        self._ax_lims["y_trace_x"] = [None, None]
        self._ax_lims["y_trace_y"] = [None, None]
        self.update()

    def create_dlc_project(
        self,
        event=None,
        name=None,
        path=None,
        experimenter=_config.EXPERIMENTER,
        seed_bundle_path=None,
        link_videos: bool | None = None,
    ) -> DLCProject:
        """
        Create a new DeepLabCut project using current annotations as training labels.

        rc2 (1.1.0rc2): on a Qt backend, project creation runs off the
        GUI thread under a modal overlay (no progress bar -- it's a
        fast op, but the overlay surfaces DLC's stdout and a Done
        button so the user can confirm the project location and any
        warnings before continuing). On non-Qt backends the call runs
        synchronously and returns the new :class:`DLCProject`.

        1.2.0a2: if the active manual layer is empty when the user
        clicks Create DLC Project on the Qt path, a multi-step
        Seed-from-bundle modal opens
        (:meth:`_prompt_seed_bundle`). Picking a valid bundle routes
        through a seeding flow that: (a) creates the project as
        usual, (b) calls :func:`import_seed_bundle_into_project` to
        install the bundle's snapshot as iteration-0's trained model
        (overwriting the project's bodyparts with the bundle's), and
        (c) runs ``analyze_videos(iteration_num=0)`` to produce a
        dense reference layer the user can refine into iteration-1.
        ``seed_bundle_path`` may also be passed programmatically to
        bypass the modal.

        Args:
            event: Mouse/keyboard event (unused, for button compatibility).
            name (str, optional): Project name. Defaults to "{video_name}_{annotation_layer}".
            path (str, optional): Directory for project. Defaults to video's parent directory.
            experimenter (str, optional): Experimenter name. Defaults to config value.
            seed_bundle_path (str, optional): Path to a seed bundle
                folder (as produced by
                :func:`extract_snapshot_for_seeding`). When supplied,
                the seeding flow runs unconditionally (Qt or non-Qt),
                bypassing the empty-layer-triggered modal.
            link_videos: Forwarded to :class:`DLCProject` to control
                how the source video is placed inside ``<project>/videos/``.
                ``None`` (default) hard-links on same-volume sources
                and falls back to copy on cross-volume. ``False`` forces
                a deep copy (the pre-1.3.0a2 behavior).

        Returns:
            DLCProject: The newly created project instance on the sync
            path. ``None`` on the Qt async path -- read
            ``self._dlcproject`` after the Done button is clicked.

        Note:
            Project names must contain an underscore for proper DLC configuration handling.
        """
        if not HAS_DLC:
            raise ImportError("deeplabcut is not installed. Cannot create DLC project.")

        qt_window = self._find_qt_window()
        active_layer_empty = not any(self.ann.data.values())

        # Qt path with empty active layer + no explicit bundle: open
        # the seeding modal sequence. The user can still cancel out
        # at every step (intent -> folder pick -> confirm). Cancel
        # leaves the UI intact (returns None).
        if active_layer_empty and seed_bundle_path is None and qt_window is not None:
            seed_bundle_path = self._prompt_seed_bundle(qt_window)
            if seed_bundle_path is None:
                return None  # user cancelled at some step

        # An empty active manual layer still needs an on-disk JSON
        # for DLCProject's constructor: ``copy_annotations`` reads
        # ``<video_stem>_annotations_<suffix>.json`` and the
        # constructor's all-same-labels assert iterates the resulting
        # files. The empty save writes ``{}`` (or ``{label: {}}``);
        # both pass through fine.
        self.ann.save()
        if name is None:
            name = f"{self.name}_{self.ann.name}"
        if path is None:
            path = str(Path(self.fname).parent)

        def _build_project():
            return DLCProject(
                path=path,
                videos=[self.fname],
                name=name,
                experimenter=experimenter,
                annotation_suffix=self.ann.name,
                link_videos=link_videos,
            )

        # Seeding path: run project creation + bundle import +
        # iteration-0 inference inside a single work_fn. The bundle's
        # bodyparts override the empty-derived bodyparts from the
        # constructor (see import_seed_bundle_into_project), and the
        # iteration-0 modelfolder is manufactured so DLC sees
        # iteration-0 as already trained.
        if seed_bundle_path is not None:

            def _build_seed_and_analyze():
                project = _build_project()
                print("Installing seed bundle...")
                import_seed_bundle_into_project(project, seed_bundle_path)
                print("Running analyze_videos on iteration-0 ...")
                project.analyze_videos(
                    iteration_num=0,
                    create_video=False,
                )
                return project

            if qt_window is None:
                # Programmatic / non-Qt path: run synchronously.
                project = _build_seed_and_analyze()
                self._dlcproject = project
                self._rewire_to_in_project_paths()
                self._refresh_dlc_layers()
                self._refresh_workflow_button_state()
                return project

            def _on_seed_success(project: DLCProject):
                self._dlcproject = project
                self._rewire_to_in_project_paths()
                # Pulls predictions from videos/iteration-0/ into a
                # dense overlay layer and points the active layer at
                # an empty iteration-1 manual.
                self._refresh_dlc_layers()
                self._refresh_workflow_button_state()

            self._run_with_overlay(
                qt_window,
                work_fn=_build_seed_and_analyze,
                on_success=_on_seed_success,
                title="Creating + seeding DLC project",
                initial_phase="Scaffolding project",
                hint=(
                    "Output is also streamed to the launching terminal. "
                    "Predictions will load when you click Done."
                ),
                # analyze_videos emits a tqdm bar ("  3/100 [") that
                # the _PROGRESS_PATTERNS' tqdm-style regex consumes;
                # the same shape the Train DLC overlay uses for
                # consistency.
                show_progress_bar=True,
                phase_patterns=_SEED_PROJECT_PHASES,
                success_summary=(
                    f"Project '{name}' seeded from bundle. "
                    "Iteration-0 predictions loaded."
                ),
            )
            return None

        # Standard (non-seeded) path.
        if qt_window is None:
            self._dlcproject = _build_project()
            self._rewire_to_in_project_paths()
            self._refresh_workflow_button_state()
            return self._dlcproject

        def _on_success(project: DLCProject):
            self._dlcproject = project
            self._rewire_to_in_project_paths()
            # Create now disables (session sits inside the new project)
            # and Train enables (_dlcproject is set).
            self._refresh_workflow_button_state()

        self._run_with_overlay(
            qt_window,
            work_fn=_build_project,
            on_success=_on_success,
            title="Creating DLC project",
            initial_phase="Saving annotations and scaffolding project",
            hint="Output is also streamed to the launching terminal.",
            show_progress_bar=False,
            phase_patterns=_CREATE_PROJECT_PHASES,
            success_summary=f"Project '{name}' created.",
        )
        return None

    def _rewire_to_in_project_paths(self):
        """Repoint the live session's video + annotation paths to the
        in-project copies so subsequent writes (apply_manual_corrections,
        process_with_lk, save_annotation_as) land inside the project
        rather than next to the original video.

        Invoked once, on the GUI thread, immediately after
        :meth:`create_dlc_project` succeeds. DLC has already copied the
        video and the active layer's annotations into ``<project>/videos/``;
        this method takes care of the in-memory rewiring:

        - ``self.fname`` -> the in-project video copy. Downstream helpers
          like :func:`make_annotation_file_name` and
          ``Path(self.fname).parent`` then naturally write inside the
          project.
        - Each annotation layer whose ``.fname`` lives outside the
          project tree is migrated: ``.fname`` and ``.fstem`` are
          rewritten to the project's ``videos/`` folder; if the layer
          has in-memory data, it's saved at the new path so future
          ``save()`` calls have a destination on disk.
        - ``self.data`` (the video reader) is intentionally left
          pointing at the original file. The DLC-side ``copy_videos``
          guarantees byte-identical content, so frame seeking continues
          to work; rebuilding the reader mid-session would invalidate
          the Qt image pane handle.
        """
        assert self._dlcproject is not None
        in_project_video = str(self._dlcproject.video_list[0])
        videos_dir = Path(in_project_video).parent
        # Derive ``project_root`` from the in-project video path
        # (``videos_dir.parent``) rather than ``self._dlcproject.path``.
        # ``DLCProject.path`` is whatever the caller passed to the
        # constructor -- for a brand-new project that's the WORKING
        # DIRECTORY (parent of the actual project dir), because
        # ``deeplabcut.create_new_project`` creates the project at
        # ``<working_dir>/<name>-<experimenter>-<date>/``. The migration
        # check below ``ann_path.relative_to(project_root)`` then matches
        # a false-positive "yes, already inside" for any annotation file
        # sitting next to the original video, because the working dir IS
        # a prefix of that path -- silently skipping the migration and
        # stranding those layers at their original (outside-project)
        # locations. Affected the train pre-flight in particular: it
        # saved cleaned annotations to the wrong path, leaving the
        # project's copy stale and feeding the stale file into
        # ``extract_frames`` -> ``labeled_data``.
        project_root = videos_dir.parent

        self.fname = in_project_video

        for ann in self.annotations:
            if ann.fname is None:
                continue
            ann_path = Path(ann.fname)
            try:
                ann_path.relative_to(project_root)
                continue  # already inside project tree
            except ValueError:
                pass
            # DLC h5 traces only ever exist inside a project, so any
            # JSON outside the project (manual, dlccorr, buffer, an
            # iteration-N stub) is what we want to migrate.
            if ann_path.suffix.lower() != ".json":
                continue
            new_path = videos_dir / ann_path.name
            ann.fname = str(new_path)
            ann.fstem = new_path.stem
            if any(ann.data.values()):
                ann.save()

    def process_dlc_project(self, event=None, *args, **kwargs):
        """
        Train the DeepLabCut model without leaving the annotation UI.

        rc2 (1.1.0rc2): training runs off the GUI thread, the DUSTrack
        window stays open under a "Training in progress" overlay
        (progress bar + scrolling stdout tail), and on success the new
        DLC prediction layers are added to the live session via
        :meth:`add_annotation_layers` -- no relaunch. The overlay
        transitions to a "Complete" / "Failed" state with a Done
        button so the user can review the final stdout before the
        predictions swap in (or read the error before retrying);
        failure paths fold into the overlay rather than popping a
        separate :class:`QMessageBox`.

        1.2.0a2: a **Training options modal** runs before everything
        else (Qt path only). It surfaces
        :meth:`DLCProject.train_iteration`'s arg surface in the UI:
        refine_mode (scratch / in-project / external), source
        iteration/snapshot picker for in-project, Browse... for
        external ``.pt``, training epochs (DLC3) / iterations (DLC2),
        and a create-labeled-video toggle. Cancel returns to the UI
        without kicking off training. On accept, the underlying call
        routes through :meth:`DLCProject.train_iteration` (the
        explicit-args sibling of :meth:`DLCProject.process`) -- so the
        DLC2 silent-drop bug on ``refine=<str>`` is gone, and external
        snapshots work on both DLC versions (DLC3 via
        ``train_network(snapshot_path=...)``, DLC2 via a pose_cfg
        ``init_weights`` edit).

        A **unified pre-flight** runs before the overlay starts,
        scanning *every* manual annotation layer in the session
        (file-pattern detection -- ``.json`` alongside the video,
        ``<video_stem>_annotations*.json``, minus ``dlccorr`` /
        ``buffer`` / ``dlc*``; agnostic to which layer is active,
        the overlay, or a placeholder, and to whether the user
        renamed ``iteration-N`` to something else). For each manual
        layer we report (a) in-memory diffs vs the on-disk JSON
        and (b) frames missing one or more bodyparts. If any layer
        has either kind of issue, a single modal lists the
        per-layer breakdown and offers two actions: *Save and clean*
        (write per-layer recovery sidecars for the dropped frames,
        drop the incomplete frames from every affected layer, then
        save every affected layer, then train) and *Cancel*
        (return to the UI). Layers without issues are not touched;
        DLC trace layers and ``dlccorr`` are not in scope. Recovery
        sidecars are written as
        ``<fstem>.dustrack-dropped-incomplete-<YYYYMMDDTHHMMSS>``;
        the composite extension avoids ``.json`` so the
        annotation-discovery glob does not re-ingest them on
        subsequent training runs.

        DLC's stdout/stderr are also teed to the launching terminal
        (``sys.__stdout__``) so callers who launched from a shell can
        watch progress there as before.

        On non-Qt backends (or if the QMainWindow can't be located), the
        method falls back to the pre-rc2 behavior: close the figure,
        run training synchronously on the calling thread, and return a
        fresh DUSTrack via :meth:`DLCProject.annotate`. The pre-flight
        modal is skipped on this path (no GUI to host it).

        Args:
            event: Mouse/keyboard event (unused, for button compatibility).
            *args: Forwarded to :meth:`DLCProject.process` on the
                non-Qt fallback path; **ignored on the Qt path** (the
                Training options modal owns the kwarg surface there).
            **kwargs: Same -- non-Qt fallback only. ``create_video``
                defaults to ``False`` on the non-Qt path
                (vs. ``True`` for direct :meth:`DLCProject.process`
                calls) so the annotate -> train -> review -> annotate
                loop doesn't write a labeled mp4 each pass; the Qt
                path's create-labeled-video checkbox owns this on its
                own.

        Returns:
            DUSTrack: ``self`` on the Qt path (training is asynchronous;
            the same DUSTrack will refresh in place when the user
            clicks Done). Also ``self`` if the user cancels either
            the Training options modal or the pre-flight modal -- the
            UI is left intact. On the fallback path, the
            freshly-launched DUSTrack from :meth:`DLCProject.annotate`.

        Raises:
            ImportError: If ``deeplabcut`` isn't installed.
            ValueError: If no DLCProject has been created yet.
        """
        if not HAS_DLC:
            raise ImportError(
                "deeplabcut is not installed. Cannot process DLC project."
            )
        if self._dlcproject is None:
            raise ValueError(
                "DLCProject not created. Use create_dlc_project() to create it."
            )

        qt_window = self._find_qt_window()
        if qt_window is None:
            # Non-Qt fallback: no Training options modal possible, so
            # route through ``DLCProject.process()`` (auto-infer + sane
            # defaults). The Qt path uses ``train_iteration`` below with
            # explicit args supplied by the modal.
            kwargs.setdefault("create_video", False)
            plt.close(self.figure)
            self._dlcproject.process(*args, **kwargs)
            return self._dlcproject.annotate()

        # Qt path: empty-active-layer guard runs first so the user
        # can bail before configuring any training options. Two
        # sub-cases, distinguished by whether *any* labels exist in
        # the project (other manual layers OR previously-extracted
        # ``labeled-data/`` from prior iterations):
        #
        # - No labels anywhere (the freshly-seeded iteration-1 case):
        #   training would fail downstream because
        #   ``create_training_dataset`` has nothing to write. Hard-
        #   block with an error overlay; the user must annotate
        #   frames (or use Apply manual corrections to copy a DLC
        #   trace into a manual layer) before training.
        # - Labels exist elsewhere (mid-refinement: user clicked
        #   Train without adding new labels this pass): training is
        #   feasible by reusing the existing ``labeled-data/`` and/or
        #   running ``extract_frames`` on the other manual layer.
        #   Confirm intent with the existing modal.
        if not any(self.ann.data.values()):
            if not self._has_trainable_labels():
                self._prompt_no_trainable_labels(qt_window)
                return self  # hard block -- UI left intact
            if not self._prompt_empty_layer_train_confirm(qt_window):
                return self  # user cancelled -- UI left intact

        # Qt path: prompt for training options FIRST, then run the
        # existing pre-flight scan. The Training options modal owns the
        # full ``DLCProject.train_iteration`` kwarg surface (refine_mode
        # + in-project source picker OR external Browse + epochs +
        # create_video), so positional ``*args`` / ``**kwargs`` passed
        # to this method are ignored on the Qt path -- the user picks
        # per-click.
        training_kwargs = self._prompt_training_options(qt_window)
        if training_kwargs is None:
            return self  # user cancelled the Training options modal

        # Pre-flight: scan every manual annotation layer for
        # in-memory-vs-disk diffs AND/OR frames missing one or more
        # bodyparts. Single modal lets the user save & clean
        # everything in one click or return to the UI; layers
        # without issues are not touched. DLC traces / dlccorr /
        # buffer / non-iteration placeholders all fall out of scope
        # by file pattern, so a user who happens to have a DLC
        # trace active at click time doesn't trigger an h5 save
        # crash or a silent-overwrite of an unrelated layer.
        issues = self._scan_unsaved_and_incomplete()
        if issues:
            decision = self._prompt_unified_pre_flight(qt_window, issues)
            if not decision.proceed:
                return self  # user cancelled -- UI left intact
            self._apply_pre_flight_remediations(
                issues, strip_strays=not decision.keep_strays,
            )
            # Cleaning can drop every frame of a layer (typical
            # post-seed case: user annotated only one label out of
            # the project bodyparts, every annotated frame is
            # incomplete, all frames get dropped). Re-evaluate the
            # trainable-labels predicate after the remediation; if
            # nothing is left to train on, hard-block here with the
            # same overlay :meth:`_has_trainable_labels` uses at the
            # click-time entry guard. Without this, the cleaning
            # leaves a project that DLC will fail on at
            # create_training_dataset time with a confusing message.
            if not self._has_trainable_labels():
                self._prompt_no_trainable_labels(qt_window)
                return self

        def _train():
            self._dlcproject.train_iteration(**training_kwargs)

        def _on_success(_unused):
            # Refresh layers BEFORE the user clicks Done so the new
            # predictions are already loaded when the overlay
            # dismisses. If the refresh raises, the helper folds the
            # exception into the Done overlay (mark_done success=False
            # with the refresh error -- training itself succeeded).
            self._refresh_dlc_layers()

        self._run_with_overlay(
            qt_window,
            work_fn=_train,
            on_success=_on_success,
            title="Training in progress",
            initial_phase="Starting up",
            hint=(
                "Output is also streamed to the launching terminal. "
                "Predictions will load when you click Done."
            ),
            show_progress_bar=True,
            phase_patterns=_TRAINING_PHASES,
            success_summary="Training complete. Predictions loaded.",
        )
        return self

    # Pre-flight scan + diff + sidecar helpers — full implementations
    # live in dustrack._preflight; these are class-level aliases so
    # ``self._scan_*`` and ``cls._is_manual_*`` callsites keep working.
    _scan_incomplete_frames = staticmethod(_preflight.scan_incomplete_frames)
    _build_dropped_incomplete_payload = staticmethod(
        _preflight.build_dropped_incomplete_payload
    )
    _build_dropped_incomplete_sidecar_name = staticmethod(
        _preflight.build_dropped_incomplete_sidecar_name
    )
    _format_incomplete_breakdown = staticmethod(_preflight.format_incomplete_breakdown)
    _normalize_layer_data = staticmethod(_preflight.normalize_layer_data)
    _load_layer_disk_data = staticmethod(_preflight.load_layer_disk_data)
    _diff_ann_vs_disk = staticmethod(_preflight.diff_ann_vs_disk)
    _format_unsaved_summary = staticmethod(_preflight.format_unsaved_summary)
    _format_pre_flight_summary = staticmethod(_preflight.format_pre_flight_summary)
    _is_manual_layer_name = staticmethod(is_manual_layer_name)
    _is_manual_annotation_layer = staticmethod(is_manual_annotation_layer)

    def _save_dropped_incomplete_sidecar(self, ann, incomplete_frames: dict):
        """Persist the dropped-frame contents next to the given layer.
        See :func:`._preflight.save_dropped_incomplete_sidecar`.
        """
        return _preflight.save_dropped_incomplete_sidecar(ann, incomplete_frames)

    def _scan_unsaved_layers(self) -> dict:
        """Per-manual-layer in-memory-vs-disk diff for the active bundle.
        See :func:`._preflight.scan_unsaved_layers`.
        """
        return _preflight.scan_unsaved_layers(self.annotations, self.fname)

    def _scan_unsaved_and_incomplete(self) -> dict:
        """Combined unsaved-diff + incomplete-frame sweep.
        See :func:`._preflight.scan_unsaved_and_incomplete`.
        """
        return _preflight.scan_unsaved_and_incomplete(
            self.annotations,
            self.fname,
            dlcproject=self._dlcproject,
        )

    def _prompt_training_options(self, qt_window):
        """Show the Training options modal; return ``train_iteration`` kwargs.
        See :func:`._train_modal.prompt_training_options`.
        """
        return _train_modal.prompt_training_options(qt_window, self._dlcproject)

    def _prompt_seed_bundle(self, qt_window) -> Optional[str]:
        """Multi-step seed-bundle pick / confirm modal sequence.
        See :func:`._seed_bundle_modal.prompt_seed_bundle`.
        """
        return _seed_bundle_modal.prompt_seed_bundle(qt_window, self.ann.name)

    def _pick_from_seed_bundles(self, qt_window, root, bundles):
        """List-picker dialog. See :func:`._seed_bundle_modal.pick_from_seed_bundles`."""
        return _seed_bundle_modal.pick_from_seed_bundles(qt_window, root, bundles)

    def _browse_for_seed_bundle(self, qt_window) -> Optional[str]:
        """File-dialog Browse flow. See :func:`._seed_bundle_modal.browse_for_seed_bundle`."""
        return _seed_bundle_modal.browse_for_seed_bundle(qt_window, self.ann.name)

    def _confirm_seed_bundle(self, qt_window, bundle_path, info) -> bool:
        """Final confirm overlay. See :func:`._seed_bundle_modal.confirm_seed_bundle`."""
        return _seed_bundle_modal.confirm_seed_bundle(qt_window, bundle_path, info)

    def _maybe_remember_seed_bundles_root(self, qt_window, bundle_path) -> None:
        """Optional remember-root prompt. See :func:`._seed_bundle_modal.maybe_remember_seed_bundles_root`."""
        _seed_bundle_modal.maybe_remember_seed_bundles_root(qt_window, bundle_path)

    def _has_trainable_labels(self) -> bool:
        """True if the project has any source of labels training could consume.
        See :func:`._preflight.has_trainable_labels`.
        """
        return _preflight.has_trainable_labels(
            self.annotations,
            dlcproject=self._dlcproject,
        )

    def _prompt_no_trainable_labels(self, qt_window) -> None:
        """Hard-block overlay when no labels exist anywhere in the project.
        See :func:`._preflight_modal.prompt_no_trainable_labels`.
        """
        _preflight_modal.prompt_no_trainable_labels(qt_window, self.ann.name)

    def _prompt_empty_layer_train_confirm(self, qt_window) -> bool:
        """Confirm modal for Train-with-empty-active-layer.
        See :func:`._preflight_modal.prompt_empty_layer_train_confirm`.
        """
        return _preflight_modal.prompt_empty_layer_train_confirm(
            qt_window,
            self.ann.name,
        )

    def _prompt_unified_pre_flight(self, qt_window, issues: dict):
        """Combined save-state + incompleteness + strays modal.
        See :func:`._preflight_modal.prompt_unified_pre_flight`.
        Returns a :class:`PreFlightDecision`.
        """
        return _preflight_modal.prompt_unified_pre_flight(qt_window, issues)

    def _apply_pre_flight_remediations(
        self, issues: dict, *, strip_strays: bool = True,
    ) -> None:
        """Drop incomplete frames, optionally strip strays, save each
        affected layer, then repaint.
        See :func:`._preflight_modal.apply_pre_flight_remediations`.
        """
        _preflight_modal.apply_pre_flight_remediations(
            self.annotations,
            self.fname,
            issues,
            strip_strays=strip_strays,
        )
        self.update()

    def _prompt_save_on_close(self, qt_window, unsaved) -> str:
        """Save / Discard / Cancel modal on window close.
        See :func:`._close_guard.prompt_save_on_close`.
        """
        return _close_guard.prompt_save_on_close(qt_window, unsaved)

    def _save_unsaved_layers(self, unsaved) -> None:
        """Persist every layer with diffs across the session's bundles.
        See :func:`._close_guard.save_unsaved_layers`.
        """
        # Multi-bundle shape (1.2.0a3+): every entry carries an ``fname``.
        # Single-bundle legacy shape ({layer_name: diff}) is preserved
        # for the old API; route through self.annotations directly.
        if unsaved and "fname" in next(iter(unsaved.values()), {}):
            _close_guard.save_unsaved_layers(unsaved, self._bundles)
            return
        for layer_name in unsaved:
            self.annotations[layer_name].save()

    def _scan_unsaved_layers_all_bundles(self) -> dict:
        """Sweep every ready bundle for unsaved diffs.
        See :func:`._close_guard.scan_unsaved_layers_all_bundles`.
        """
        return _close_guard.scan_unsaved_layers_all_bundles(self._bundles)

    def _install_close_guard(self) -> None:
        """Install the QMainWindow closeEvent hook.
        See :func:`._close_guard.install_close_guard`.
        """
        _close_guard.install_close_guard(self)

    def _record_session_in_history(self) -> None:
        """Append this session's bundle list to the recent-sessions store.
        See :func:`._close_guard.record_session_in_history`.
        """
        _close_guard.record_session_in_history(self)

    def _find_qt_window(self):
        """Return the QMainWindow hosting ``self.figure``, or ``None``
        if we can't locate it (non-Qt backend, headless run, etc.).
        Wraps :func:`datanavigator._qt.find_qt_window` with the
        try/except shape that callers want.
        """
        try:
            from datanavigator._qt import find_qt_window

            return find_qt_window(self.figure)
        except Exception:
            return None

    def _install_tools_menu(self) -> None:
        """Install a Tools menu on the host QMainWindow with a "Batch
        process..." entry that opens the batch-process modal.

        Mirror of the welcome-modal's secondary Batch button, so users
        with a real session open can warm a sibling folder without
        relaunching ``dustrack.open()``. No-op when ``_find_qt_window``
        can't locate a QMainWindow (mpl fallback, headless, etc.).
        """
        qt_window = self._find_qt_window()
        if qt_window is None:
            return
        try:
            from qtpy.QtWidgets import QAction
        except ImportError:
            try:
                from qtpy.QtGui import QAction  # Qt6 home
            except ImportError:
                return
        try:
            mb = qt_window.menuBar()
        except Exception:  # noqa: BLE001
            return
        # Reuse an existing Tools menu if a re-install path ever appears;
        # otherwise create one. ``mb.actions()`` returns each top-level
        # menu's QAction, whose ``.menu()`` is the QMenu itself.
        tools_menu = None
        for action in mb.actions():
            if action.text() == "Tools":
                tools_menu = action.menu()
                break
        if tools_menu is None:
            tools_menu = mb.addMenu("Tools")
        batch_action = QAction("Batch process...", qt_window)

        def _on_batch():
            # Lazy import to keep qtpy off the import path when only
            # the library API is used.
            from ._batch_modal import open_batch_modal

            open_batch_modal(qt_window)

        batch_action.triggered.connect(_on_batch)
        tools_menu.addAction(batch_action)

    def _run_with_overlay(
        self,
        qt_window,
        *,
        work_fn,
        on_success=None,
        title: str = "Working",
        initial_phase: str = "Starting up",
        hint: str = "",
        show_progress_bar: bool = True,
        phase_patterns=None,
        success_summary: str = "Done.",
    ):
        """Drive ``work_fn`` on a worker thread under a progress
        overlay, tee its stdout/stderr to both the launching terminal
        and the overlay log, and on completion transition the overlay
        to a "Done" state. The user clicks Done to dismiss the overlay
        and (on success) fire ``on_success(result)`` on the GUI
        thread; ``result`` is the return value of ``work_fn``.

        Failure paths fold into the overlay (title flips to "Failed",
        progress bar reads "Failed", phase becomes the exception
        message); the Done button still dismisses cleanly. If
        ``on_success`` itself raises after a successful ``work_fn``,
        the overlay shows that as a separate failure mode -- the work
        succeeded, but the follow-up didn't.

        Pure GUI plumbing: the heavy thread/queue/QTimer machinery is
        all in here so the button handlers stay small.
        """
        from qtpy.QtCore import QTimer

        ProgressOverlay = _make_progress_overlay_class()
        overlay = ProgressOverlay(
            qt_window,
            title=title,
            initial_phase=initial_phase,
            hint=hint,
            show_progress_bar=show_progress_bar,
        )

        log_queue: "queue.Queue[str]" = queue.Queue()
        result_state: dict = {"exc": None, "tb": "", "done": False, "value": None}
        phase_patterns = phase_patterns or []

        def _worker():
            sink = _Tee(sys.__stdout__, _QueueWriter(log_queue))
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = sink
            sys.stderr = sink
            try:
                result_state["value"] = work_fn()
            except BaseException as e:  # noqa: BLE001 (re-raised on GUI thread)
                result_state["exc"] = e
                # Capture the traceback now (on the worker thread) so
                # the GUI thread doesn't have to walk a stale stack.
                result_state["tb"] = traceback.format_exc()
                # Push the traceback through the same teed sink so it
                # lands in the overlay log + the launching terminal
                # without us having to wire a second channel.
                try:
                    sys.stdout.write(result_state["tb"])
                    sys.stdout.flush()
                except Exception:
                    pass
            finally:
                sys.stdout, sys.stderr = old_out, old_err
                result_state["done"] = True

        thread = threading.Thread(
            target=_worker,
            name="dustrack-overlay-worker",
            daemon=True,
        )

        timer = QTimer(qt_window)
        timer.setInterval(200)

        # Per-tick state lives in a dict so the inner closure can mutate
        # without nonlocal gymnastics. ``line_buf`` is the unterminated
        # tail (no \n yet); ``last_cr_match`` lets us short-circuit
        # repeated tqdm redraws.
        tick_state = {"line_buf": ""}

        def _scan_segment(seg: str):
            """Feed a single fragment (may be \\r-terminated tqdm
            redraw or a full line) through phase + progress matchers.
            Does not append to the visible log.
            """
            if not seg.strip():
                return
            for pat, label in phase_patterns:
                if pat.search(seg):
                    overlay.set_phase(label)
                    break
            for pat in _PROGRESS_PATTERNS:
                m = pat.search(seg)
                if m:
                    cur, tot = int(m.group(1)), int(m.group(2))
                    if tot > 0 and cur <= tot:
                        overlay.set_progress(cur, tot)
                    break

        def _drain_and_update():
            chunks = []
            try:
                while True:
                    chunks.append(log_queue.get_nowait())
            except queue.Empty:
                pass
            if chunks:
                tick_state["line_buf"] += "".join(chunks)
                # Split into \n-terminated lines + unterminated tail.
                *full_lines, tick_state["line_buf"] = tick_state["line_buf"].split("\n")
                for line in full_lines:
                    # tqdm redraws use \r to overwrite the previous
                    # frame within a single logical line. Run each
                    # \r-segment through the matchers (so the
                    # progress bar updates), but only show the final
                    # segment in the log (the latest frame).
                    segments = line.split("\r") if "\r" in line else [line]
                    for seg in segments:
                        _scan_segment(seg)
                    visible = segments[-1] if segments else line
                    overlay.append_log(visible)
                # Drive progress detection off the still-unterminated
                # tail too, so tqdm's "in flight" redraws update the
                # bar between newlines.
                if "\r" in tick_state["line_buf"]:
                    for seg in tick_state["line_buf"].split("\r"):
                        _scan_segment(seg)

            if not result_state["done"]:
                return

            timer.stop()
            # Flush any trailing partial line into the log so the user
            # can read it.
            if tick_state["line_buf"].strip():
                overlay.append_log(tick_state["line_buf"].split("\r")[-1])

            exc = result_state["exc"]
            if exc is not None:
                # Stringified KeyError / IndexError / etc. is often just
                # the offending key ("0"), which is useless on its own.
                # Pair it with the type name + repr fallback so the
                # summary is always self-describing; the traceback was
                # already streamed into the overlay log on the worker
                # side.
                exc_str = str(exc) or repr(exc)
                sys.__stderr__.write(
                    f"{title} failed: {type(exc).__name__}: {exc_str}\n"
                )
                overlay.mark_done(
                    success=False,
                    summary=f"{type(exc).__name__}: {exc_str}",
                )
                return

            # work_fn succeeded -- run on_success (e.g. layer refresh)
            # on the GUI thread BEFORE showing Done, so a follow-up
            # error is surfaced in the overlay instead of after it
            # dismisses.
            value = result_state["value"]
            if on_success is not None:
                try:
                    on_success(value)
                except Exception as e:  # noqa: BLE001
                    # Mirror the worker-side failure path: stream the
                    # traceback into the overlay log so the user can
                    # diagnose without depending on the launching
                    # terminal, and fall back to repr() for exceptions
                    # whose str() is empty (bare ``assert X`` produces
                    # ``AssertionError`` with str() == "").
                    tb = traceback.format_exc()
                    sys.__stderr__.write(
                        f"{title} succeeded but follow-up failed:\n{tb}"
                    )
                    for line in tb.splitlines():
                        overlay.append_log(line)
                    exc_str = str(e) or repr(e)
                    overlay.mark_done(
                        success=False,
                        summary=(
                            f"Work succeeded, but follow-up step raised "
                            f"{type(e).__name__}: {exc_str}"
                        ),
                    )
                    return

            overlay.mark_done(success=True, summary=success_summary)

        timer.timeout.connect(_drain_and_update)
        thread.start()
        timer.start()

    def _refresh_dlc_layers(self, video_index: int = 0):
        """Load any annotation files produced by DLC training into the
        live DUSTrack session, plus a fresh empty ``iteration-{N+1}``
        layer to capture next-round manual refinements, set newly-added
        dense layers (DLC inference + any LK-RSTC output, see
        :func:`_is_dense_layer_name`) to a line plot, point the overlay
        statevar at the freshest ``dlc_*`` trace, and activate the new
        iteration layer so the user can immediately start annotating.

        Mirrors the loading + new-layer logic in
        :meth:`DLCProject.annotate` but operates on the existing
        ``self`` rather than spawning a new DUSTrack window. Idempotent:
        layers already present in ``self.annotations`` are skipped, and
        the new-iteration layer is only requested if the iteration
        suffix isn't already in the session.
        """
        # Mirror annotate()'s suffix logic: after a successful training,
        # latest_iteration_is_trained() is True so we want N+1; if
        # training is partial / not yet done, fall back to the current
        # iteration number (matches annotate() exactly).
        if self._dlcproject.latest_iteration_is_trained():
            new_iter = self._dlcproject.latest_iteration + 1
        else:
            new_iter = self._dlcproject.latest_iteration
        new_suffix = f"iteration-{new_iter}"

        fm = VideoFileManager(self._dlcproject, video_index)
        # get_new_json (called inside get_all_annotation_layers when a
        # suffix is passed) raises ValueError if the file already exists
        # on disk -- e.g. user trained, refined + saved, then re-trained
        # without restarting. Fall back to the no-suffix call in that
        # case; the layer is already known to the session via the
        # original annotate() / earlier _refresh_dlc_layers run.
        try:
            all_layers = fm.get_all_annotation_layers(new_suffix)
            requested_new_iter = True
        except ValueError:
            all_layers = fm.get_all_annotation_layers()
            requested_new_iter = False

        existing = set(self.annotations.names)
        new_layers = {
            name: path for name, path in all_layers.items() if name not in existing
        }
        if not new_layers:
            # Even with nothing to load, point the active layer at the
            # next-iteration layer if it's already present -- keeps the
            # post-training UX consistent across first-run vs re-run.
            if requested_new_iter and new_suffix in self.annotations.names:
                self.statevariables["annotation_layer"].set_state(new_suffix)
                self.update()
            return

        self.add_annotation_layers(new_layers)
        # Limit scope to newly-added layers so an in-place refresh that
        # adds zero new dlc_ layers preserves whatever overlay the user
        # had selected. On a fresh DUSTrack (DLCProject.annotate path),
        # the same helper is called with scope=None so it operates on
        # all current dlc_ layers.
        self._normalize_dlc_layer_display(scope=new_layers.keys())
        # Reset empty manual layers' labels to match the project's
        # bodyparts. Two issues this fixes (both visible after
        # seeding from an external snapshot): (a) ``add_annotation_layers``'
        # union pass adds missing labels but never removes the
        # session-bootstrap default ``"0"``, leaving spurious labels
        # when bodyparts don't start at 0; (b) for the simple
        # ``["point0", "point1"]`` case the data ends up correct but
        # the ``annotation_label`` dropdown was set at __init__ from
        # a one-label ``self.ann.labels`` snapshot and would otherwise
        # stay stale. See :func:`_dlc_bodyparts_to_layer_labels`.
        self._normalize_empty_manual_layer_labels()
        if new_suffix in self.annotations.names:
            self.statevariables["annotation_layer"].set_state(new_suffix)
        self._restructure_annotation_order()
        # Re-bootstrap the label-related statevariables from the
        # (possibly just-normalised) active layer so the dropdown
        # reflects the project's bodyparts instead of the stale
        # ``__init__``-time snapshot.
        self._rebootstrap_label_states()
        self.update()
        # 1.2.0a3: Train DLC writes ``<video>_DLC*.h5`` for every
        # video in the project, not just the one we're sitting on.
        # Propagate the new dlc_* layers to every ready non-active
        # bundle so a swap to bundle k+3 immediately shows the fresh
        # inference instead of needing a session restart. Pending
        # bundles don't need this -- when the hydration worker
        # reaches them they'll discover the new files naturally.
        self._refresh_dlc_layers_other_bundles()

    def _refresh_dlc_layers_other_bundles(self) -> None:
        """Propagate post-Train new dlc_* layers to every ready
        non-active bundle (1.2.0a3 multi-video Slice 3).

        For each ready non-active bundle:
        - Re-runs :class:`VideoFileManager` to discover the full
          current layer set for that bundle's video.
        - Identifies layers not already in the bundle's annotations
          (the just-written dlc_* files post-Train plus the new
          ``iteration-{N+1}`` empty manual).
        - Constructs each new annotation against the shell's axes
          (per-layer marker group on Tier 2 + shared trace axes),
          parks artists invisible, applies the dense-layer line
          style.

        Does NOT touch the bundle's ``selections`` -- the user might
        have a specific active layer selected for that bundle and
        we shouldn't auto-clobber it. On swap-in the rebuilt
        ``annotations.names`` rotation surfaces the new layers in
        the dropdowns; the user picks one if they want.

        No-op when no DLCProject is bound or fewer than two bundles
        exist.
        """
        if self._dlcproject is None or len(self._bundles) <= 1:
            return
        project = self._dlcproject
        if project.latest_iteration_is_trained():
            new_iter = project.latest_iteration + 1
        else:
            new_iter = project.latest_iteration
        new_suffix = f"iteration-{new_iter}"

        for bundle in self._bundles:
            if bundle.video_index == self._active_index:
                continue
            if not bundle.is_ready or bundle.annotations is None:
                continue
            try:
                self._add_new_dlc_layers_to_bundle(
                    bundle,
                    project,
                    new_suffix,
                )
            except Exception:  # noqa: BLE001 - never abort the post-train UX
                sys.__stderr__.write(
                    f"[dustrack] post-train refresh failed for bundle "
                    f"{bundle.video_index} ({bundle.fname}):\n"
                    f"{traceback.format_exc()}\n"
                )

    def _add_new_dlc_layers_to_bundle(
        self,
        bundle: _BundleState,
        project,
        new_suffix: str,
    ) -> None:
        """For a ready non-active bundle, discover + add every layer
        not already in its annotations container.

        Mirrors the new-layer pass in :meth:`_refresh_dlc_layers` but
        operates on ``bundle.annotations`` rather than
        ``self.annotations`` and parks new artists immediately.
        ``get_all_annotation_layers`` raises ``ValueError`` when the
        next-iteration JSON already exists; we fall back to the
        no-suffix discovery in that case (mirrors the active-bundle
        path).
        """
        # Re-resolve this bundle's index inside the project (the
        # bundle's queue position is independent of the project's
        # video_list order, so we look up by fname stem).
        video_index = _find_video_index(project, bundle.fname)
        if video_index is None:
            return  # not in the project -- shouldn't happen post-validation

        fm = VideoFileManager(project, video_index)
        try:
            all_layers = fm.get_all_annotation_layers(new_suffix)
        except ValueError:
            all_layers = fm.get_all_annotation_layers()

        existing = set(bundle.annotations.names)
        new_layers = {
            name: path for name, path in all_layers.items() if name not in existing
        }
        if not new_layers:
            return

        for name, fname in new_layers.items():
            if self._fast_render:
                ax_list_scatter = [self._image_pane.add_marker_group()]
            else:
                ax_list_scatter = [self._ax_image]
            ann = bundle.annotations.add(
                name=name,
                fname=fname,
                vname=str(bundle.fname),
                video=bundle.reader,
                ax_list_scatter=ax_list_scatter,
                ax_list_trace_x=[self._ax_trace_x],
                ax_list_trace_y=[self._ax_trace_y],
                palette_name="Set2",
                n_labels=1,
            )
            ann.__class__ = VideoAnnotation
            # Apply the dense-layer line style up front (matches
            # ``_normalize_dlc_layer_display``'s convention) so the
            # next swap-in renders correctly.
            if _is_dense_layer_name(name):
                try:
                    ann.set_plot_type("line", draw=False)
                except Exception:  # noqa: BLE001
                    pass
            # Park immediately so the leaving bundle's artists keep
            # painting until the user actually swaps to this one.
            ann.hide(draw=False)

    def _normalize_empty_manual_layer_labels(self) -> None:
        """When the project's bodyparts are known, reset every *empty*
        manual annotation layer's labels to match the bodyparts exactly.

        Empty here means ``not any(ann.data.values())`` -- no label
        has any frames. Layers with real annotations are untouched.

        Without this, two failure modes show up after seeding from
        an external snapshot bundle:

        - The session-bootstrap ``"manual"`` layer is created with
          ``{"0": {}}`` (the ``n_labels=1`` default in
          :class:`VideoAnnotation`). The union pass in
          :meth:`add_annotation_layers` adds new labels but never
          removes the bootstrap ``"0"``, so a bundle with bodyparts
          ``["point1", "point3"]`` produces a layer with labels
          ``["0", "1", "3"]`` instead of ``["1", "3"]``.
        - Even when bodyparts are ``["point0", "point1"]`` and the
          union ends up correct, the active layer's labels were
          captured at ``__init__`` time into the
          ``annotation_label`` statevariable; without re-bootstrap
          (see :meth:`_rebootstrap_label_states`) the dropdown shows
          only the original ``"0"``.

        Saves the layer to disk if its JSON path is already known
        (set by :meth:`_rewire_to_in_project_paths`) so the on-disk
        annotation file reflects the new label set on the next
        cold-open.
        """
        if self._dlcproject is None:
            return
        bodyparts = self._dlcproject.config.get("bodyparts") or []
        if not bodyparts:
            return
        target_labels = _dlc_bodyparts_to_layer_labels(bodyparts)
        for ann in self.annotations._list:
            if ann.name == "buffer":
                continue
            if not self._is_manual_layer_name(ann.name):
                continue
            if any(ann.data.values()):
                continue  # real data present -- preserve as-is
            if list(ann.data.keys()) == target_labels:
                continue  # already canonical
            ann.data = {label: {} for label in target_labels}
            ann.sort_labels()
            ann.re_setup_display()
            if ann.fname is not None and Path(ann.fname).suffix == ".json":
                ann.save()

    def _rebootstrap_label_states(self) -> None:
        """Refresh the ``label_range`` + ``annotation_label``
        statevariables from the *current* active layer's labels.

        At construction time, :class:`_DUSTrackBase.__init__` reads
        ``self.ann.labels`` once to seed these statevariables. Any
        downstream code that changes the active layer's labels
        (e.g. :meth:`_refresh_dlc_layers` after train / seed, which
        adds DLC trace layers and may rewrite ``iteration-N``
        labels via :meth:`_normalize_empty_manual_layer_labels`)
        leaves the dropdown stale unless this is called. The
        contents mirror the ``__init__`` bootstrap.
        """
        if not self.ann.labels:
            return
        first_label = self.ann.labels[0]
        try:
            label_int = int(first_label)
        except (TypeError, ValueError):
            return  # non-numeric labels skip the range bootstrap
        self.statevariables["label_range"].set_state(label_int // 10)
        self.update_annotation_label_states()
        states = self.statevariables["annotation_label"].states
        if states:
            self.statevariables["annotation_label"].set_state(states[0])

    def _normalize_dlc_layer_display(self, scope=None):
        """Apply the post-load display convention for DLC-pipeline layers:
        every *dense* layer renders as a line plot (DLC inference plus
        any LK-RSTC jitter-reduced output, regardless of source layer),
        and the latest ``dlc_*`` inference layer is set as the
        ``annotation_overlay``.

        Two predicates, on purpose:

        - *Dense* (see :func:`_is_dense_layer_name`) drives the
          plot-type pass. Broader than ``dlc_*`` so a Reduce-jitter
          output named ``dlccorr_lkmovavg_0.500`` (LK over the
          manual-corrections layer) renders continuously like every
          other DLC-pipeline trace.
        - ``dlc_*`` (narrow) drives the overlay pin. Overlay semantics
          are "show me the model prediction next to manual", which a
          smoothing artifact doesn't satisfy -- if the latest dense
          layer were used, clicking Reduce jitter on a manual layer
          would silently swap the overlay onto the smoothed layer.

        ``scope`` selects which layer names participate (the plot-type
        pass is idempotent so re-running on an already-line layer is a
        no-op):

        - ``None`` -- fresh-construction path: scope = all current
          layers in the session. Always (re-)points the overlay if at
          least one dlc_* layer exists.
        - iterable of names -- in-place refresh path: scope = the
          freshly-added layers only. If none of them are ``dlc_*``,
          the overlay isn't touched (preserves prior selection).

        Single source of truth shared by :meth:`DLCProject.annotate`,
        :meth:`_refresh_dlc_layers`, and :meth:`_adopt_layer` so the
        on-screen state is identical regardless of how the user
        entered the session.
        """
        if scope is None:
            names = [a.name for a in self.annotations]
        else:
            names = list(scope)
        dense_names = [n for n in names if _is_dense_layer_name(n)]
        for name in dense_names:
            self.annotations[name].set_plot_type("line")
        dlc_names = [n for n in names if n.startswith("dlc_")]
        if dlc_names:
            self.statevariables["annotation_overlay"].set_state(dlc_names[-1])

    def _restructure_annotation_order(self) -> None:
        """Regroup ``self.annotations`` into the canonical layer order
        produced by :meth:`DLCProject.annotate` at fresh load:
        ``manuals -> manual_corrections -> labeled_data -> dlc_* ->
        dlccorr* -> buffer``.

        In-session adds (post-train :meth:`_refresh_dlc_layers`,
        Reduce-jitter / :meth:`apply_manual_corrections` via
        :meth:`_adopt_layer`) append to the end of ``self.annotations``,
        which interleaves manuals with prior DLC layers and breaks the
        grouping a returning user would see on close+reopen. This helper
        re-runs after those paths so the dropdown rotation matches the
        fresh-load order regardless of how the session got here.

        Six groups, classified by layer name:

        - ``buffer`` -- the scratch layer, always last.
        - ``labeled_data`` -- DLC training-input HDF5.
        - ``dlc_*`` (prefix) -- DLC inference traces and any LK-RSTC
          smoothed version of one. Names emitted only by
          :meth:`VideoFileManager.canonical_layer_name` for files under
          ``videos/iteration-*/``.
        - ``dlccorr*`` (prefix) -- the manual-corrections splice
          produced by :meth:`apply_manual_corrections` and any
          downstream LK output of it (``dlccorr_lkmovavg_*``). Its own
          group at the tail of the DLC chain rather than folded into
          manuals: it isn't a hand-edited layer (the manual entries it
          incorporates live in a separate active layer), and isn't
          folded into the ``dlc_*`` group either since it isn't a model
          inference.
        - ``*_manual_corrections`` (suffix) -- the source-of-corrections
          layer (typically ``iteration-N_manual_corrections``) that
          :meth:`apply_manual_corrections` renames the patch layer to
          on a successful splice. Distinct from the regular manuals
          block: it's the canonical "this was the live correction
          source for the dlccorr that exists" marker, and grouped
          right after manuals so the relationship to its
          ``iteration-N`` prefix peer stays visually adjacent.
        - manuals -- everything else, i.e. ``*_annotations_*.json``-
          backed layers that aren't ``buffer`` / ``dlccorr*`` /
          ``*_manual_corrections``.

        Intra-group order is preserved (so the user's current ordering
        within each group survives a refresh). Active layer and overlay
        are preserved by name across the reorder.
        """
        manuals: list[str] = []
        manual_corr: list[str] = []
        labeled: list[str] = []
        dlc: list[str] = []
        dlccorr: list[str] = []
        buf: list[str] = []
        for ann in self.annotations._list:
            name = ann.name
            if name == "buffer":
                buf.append(name)
            elif name == "labeled_data":
                labeled.append(name)
            elif name.startswith("dlccorr"):
                dlccorr.append(name)
            elif name.startswith("dlc_"):
                dlc.append(name)
            elif name.endswith("_manual_corrections"):
                manual_corr.append(name)
            else:
                manuals.append(name)

        target = manuals + manual_corr + labeled + dlc + dlccorr + buf
        if target == self.annotations.names:
            return

        active = None
        overlay = None
        if "annotation_layer" in self.statevariables.names:
            active = self.statevariables["annotation_layer"].current_state
        if "annotation_overlay" in self.statevariables.names:
            overlay = self.statevariables["annotation_overlay"].current_state

        self.annotations.reorder(target)
        self._refresh_annotation_state_lists()

        if active is not None and active in self.annotations.names:
            self.statevariables["annotation_layer"].set_state(active)
        if "annotation_overlay" in self.statevariables.names:
            if overlay is None or overlay in self.annotations.names:
                self.statevariables["annotation_overlay"].set_state(overlay)

    def _adopt_layer(
        self,
        ann_or_fname,
        *,
        set_active: bool = False,
        set_overlay=None,
    ) -> str | None:
        """Adopt an in-session-produced annotation file into the live layer list
        under the same conventions used by the cold-open (:meth:`DLCProject.annotate`)
        and post-train (:meth:`_refresh_dlc_layers`) paths.

        Single entry point for layer additions that bypass
        :class:`VideoFileManager` (e.g. :meth:`process_with_lk`):

        - Re-derives the layer name from the filepath via
          :meth:`VideoFileManager.canonical_layer_name`. Any caller-provided
          ``.name`` on a :class:`VideoAnnotation` is ignored -- the path is
          authoritative, so a freshly-built ``VideoAnnotation`` whose
          ``.name`` would otherwise fall back to ``"noname"`` gets the
          same name it would receive on reload.
        - Adds via :meth:`add_annotation_layers` ``{name: fname}`` so the
          dnav plotting / buffer / label-union plumbing fires identically
          to the bulk-load path.
        - If the layer name is a *dense* DLC-pipeline output (see
          :func:`_is_dense_layer_name`), runs
          :meth:`_normalize_dlc_layer_display` over just this one layer
          so plot-type (and, for ``dlc_*`` names, ``annotation_overlay``)
          end up where they would after a close + reopen.
        - Skips (returns ``None``) if a layer with that name is already
          loaded -- mirroring :meth:`_refresh_dlc_layers`'s idempotency.

        Args:
            ann_or_fname: A :class:`VideoAnnotation`, ``Path``, or path
                string pointing at the annotation file on disk.
            set_active: If True, set ``annotation_layer`` to the new layer.
            set_overlay: If not None, set ``annotation_overlay`` to this
                layer name. Use this to pin the original source layer as
                overlay when the new layer is a derived (e.g. smoothed)
                version, matching the pre-harmonization behaviour of
                :meth:`process_with_lk`.

        Returns:
            The canonical layer name on success, or ``None`` if the layer
            was already loaded.
        """
        # Before dustrack 1.2.0a1 dnav.VideoAnnotation was the parent
        # and the dustrack subclass added the ``postprocess`` hook --
        # ``lk_moving_average_filter`` would return the parent type so
        # ``isinstance(..., dustrack.VideoAnnotation)`` silently fell
        # through to ``str(obj)``. Now there's a single VideoAnnotation
        # class (relocated to dustrack with postprocess attached at
        # import time in dustrack/__init__.py); the check is unambiguous.
        if isinstance(ann_or_fname, VideoAnnotation):
            fname = ann_or_fname.fname
        else:
            fname = str(ann_or_fname)
        name = VideoFileManager.canonical_layer_name(fname)
        already_loaded = name in self.annotations.names
        if not already_loaded:
            self.add_annotation_layers({name: fname})
            # Promote the freshly-added layer to the dustrack subclass so
            # ``ann.postprocess`` is available in-session (matches the
            # __init__-time promotion done after the bulk add).
            new_layer = self.annotations[name]
            if not isinstance(new_layer, VideoAnnotation):
                new_layer.__class__ = VideoAnnotation
            if _is_dense_layer_name(name):
                self._normalize_dlc_layer_display(scope=[name])
        # Apply the requested overlay / active state even if the layer
        # was already present -- e.g. Reduce jitter on a layer whose
        # cached output is already loaded should still swap the UI to
        # the smoothed layer with the source pinned as overlay.
        if set_overlay is not None:
            self.statevariables["annotation_overlay"].set_state(set_overlay)
        if set_active:
            self.statevariables["annotation_layer"].set_state(name)
        if not already_loaded:
            self._restructure_annotation_order()
        return None if already_loaded else name

    def process_with_lk(self, event=None, *args, **kwargs):
        """
        Apply Lucas-Kanade optical flow post-processing to reduce tracking jitter.

        rc2 (1.1.0rc2): on a Qt backend, the LK-RSTC pass runs off the
        GUI thread under a progress overlay that mirrors the Train DLC
        flow (tqdm bars drive the progress widget; phase label
        reflects "Submitting" vs "Processing"; Done button lets the
        user confirm before the smoothed layer swaps in). On non-Qt
        backends the call runs synchronously and returns the new
        :class:`VideoAnnotation`.

        Uses the Lucas-Kanade RSTC (Reverse Sigmoid Tracking
        Correction) algorithm to smooth trajectories. The processed
        annotation is saved and added as a new layer, with the
        original set as overlay for comparison.

        rc2 layer-naming harmonisation: the new layer is adopted via
        :meth:`_adopt_layer`, which derives its name from the output
        filepath via :meth:`VideoFileManager.canonical_layer_name` --
        identical to what a close + reopen would show. Replaces the
        earlier behaviour where the in-session layer briefly carried
        the ``"noname"`` fallback until reload. When the source layer
        is a DLC trace, the smoothed output is also plot-type-normalised
        to ``"line"`` so it looks identical to other ``dlc_*`` layers.

        The source layer is `save()`-ed to disk right before LK
        kicks off (mirroring the pre-train save in
        :meth:`process_dlc_project`), so the on-disk state matches
        what LK sees. In the typical workflow the source is the
        ``dlccorr`` layer (active after
        :meth:`apply_manual_corrections`) and the save persists any
        in-memory manual edits. Sources without a ``.json`` filename
        (e.g. raw DLC traces loaded from ``.h5``) are read-only
        inputs; the save is skipped for them with a one-line note.

        Args:
            event: Mouse/keyboard event (unused, for button compatibility).
            *args: Additional arguments passed to lk_moving_average_filter.
            **kwargs: Additional keyword arguments (e.g., window_size) passed to filter.

        Returns:
            VideoAnnotation: The smoothed annotation layer on the sync
            path. ``None`` on the Qt async path -- the new layer is
            added to ``self.annotations`` when the Done button is
            clicked.

        See Also:
            dustrack.lk_filter.lk_moving_average_filter: The filtering algorithm.
        """
        source_ann = self.ann
        source_layer_name = source_ann.name

        # Persist the source layer to disk before smoothing kicks off
        # so the on-disk state matches what LK sees. Skip gracefully
        # for non-JSON-backed sources (raw DLC traces / h5) since
        # VideoAnnotation.save() raises on non-json filenames.
        source_fname = getattr(source_ann, "fname", None)
        if source_fname is not None and Path(source_fname).suffix == ".json":
            source_ann.save()
        else:
            print(
                f"[reduce_jitter] skipping pre-save of source layer "
                f"{source_layer_name!r} (no .json filename to anchor to)."
            )

        # GUI default: skip the per-window .pkl sidecar. The .pkl is a
        # contract for ``pn-projects/wobble`` and ``gaitmusic`` callers
        # (their ``.rawlk`` property feeds ``lk_gradients`` velocity
        # estimation) -- but those go through the direct API, not the
        # button. GUI users almost never inspect per-window data; the
        # sidecar is ~10-12x larger than the averaged .json and the
        # write costs a couple of seconds on real videos. Callers who
        # want it back can pass ``save_raw=True`` explicitly.
        kwargs.setdefault("save_raw", False)

        def _smooth():
            ann_processed = lk_moving_average_filter(source_ann, *args, **kwargs)
            ann_processed.save()
            return ann_processed

        qt_window = self._find_qt_window()
        if qt_window is None:
            ann_processed = _smooth()
            self._adopt_layer(
                ann_processed,
                set_active=True,
                set_overlay=source_layer_name,
            )
            self.update()
            return ann_processed

        def _on_success(ann_processed):
            self._adopt_layer(
                ann_processed,
                set_active=True,
                set_overlay=source_layer_name,
            )
            self.update()

        self._run_with_overlay(
            qt_window,
            work_fn=_smooth,
            on_success=_on_success,
            title=f"Reducing jitter ({source_layer_name})",
            initial_phase=f"Preparing LK-RSTC pass on layer {source_layer_name!r}",
            hint=(
                "Output is also streamed to the launching terminal. "
                "The smoothed layer will load when you click Done."
            ),
            show_progress_bar=True,
            phase_patterns=_JITTER_PHASES,
            success_summary=(
                f"Jitter reduction complete on layer {source_layer_name!r}. "
                f"Smoothed layer loaded."
            ),
        )
        return None

    def detect_blips_workflow(self, event=None):
        """Detect blip outliers on the active layer + remove them.

        Two-stage flow that mirrors :meth:`process_with_lk`'s Qt-vs-mpl
        dispatch:

        1. On Qt: pop the ``BlipOptionsDialog`` modal (knob tuning +
           in-modal Detect button + results pane + drop-frame
           checkbox). Cancel returns; on Remove blips, save a
           without-blip copy of the source layer via
           :func:`dustrack.blip.remove_blips` and adopt it.
        2. On mpl-fallback: run detect + remove synchronously with
           module defaults; print a one-line summary.

        Either way, the without-blip layer adopts via
        :meth:`_adopt_layer` with the source layer pinned as overlay
        (mirrors Reduce jitter's adoption shape so the user sees the
        source DLC trace + cleaned version side-by-side immediately).

        The LK-interpolation alternative (per-blip RSTC re-track,
        producing a sparse corrections layer) shipped first but turned
        out to be less useful in the pia02 workflow than just dropping
        the contaminating frames -- the model trains on a cleaner
        subset rather than on synthesized positions. The LK function
        :func:`dustrack.blip.interpolate_blips` stays available for
        headless callers who explicitly want it.
        """
        source_ann = self.ann
        source_layer_name = source_ann.name

        def _adopt_fresh_removed(out):
            """Same reload-then-adopt as the corrections flow: on a
            re-run the in-session layer object holds the stale data
            even after the disk file is overwritten, so reload before
            handing off to the idempotent _adopt_layer.
            """
            layer_name = VideoFileManager.canonical_layer_name(out.fname)
            if layer_name in self.annotations.names:
                self.annotations[layer_name].reload()
            self._adopt_layer(
                out,
                set_active=True,
                set_overlay=source_layer_name,
            )
            self.update()

        qt_window = self._find_qt_window()
        if qt_window is None:
            # mpl-fallback: synchronous, default knobs, no modal, no
            # drop-frame option (header API ergonomics).
            report = _blip.detect_blips(source_ann)
            if len(report) == 0:
                print(
                    f"[detect_blips] no blips found on layer "
                    f"{source_layer_name!r}; nothing to remove."
                )
                return None
            out = _blip.remove_blips(
                source_ann, report, drop_frame_if_any_blip=False
            )
            out.save()
            _adopt_fresh_removed(out)
            print(
                f"[detect_blips] {len(report)} blips on layer "
                f"{source_layer_name!r}; without-blip layer saved to "
                f"{out.fname}."
            )
            return out

        modal_result = _blip_modal.prompt_blip_options(qt_window, source_ann)
        if modal_result is None:
            return None  # user clicked Cancel
        report, _knobs, drop_frame_if_any_blip = modal_result

        # Refuse to silently overwrite an existing _blip_removed file.
        from ._overlays import _make_confirm_overlay_class
        from .blip import _removed_fname

        out_path = _removed_fname(source_ann)
        if os.path.exists(out_path):
            ConfirmOverlay = _make_confirm_overlay_class()
            choice = ConfirmOverlay(
                qt_window,
                title="Without-blip file exists",
                message=(
                    f"A without-blip file already exists at\n  {out_path}\n\n"
                    "Overwrite it (the existing file is lost), or cancel?"
                ),
                buttons=[("Overwrite", "destructive"), ("Cancel", "neutral")],
                default="Cancel",
                severity="warning",
            ).exec_()
            if choice != "Overwrite":
                return None
            os.remove(out_path)

        def _remove():
            # remove_blips is a synchronous in-memory rebuild (no
            # decode, no per-blip LK); finishes in milliseconds even
            # on the 36715-frame pia02 trace, so no progress callback
            # is needed -- the ProgressOverlay's tqdm-style bar just
            # snaps to 100% before the user notices.
            out = _blip.remove_blips(
                source_ann,
                report,
                drop_frame_if_any_blip=drop_frame_if_any_blip,
            )
            out.save()
            return out

        def _on_success(out):
            _adopt_fresh_removed(out)

        policy_desc = (
            "drop whole frame on any blip"
            if drop_frame_if_any_blip
            else "drop blipped label only"
        )
        self._run_with_overlay(
            qt_window,
            work_fn=_remove,
            on_success=_on_success,
            title=f"Removing {len(report)} blips ({source_layer_name})",
            initial_phase=f"Building without-blip copy ({policy_desc})",
            hint=(
                "Output is also streamed to the launching terminal. "
                "The without-blip layer will load when you click Done."
            ),
            show_progress_bar=False,
            success_summary=(
                f"Blip removal complete: {len(report)} blips dropped "
                f"from layer {source_layer_name!r}. Without-blip layer "
                f"loaded."
            ),
        )
        return None

    def remove_current_layer(self, event=None):
        """Remove the active annotation layer from the DUSTrack session.

        Session-only: the underlying JSON / HDF5 file on disk is *not*
        touched -- so on next launch the layer reappears if its file
        is still next to the video. Pair with a manual file delete
        (or ``Save annotation as...`` to a different name + delete the
        original) when the intent is "undo manual corrections".

        Refuses with a notice if only one removable layer remains in
        the session (excluding the implicit ``"buffer"`` layer, which
        is never user-visible but always present). Otherwise confirms
        via :class:`ConfirmOverlay`; the confirm body is severity-
        aware via :func:`_is_dense_layer_name`:

        - Dense / derived (``dlc_*``, ``dlccorr``, ``*lkmovavg*``):
          regenerable, default button = Remove.
        - Sparse / authored (manual labels): irreversible, default
          button = Cancel.
        """
        qt_window = self._find_qt_window()
        layer_name = self._current_layer
        if layer_name == "buffer":
            # Defensive: ``buffer`` should never be the user-selected
            # primary, but guard anyway.
            return

        removable = [n for n in self.annotations.names if n != "buffer"]
        if len(removable) <= 1:
            if qt_window is None:
                return
            ConfirmOverlay = _make_confirm_overlay_class()
            ConfirmOverlay(
                qt_window,
                title="Cannot remove only remaining layer",
                message=(
                    f"Layer {layer_name!r} is the only annotation "
                    "layer in this session (excluding the internal "
                    "buffer). Removing it would leave the session "
                    "with no editable layer.\n\n"
                    "Use Discard unsaved annotations to reset its "
                    "contents instead."
                ),
                buttons=[("OK", "neutral")],
                default="OK",
                severity="info",
            ).exec_()
            return

        if qt_window is None:
            # mpl fallback: drop the layer silently.
            self.remove_annotation_layer(layer_name)
            self.update()
            self._refresh_workflow_button_state()
            return

        ConfirmOverlay = _make_confirm_overlay_class()
        if _is_dense_layer_name(layer_name):
            if layer_name == "dlccorr":
                regen_hint = "re-running Apply manual corrections"
            elif "lkmovavg" in layer_name:
                regen_hint = "re-running Reduce jitter"
            else:
                regen_hint = "re-running the DLC inference / training pipeline"
            body = (
                f"Remove regenerable layer {layer_name!r}?\n\n"
                f"This layer can be reproduced by {regen_hint}. "
                "The layer is dropped from the current session only; "
                "the backing file on disk is not deleted, so the "
                "layer will reappear on next launch unless you "
                "remove the file manually."
            )
            default = "Remove layer"
            severity = "warning"
        else:
            n_frames = len(self.annotations[layer_name].frames)
            body = (
                f"Remove layer {layer_name!r}? "
                f"{n_frames} manually annotated frame(s) will be "
                "dropped from this session.\n\n"
                "The backing file on disk is not deleted, so the "
                "layer will reappear on next launch unless you "
                "remove the file manually. Once removed from the "
                "session, any in-memory edits since the last save "
                "are lost."
            )
            default = "Cancel"
            severity = "destructive"

        result = ConfirmOverlay(
            qt_window,
            title="Remove layer",
            message=body,
            buttons=[
                ("Remove layer", "destructive"),
                ("Cancel", "neutral"),
            ],
            default=default,
            severity=severity,
        ).exec_()
        if result == "Remove layer":
            self.remove_annotation_layer(layer_name)
            self.update()
            # Removing the active or overlay layer may change which
            # buttons are valid; the statevar on_change covers the
            # active/overlay re-pick but not the more general "an
            # overlay layer disappeared from the dropdown" case.
            self._refresh_workflow_button_state()

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
            raise ValueError("No annotation overlay selected.")
        overlay_ann = self.annotations[overlay_name]
        current_label = self._current_label
        if (self._current_layer, current_label) in self.events[0].to_dict():
            event_start, event_end = self.events[0].to_dict()[
                (self._current_layer, current_label)
            ][0]
        else:
            event_start, event_end = 0, self.ann.n_frames - 1
        # if an event is specified, nudge data only in the selected interval
        for frame_num in self.ann.frames:
            if event_start <= frame_num <= event_end:
                self.ann.add(
                    overlay_ann.data[current_label][frame_num], current_label, frame_num
                )
        self.update()

    @staticmethod
    def _merge_overlay_with_patch(source_data, patch_data):
        """Merge two annotation ``data`` dicts -- patch overrides source.

        Both inputs are nested dicts in :class:`VideoAnnotation` shape:
        ``{label: {frame_num: [x, y]}}``. The result starts from
        ``source_data`` (the baseline, typically dense DLC predictions),
        then layers ``patch_data`` on top: at any (label, frame) present
        in patch, the patch value wins. Labels that appear in only one
        of the two inputs are carried through with their full contents.

        Pure function so the splicing logic can be exercised from
        synthetic inputs without instantiating the GUI; consumed by
        :meth:`apply_manual_corrections`.
        """
        all_labels = sorted(set(source_data) | set(patch_data))
        merged = {}
        for label in all_labels:
            merged[label] = {}
            if label in source_data:
                merged[label].update(source_data[label])
            if label in patch_data:
                merged[label].update(patch_data[label])
        return merged

    # Suffix appended to the source-of-corrections layer's name on a
    # successful :meth:`apply_manual_corrections` run. The full new name
    # is ``{old_name}_manual_corrections`` -- typically
    # ``iteration-N_manual_corrections`` -- so the rename is visually
    # adjacent to its ``iteration-N`` peer in the layer dropdown and
    # picked up by the ``_manual_corrections`` suffix branch in
    # :meth:`_restructure_annotation_order`.
    MANUAL_CORRECTIONS_SUFFIX = "_manual_corrections"

    def apply_manual_corrections(self, event=None):
        """Splice the active layer's manual entries into the overlay to produce/refresh the ``dlccorr`` layer.

        **Workflow context** (step 4 of the DUSTrack pipeline):
        after iterating DLC training (steps 2-3), you flip into a
        manual-correction mode. The active annotation layer holds
        sparse hand-edits (a few frames where DLC was wrong); the
        annotation overlay points at the DLC trace you want to
        correct. Clicking this button produces a layer named
        :attr:`CORRECTIONS_LAYER_NAME` (``"dlccorr"``) that is the
        overlay's data with your manual entries spliced in wherever
        they exist. The file lives next to the video as
        ``<video>_annotations_dlccorr.json`` and is automatically
        excluded from DLC training input (the ``_dlccorr`` filter in
        :func:`_extract_frames`) -- this is a terminal output.

        **Preflight save.** If the active (patch) layer has any
        unsaved in-memory edits, they are written to disk before the
        splice runs. The user has explicitly clicked Apply, so
        "intent to commit" is signalled; saving the source as a side
        effect keeps the on-disk state coherent with ``dlccorr``
        (which is always saved by this method).

        **Source-layer rename.** On a successful splice the patch
        layer is renamed in place from ``<old_name>`` to
        ``<old_name>_manual_corrections`` (the file is moved on disk
        too, old file deleted). The rename marks "this is the layer
        whose manual entries the on-disk ``dlccorr`` was spliced from"
        so the relationship survives reload. No-op if the patch is
        already named ``*_manual_corrections`` (idempotent re-apply).

        **Post-apply state.** The corrections layer becomes the
        active annotation layer and the (now-renamed) manual layer
        becomes the overlay so you can see where your hand was. To
        iterate, switch the active layer back to your manual layer,
        set the overlay back to the DLC trace, add more points, click
        again. Each click regenerates the corrections layer from the
        current ``(overlay, active)`` pair, so adding annotations
        directly to the corrections layer is not recommended --
        they'll be discarded on the next apply.

        **Idempotency.** If a ``dlccorr`` layer is already present in
        the session, its in-memory data is replaced wholesale (with
        a ``_revision`` bump so the trace cache invalidates) and the
        file overwritten. Otherwise a fresh :class:`VideoAnnotation`
        is built, saved, and adopted under the canonical machinery.

        Args:
            event: Mouse/keyboard event (unused, for button compat).

        Raises:
            ValueError: If no annotation overlay is currently set, or
                if the active annotation layer is already the
                corrections layer (would create a circular splice).
        """
        overlay_name = self._current_overlay
        if overlay_name is None:
            raise ValueError(
                "No annotation overlay selected. Set the layer you want to "
                "correct (typically a dlc_* trace) as the overlay before "
                "applying."
            )
        patch = self.ann
        patch_name = patch.name
        if patch_name == self.CORRECTIONS_LAYER_NAME:
            raise ValueError(
                f"The active annotation layer is already {self.CORRECTIONS_LAYER_NAME!r}. "
                "Switch the active layer to your manual annotations layer first; "
                "the corrections layer is a derived output and shouldn't be the "
                "splice input."
            )

        # Preflight: save the patch layer if it has unsaved edits, so
        # the on-disk file matches the in-memory data the splice is
        # about to consume. Scoped to the patch layer only -- the
        # overlay is typically a dlc_* h5 trace and isn't editable
        # interactively. See the apply_manual_corrections docstring
        # ("Preflight save") for the rationale.
        if patch.fname is not None:
            mem_data = self._normalize_layer_data(patch.data)
            disk_data = self._load_layer_disk_data(patch.fname)
            diff = self._diff_ann_vs_disk(mem_data, disk_data)
            if any(diff.values()):
                patch.save()
                print(
                    f"Apply manual corrections: auto-saved {patch_name!r} "
                    "(had unsaved edits) before splicing."
                )

        source = self.annotations[overlay_name]

        merged_data = self._merge_overlay_with_patch(source.data, patch.data)

        if self.CORRECTIONS_LAYER_NAME in self.annotations.names:
            # Refresh in place: replace the layer's data wholesale so the
            # file on disk and the in-memory layer agree. Bumping
            # _revision invalidates the per-label trace cache (the
            # caches keyed on (label, plot_type, _revision) -- see the
            # revision-counter pattern note).
            layer = self.annotations[self.CORRECTIONS_LAYER_NAME]
            layer.data = merged_data
            layer._revision += 1
            layer.save()
        else:
            # First-time creation: build a fresh VideoAnnotation pointing
            # at the canonical filename, populate its data, save, and
            # adopt under the same machinery used for any other layer.
            fname = make_annotation_file_name(self.fname, self.CORRECTIONS_LAYER_NAME)
            new_ann = VideoAnnotation(fname=str(fname), vname=self.fname)
            new_ann.data = merged_data
            new_ann._revision += 1
            new_ann.save()
            self._adopt_layer(new_ann)

        # Rename the patch layer to <old>_manual_corrections so the
        # source-of-corrections is identifiable on reload. Skip if
        # already suffixed (idempotent re-apply -- e.g. user re-clicked
        # Apply without changing the active layer back to the renamed
        # one in the overlay flip).
        if not patch_name.endswith(self.MANUAL_CORRECTIONS_SUFFIX):
            new_patch_name = patch_name + self.MANUAL_CORRECTIONS_SUFFIX
            self._rename_annotation_layer(patch_name, new_patch_name)
            patch_name = new_patch_name

        self.statevariables["annotation_layer"].set_state(self.CORRECTIONS_LAYER_NAME)
        self.statevariables["annotation_overlay"].set_state(patch_name)
        self._restructure_annotation_order()
        self.update()

    def _rename_annotation_layer(self, old_name: str, new_name: str) -> None:
        """Rename an annotation layer in-place: ``.name`` + ``.fname`` +
        on-disk file. The old on-disk file is deleted after the new file
        is written so a crash mid-rename leaves at least one copy on
        disk. Statevariable rotations and the active/overlay selections
        are resynced; selections that pointed at the old name are
        re-pinned to the new name.

        Used by :meth:`apply_manual_corrections` to tag the
        source-of-corrections layer with the ``_manual_corrections``
        suffix on a successful splice; written as a general helper
        because the rename mechanics (file move + statevar resync +
        selection preservation) are independent of that workflow.
        """
        if old_name not in self.annotations.names:
            raise KeyError(f"No annotation layer named {old_name!r}.")
        if new_name in self.annotations.names:
            raise ValueError(
                f"Cannot rename {old_name!r} -> {new_name!r}: a layer with "
                f"the target name already exists."
            )

        ann = self.annotations[old_name]
        old_fname = ann.fname
        # ``make_annotation_file_name`` produces
        # ``<video>_annotations_<new_name>.json`` so the on-disk filename
        # stem-after-``_annotations_`` equals the layer name -- the same
        # invariant :meth:`VideoFileManager.canonical_layer_name` relies
        # on when round-tripping a name from disk.
        new_fname = make_annotation_file_name(self.fname, new_name)

        # Write the new file before unlinking the old, so an
        # interrupted rename leaves the data on disk.
        ann.save(new_fname)
        ann.fname = new_fname
        ann.name = new_name
        if (
            old_fname is not None
            and Path(old_fname).exists()
            and str(old_fname) != str(new_fname)
        ):
            try:
                Path(old_fname).unlink()
            except OSError as exc:
                # Don't strand the user with a broken rename: log and
                # continue. The new file is in place; the stale old
                # file at worst gets picked up by the next training
                # pass alongside the new one (extract_frames will
                # merge them), which is recoverable.
                print(
                    f"Warning: could not delete old annotation file "
                    f"{old_fname!r} after rename to {new_fname!r}: {exc}"
                )

        # Resync statevariable rotations + preserve selections that
        # pointed at the old name.
        self._refresh_annotation_state_lists()
        if "annotation_layer" in self.statevariables.names:
            sv = self.statevariables["annotation_layer"]
            if sv.current_state == old_name:
                sv.set_state(new_name)
        if "annotation_overlay" in self.statevariables.names:
            sv = self.statevariables["annotation_overlay"]
            if sv.current_state == old_name:
                sv.set_state(new_name)

    def save_annotation_as(self, event=None):
        """Save the active annotation layer to a user-chosen path.

        Opens a Qt save-file dialog seeded with the video's folder and a
        suggested filename of ``<video_stem>_annotations_<layer>.json``.
        Falls back to ``self.ann.save()`` (which writes to the layer's
        existing ``.fname``) when no Qt window is available -- e.g.
        headless / Agg backend.
        """
        layer = self.ann
        suggested_dir = str(Path(self.fname).parent)
        layer_name = self._current_layer or ""
        suggested_name = Path(make_annotation_file_name(self.fname, layer_name)).name
        suggested_path = str(Path(suggested_dir) / suggested_name)

        qt_window = self._find_qt_window()
        if qt_window is None:
            layer.save()
            return

        from qtpy.QtWidgets import QFileDialog

        fname, _ = QFileDialog.getSaveFileName(
            qt_window,
            "Save annotation layer as...",
            suggested_path,
            "Annotation JSON (*.json);;All files (*)",
        )
        if not fname:
            return
        if not fname.lower().endswith(".json"):
            fname += ".json"
        layer.save(fname)

    def swap_active_and_overlay(self, event=None):
        """Swap the active annotation layer with the overlay layer.

        No-op if no overlay is currently selected.
        """
        active = self._current_layer
        overlay = self._current_overlay
        if overlay is None:
            return
        self.statevariables["annotation_layer"].set_state(overlay)
        self.statevariables["annotation_overlay"].set_state(active)
        self.update()

    def update(self):
        """
        Update the display with current frame and maintain frozen axis limits if set.

        Returns:
            Result from parent class update() method.
        """
        # Inlined from the former _DUSTrackBase.update (merged into
        # DUSTrack in 1.2.0rc1): refresh annotation visibility,
        # statevars text, and the frame marker before delegating to
        # VideoBrowser.update + plt.draw().
        self.update_annotation_visibility(draw=False)
        self.statevariables.update_display(draw=False)
        self.update_frame_marker(draw=False)
        ret = super().update()
        plt.draw()
        if self._ax_lims["state"]:
            if self._ax_lims["x"][0] is not None:
                self._ax_trace_x.set_xlim(self._ax_lims["x"])
                self._ax_trace_y.set_xlim(self._ax_lims["x"])
            if self._ax_lims["y_trace_x"][0] is not None:
                self._ax_trace_x.set_ylim(self._ax_lims["y_trace_x"])
            if self._ax_lims["y_trace_y"][0] is not None:
                self._ax_trace_y.set_ylim(self._ax_lims["y_trace_y"])
        # Two-step paint trigger for multi-video reliability:
        #
        # (1) ``QWidget.update()`` on the canvas posts a Qt-level
        #     paintEvent. Cheap (coalesced if one is already pending)
        #     and bypasses mpl's ``draw_idle`` chain, which is the
        #     diagnosed failure mode in multi-video sessions: every
        #     mechanism that empirically repairs the stale trace pane
        #     (dropdown popup closing, modal Cancel, alt-tab, opening
        #     the keyboard-shortcuts QDialog, window resize, the zoom
        #     tool's ``copy_from_bbox``) does so by causing the Qt
        #     event dispatcher to deliver a paintEvent to the canvas
        #     widget. ``draw_idle()``'s ``QTimer.singleShot(0,
        #     _draw_idle)`` evidently isn't being delivered after
        #     multi-video init even though ``figure.stale`` is True
        #     and ``_draw_pending`` looks correct.
        # (2) ``flush_events()`` (wraps ``QApplication.processEvents``)
        #     then drains the posted paintEvent synchronously so the
        #     repaint lands before this ``update()`` returns. Without
        #     the drain the paintEvent waits for the next idle, which
        #     in interactive multi-video can be too late (next user
        #     keystroke arrives first and the trace stays stale until
        #     the next external paint trigger).
        #
        # Bench: paintEvent + processEvents should be comparable to
        # the prior ``flush_events()``-only path (~22.6 ms / 44 fps on
        # probe 28). Falls back silently on the mpl-fallback (non-Qt)
        # path. If this combo still leaves the multi-video trace
        # stale, the next diagnostic step is to log
        # ``figure.stale`` + ``canvas._draw_pending`` at the entry
        # to ``update()`` and around each repair trigger to pinpoint
        # whether (a) ``draw_idle`` is being scheduled but never
        # delivered, or (b) ``set_data`` is not setting stale=True
        # after the first paint cycle.
        # try:
        #     self.figure.canvas.update()
        # except Exception:  # noqa: BLE001
        #     pass
        try:
            self.figure.canvas.flush_events()
        except Exception:  # noqa: BLE001
            pass
        return ret

    # ------------------------------------------------------------------
    # Multi-video swap-state machinery (1.2.0a3)
    # ------------------------------------------------------------------
    #
    # See ``specs/dustrack.md`` (Roadmap *Next 1.2.0* item 3) and
    # ``dustrack/_bundle.py`` for the contract. The shell holds one set
    # of UI widgets (figure + axes + dock + sidebar + EnhanceWidget +
    # statevars container + image pane) and one ``_BundleState`` per
    # input video. A swap = park leaving bundle's artists (set_visible
    # False), rebind shell to arriving bundle, show arriving bundle's
    # artists, restore lightweight UI snapshot.
    # ------------------------------------------------------------------

    # Statevars that propagate across bundles when changed on any one
    # bundle. Only the genuinely UI-mode statevar (``number_keys``,
    # select-vs-place) broadcasts -- everything else stays per-bundle
    # so the user's per-video work (active label, layer rotation,
    # etc.) survives swap-out / swap-in cycles independently of what
    # they're doing on other bundles. Per the 2026-05-21 user-cue:
    # "If I switch to label 1 in video 1, then switch to the next
    # video, I see label 1 (instead of label 0)" -- label state must
    # be per-bundle, not broadcast.
    _BROADCAST_STATEVARS = ("number_keys",)

    # Names of all five DUSTrack statevars; the snapshot/restore loop
    # walks this list and silently ignores names that aren't present
    # (e.g. mpl-fallback path that skips some Qt-only statevars).
    _ALL_TRACKED_STATEVARS = (
        "annotation_layer",
        "annotation_overlay",
        "annotation_label",
        "label_range",
        "number_keys",
    )

    def _init_bundles(self, project, video_paths: list) -> None:
        """Populate ``_bundles`` from the shell + queued paths.
        See :func:`._bundle_swap.init_bundles`.
        """
        _bundle_swap.init_bundles(self, project, video_paths)

    def _install_broadcast_statevar_hooks(self) -> None:
        """Wire cross-bundle broadcast for UI-mode statevars.
        See :func:`._bundle_swap.install_broadcast_statevar_hooks`.
        """
        _bundle_swap.install_broadcast_statevar_hooks(self)

    def _broadcast_statevar(self, sv_name: str) -> None:
        """Propagate a statevar change to every bundle.
        See :func:`._bundle_swap.broadcast_statevar`.
        """
        _bundle_swap.broadcast_statevar(self, sv_name)

    def _await_hydration(self, bundle: _BundleState) -> bool:
        """Block (pumping Qt) until ``bundle`` is terminal.
        See :func:`._bundle_swap.await_hydration`.
        """
        return _bundle_swap.await_hydration(bundle)

    def _hydrate_bundle_data_only(
        self,
        bundle: _BundleState,
        project,
    ) -> dict:
        """Off-thread half of Phase 2 bundle hydration.
        See :func:`._bundle.hydrate_bundle_data_only`.
        """
        return _bundle_helpers.hydrate_bundle_data_only(self, bundle, project)

    def _hydrate_phase1_bundle_data(
        self,
        bundle: _BundleState,
        *,
        layer_name: str = "iteration-0",
    ) -> dict:
        """Off-thread half of Phase 1 hydration.
        See :func:`._bundle.hydrate_phase1_bundle_data`.
        """
        return _bundle_helpers.hydrate_phase1_bundle_data(
            self,
            bundle,
            layer_name=layer_name,
        )

    def _finalise_bundle_artists(
        self,
        bundle: _BundleState,
        payload: dict,
        project,
    ) -> None:
        """Qt-thread half of bundle hydration.
        See :func:`._bundle.finalise_bundle_artists`.
        """
        _bundle_helpers.finalise_bundle_artists(self, bundle, payload, project)

    def _derive_initial_bundle_selections(
        self,
        container: VideoAnnotations,
        project=None,
    ) -> dict:
        """First-time statevar selections for a hydrated bundle.
        See :func:`._bundle.derive_initial_bundle_selections`.
        """
        return _bundle_helpers.derive_initial_bundle_selections(
            self,
            container,
            project=project,
        )

    def _capture_statevar_selections(self) -> dict:
        """Snapshot the shell's statevar selections for the active bundle.
        See :func:`._bundle_swap.capture_statevar_selections`.
        """
        return _bundle_swap.capture_statevar_selections(self)

    def _restore_statevar_selections(
        self,
        selections: dict,
        layer_names: list,
    ) -> None:
        """Restore bundle selections + sync the Qt sidebar widgets.
        See :func:`._bundle_swap.restore_statevar_selections`.
        """
        _bundle_swap.restore_statevar_selections(self, selections, layer_names)

    # ------------------------------------------------------------------
    # Image-pane viewport snapshot / restore (Tier 1 + Tier 2 dispatch)
    # ------------------------------------------------------------------

    def _get_image_view_state(self):
        """Snapshot the image pane's zoom / pan state.
        See :func:`._view_state.get_image_view_state`.
        """
        return _view_state.get_image_view_state(self)

    def _set_image_view_state(self, state) -> None:
        """Restore a snapshotted image-pane viewport.
        See :func:`._view_state.set_image_view_state`.
        """
        _view_state.set_image_view_state(self, state)

    # ------------------------------------------------------------------
    # Swap entry points
    # ------------------------------------------------------------------

    def swap_to(self, index: int) -> bool:
        """Switch the active video to ``self._bundles[index]``.
        See :func:`._bundle_swap.swap_to`.
        """
        return _bundle_swap.swap_to(self, index)

    def swap_prev(self, event=None) -> bool:
        """Move to the previous bundle (no-op at index 0).

        Connected to the sidebar's ``◀`` button and the ``Alt+Left``
        keybinding. Returns the underlying :meth:`swap_to` result so
        keybinding-handler callers can short-circuit if desired.
        """
        return self.swap_to(self._active_index - 1)

    def swap_next(self, event=None) -> bool:
        """Move to the next bundle (no-op at last index).

        Connected to the sidebar's ``▶`` button and the ``Alt+Right``
        keybinding.
        """
        return self.swap_to(self._active_index + 1)

    # ------------------------------------------------------------------
    # Bundle list management (add / remove / replace-active)
    # ------------------------------------------------------------------

    def add_video(
        self,
        path_or_paths,
        *,
        layer_name=None,
        set_active=False,
        **dustrack_kwargs,
    ) -> list[int]:
        """Append one or more videos to this tracker's bundle list.
        See :func:`._bundle_swap.add_video`.
        """
        return _bundle_swap.add_video(
            self,
            path_or_paths,
            layer_name=layer_name,
            set_active=set_active,
            **dustrack_kwargs,
        )

    def remove_video(self, index: int) -> bool:
        """Drop a bundle from the tracker's bundle list.
        See :func:`._bundle_swap.remove_video`.
        """
        return _bundle_swap.remove_video(self, index)

    def replace_active_with(
        self,
        path_or_paths,
        *,
        layer_name=None,
        **dustrack_kwargs,
    ) -> list[int]:
        """Swap the active bundle for newly-picked video(s).
        See :func:`._bundle_swap.replace_active_with`.
        """
        return _bundle_swap.replace_active_with(
            self,
            path_or_paths,
            layer_name=layer_name,
            **dustrack_kwargs,
        )

    def _validate_bundle_paths(self, path_or_paths) -> tuple:
        """Resolve ``path_or_paths`` into ``(project_or_None, [Path...])``.
        See :func:`._bundle_swap.validate_bundle_paths`.
        """
        return _bundle_swap.validate_bundle_paths(path_or_paths)

    def _hydrate_bundle_sync(self, bundle: _BundleState, project=None) -> None:
        """Sync hydration (data half + artists) for one bundle.
        See :func:`._bundle.hydrate_bundle_sync`.
        """
        _bundle_helpers.hydrate_bundle_sync(self, bundle, project=project)

    def _snapshot_active_bundle(self) -> None:
        """Write the shell's UI state back to the active bundle.
        See :func:`._bundle_swap.snapshot_active_bundle`.
        """
        _bundle_swap.snapshot_active_bundle(self)

    def _get_enhance_state(self) -> dict:
        """Snapshot CLAHE/gamma/brightness for per-bundle preservation.
        See :func:`._view_state.get_enhance_state`.
        """
        return _view_state.get_enhance_state(self)

    def _set_enhance_state(self, state) -> None:
        """Restore enhance sliders (or initial defaults on first visit).
        See :func:`._view_state.set_enhance_state`.
        """
        _view_state.set_enhance_state(self, state)

    def _get_trace_view_state(self) -> dict:
        """Snapshot the trace axes' xlim / ylim.
        See :func:`._view_state.get_trace_view_state`.
        """
        return _view_state.get_trace_view_state(self)

    def _set_trace_view_state(self, state) -> None:
        """Restore snapshotted trace-axes view (None = first visit).
        See :func:`._view_state.set_trace_view_state`.
        """
        _view_state.set_trace_view_state(self, state)

    def _attach_bundle(self, bundle: _BundleState) -> None:
        """Rebind shell attributes onto ``bundle``'s heavy state.
        See :func:`._bundle_swap.attach_bundle`.
        """
        _bundle_swap.attach_bundle(self, bundle)

    def _park_bundle_artists(self, bundle: _BundleState) -> None:
        """Hide every annotation artist owned by ``bundle``.
        See :func:`._bundle.park_bundle_artists`.
        """
        _bundle_helpers.park_bundle_artists(bundle)

    def _show_bundle_artists(self, bundle: _BundleState) -> None:
        """Show every annotation artist owned by ``bundle``.
        See :func:`._bundle.show_bundle_artists`.
        """
        _bundle_helpers.show_bundle_artists(bundle)

    def _notify_bundle_failure(self, bundle: _BundleState) -> None:
        """Surface a hydration failure to stderr.
        See :func:`._bundle.notify_bundle_failure`.
        """
        _bundle_helpers.notify_bundle_failure(bundle)

    # ------------------------------------------------------------------
    # Sidebar nav row + key bindings
    # ------------------------------------------------------------------

    def _add_nav_widget(self) -> None:
        """Mount the multi-video nav row at the top of the dock.
        See :func:`._nav_widget.add_nav_widget`.
        """
        _nav_widget.add_nav_widget(self)

    def _on_nav_combo_activated(self, index: int) -> None:
        """Dropdown-selection -> swap_to.
        See :func:`._nav_widget.on_nav_combo_activated`.
        """
        _nav_widget.on_nav_combo_activated(self, index)

    def _add_video_nav_key_bindings(self) -> None:
        """Bind Alt+Left / Alt+Right to prev / next video.
        See :func:`._nav_widget.add_video_nav_key_bindings`.
        """
        _nav_widget.add_video_nav_key_bindings(self)

    def _refresh_nav_buttons(self) -> None:
        """Sync nav row to current bundles + active.
        See :func:`._nav_widget.refresh_nav_buttons`.
        """
        _nav_widget.refresh_nav_buttons(self)

    _format_nav_combo_item = staticmethod(_nav_widget.format_nav_combo_item)

    def _sync_nav_combo(self, combo, *, n: int, active: int) -> None:
        """Sync the QComboBox items + selection to the bundle list.
        See :func:`._nav_widget.sync_nav_combo`.
        """
        _nav_widget.sync_nav_combo(self, combo, n=n, active=active)

    # ==================================================================
    # Methods absorbed from the former _DUSTrackBase parent class.
    # Collapsed in 1.2.0rc1; the parent/child split was no longer
    # meaningful (one subclass, no enforcement). The original docstrings
    # + behavior are preserved verbatim, only the @classmethod's return
    # annotation flipped from ``_DUSTrackBase`` to ``DUSTrack``.
    # ==================================================================

    @classmethod
    def from_annotations(
        cls, annotations: "list[VideoAnnotation]", *args, **kwargs
    ) -> "DUSTrack":
        if isinstance(annotations, VideoAnnotation):
            annotations = [annotations]
        video_names = {a.video.fname for a in annotations}
        assert len(video_names) == 1  # same video across all annotations
        return cls(video_names.pop(), annotations, *args, **kwargs)

    def add_annotation_layers(
        self,
        annotation_names: "list[str] | dict[str, Path] | list[VideoAnnotation]",
        n_labels: int = 1,
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
        assert (
            name in self.annotations.names
        ), f"layer {name!r} not in {self.annotations.names!r}"
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
        keyboard_shortcuts.png``: layer selection -> label selection ->
        frame navigation -> edit -> refine. ``self._section_order`` pins
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
        self.add_key_binding(
            "=",
            self.next_annotation_layer,
            "Next annotation layer (primary)",
            group=sec1,
        )
        self.add_key_binding(
            "-",
            self.previous_annotation_layer,
            "Previous annotation layer (primary)",
            group=sec1,
        )
        self.add_key_binding(
            "]",
            self.next_annotation_overlay,
            "Next annotation layer (overlay)",
            group=sec1,
        )
        self.add_key_binding(
            "[",
            self.previous_annotation_overlay,
            "Previous annotation layer (overlay)",
            group=sec1,
        )

        # 2. Select annotation number (#)
        self.add_key_binding(
            "'", self.next_annotation_label, "Next annotation label (#)", group=sec2
        )
        self.add_key_binding(
            ";",
            self.previous_annotation_label,
            "Previous annotation label (#)",
            group=sec2,
        )
        self.add_key_binding(
            "w", self.increment_label_range, "Next annotation # range", group=sec2
        )
        self.add_key_binding(
            "q", self.decrement_label_range, "Previous annotation # range", group=sec2
        )

        # 3. Navigate to the desired video frame
        self.add_key_binding("g", self.increment, "Next video frame", group=sec3)
        self.add_key_binding(
            "f",
            self.increment_if_unannotated,
            "Next video frame if unannotated",
            group=sec3,
        )
        self.add_key_binding(
            "d",
            self.decrement_if_unannotated,
            "Previous video frame if unannotated",
            group=sec3,
        )
        self.add_key_binding(
            ",",
            self.previous_frame_with_any_label,
            "Previous frame with any annotation",
            group=sec3,
        )
        self.add_key_binding(
            ".",
            self.next_frame_with_any_label,
            "Next frame with any annotation",
            group=sec3,
        )
        self.add_key_binding(
            "alt+,",
            self.previous_frame_of_interest,
            "Previous frame of interest",
            group=sec3,
        )
        self.add_key_binding(
            "alt+.", self.next_frame_of_interest, "Next frame of interest", group=sec3
        )
        self.add_key_binding(
            "n",
            self.next_frame_with_current_label,
            "Next frame with current annotation",
            group=sec3,
        )
        self.add_key_binding(
            "p",
            self.previous_frame_with_current_label,
            "Previous frame with current annotation",
            group=sec3,
        )
        self.add_key_binding(
            "b",
            self.previous_frame_with_current_label,
            "Previous frame with current annotation (alias of p)",
            group=sec3,
        )

        # 4. Edit annotation
        self.add_key_binding(
            "t", self.add_annotation, "Add annotation (hover on image)", group=sec4
        )
        self.add_key_binding(
            "y",
            self.remove_annotation,
            "Remove annotation (hover near it on image)",
            group=sec4,
        )

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
                "z",
                z_binding.callback,
                "Select interval (press once at start, once at end)",
                group=sec5b,
            )
        self.add_key_binding(
            "a",
            self.interpolate_with_lk,
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
            "alt+a",
            self.remove_labels_in_interval,
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
            "x",
            self.decimate_annotations_in_interval,
            "Decimate in selected interval -- drop incomplete, then halve",
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
        self.add_key_binding(
            "m",
            self.toggle_frame_of_interest,
            "Toggle (mark / unmark) current frame as a frame of interest",
            group=sec5c,
        )
        self.add_key_binding(
            "alt+c",
            self.copy_frames_of_interest_from_overlay,
            "Copy annotations at frames of interest from overlay",
            group=sec5c,
        )
        self.add_key_binding(
            "c",
            self.copy_current_annotation_from_overlay,
            "Copy current annotation at current frame from overlay",
            group=sec5c,
        )
        self.add_key_binding(
            "ctrl+alt+c",
            self.copy_frames_in_interval_from_overlay,
            "Copy annotations in selected interval from overlay",
            group=sec5c,
        )

        # Bindings not depicted on the docs PNG -- fall through to "Other".
        self.add_key_binding("s", self.save, "Save current annotation layer")
        self.add_key_binding(
            "`",
            self.cycle_number_keys_behavior,
            "Toggle num-keys mode (select / place)",
        )
        self.add_key_binding(
            "alt+q",
            self.keep_overlapping_continuous_frames,
            "Keep only consecutive frames where every label is annotated",
        )
        self.add_key_binding(
            "f5",
            self.refresh,
            "Refresh UI from current annotation data",
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

        self.remove_key_binding(
            "e"
        )  # remove the "Extract clip" feature from VideoBrowser

        if self._fast_render:
            # Overwrite the inherited 'r' binding (GenericBrowser.reset_axes)
            # so a single keystroke resets the pane under the cursor (image
            # OR traces, not both), with a fall-through to the pre-rc2
            # "reset everything" behaviour when the cursor is undetectable
            # -- preserving the catch-all muscle memory. Tier 1 keeps the
            # inherited binding (no image zoom to reset).
            self.add_key_binding(
                "r",
                self._reset_view_all,
                "Reset view under cursor (traces: full-video x, active label y)",
            )
            self.add_key_binding(
                "alt+r",
                self._reset_view_to_data_extent,
                "Reset view under cursor (traces: data-extent x, all-labels y)",
            )

    def _reset_view_all(self, event: "Any | None" = None) -> None:
        """Cursor-aware ``r`` dispatch (Tier 2 only); traces use full-video x.

        Three branches, keyed on ``event.inaxes`` after
        :meth:`_patch_event_for_image_pane` has patched the event in
        :meth:`__call__`:

        - Cursor over the Tier 2 image pane (``inaxes == self._ax_image``,
          which mirrors ``self._image_pane``) -> image-pane zoom/pan reset
          only; trace axes untouched.
        - Cursor over a trace axis (``inaxes in (self._ax_trace_x,
          self._ax_trace_y)``) -> trace x set to ``(0, ann.n_frames)`` and
          y fit to the **active label** (active + overlay layers, if both
          carry it; see :meth:`_fit_y_to_active_label`). Image pane
          untouched. Setting x to the full video range (rather than
          autoscaling to the annotation data extent) keeps frames
          *outside* the current annotation envelope visible, which is the
          usual case when extending annotations to a new region. See
          ``_reset_view_to_data_extent`` for the autoscale-x +
          union-y sibling, bound to ``alt+r``.
        - Cursor anywhere else / event undetectable (``event is None`` or
          ``event.inaxes is None`` or some unrelated mpl axis) -> reset
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

    def _reset_view_to_data_extent(self, event: "Any | None" = None) -> None:
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

    def _reset_traces_to_full_video(self, event: "Any | None" = None) -> None:
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

    def _fit_y_to_active_label(self, event: "Any | None" = None) -> None:
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

        y_x_vals: "list[np.ndarray]" = []
        y_y_vals: "list[np.ndarray]" = []
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

        def _apply(ax: plt.Axes, parts: "list[np.ndarray]") -> None:
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

    def refresh(self, event: "Any | None" = None) -> None:
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
            data_id_func=(
                lambda s: (Path(s.fname).stem, s._current_layer, s._current_label)
            ).__get__(self),
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
    def ann(self) -> "VideoAnnotation":
        """Return current annotation layer."""
        return self.annotations[self._current_layer]

    @property
    def _current_label(self) -> str:
        """Return current label."""
        return str(
            int(self.statevariables["annotation_label"].current_state)
            + int(self.statevariables["label_range"]._current_state_idx) * 10
        )

    @property
    def _current_layer(self) -> str:
        """Return current annotation layer"""
        return self.statevariables["annotation_layer"].current_state

    @property
    def _current_overlay(self) -> "str | None":
        """Return current annotation overlay layer"""
        return self.statevariables["annotation_overlay"].current_state

    def _get_fname_annotations(
        self, annotation_name: str, suffix: str = ".json"
    ) -> str:
        """Construct the filename corresponding to an annotation layer named annotation_name."""
        return get_fname_annotations(self.fname, annotation_name, suffix)

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
        if (
            event.name == "key_press_event"
            and str(event.key).isdigit()
            and int(event.key) in range(10)
        ):
            key_str = str(event.key)
            key_int = int(event.key)
            label = str(
                key_int
                + int(self.statevariables["label_range"]._current_state_idx) * 10
            )

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

            default_x, default_y = (
                self._ax_trace_x.get_ylim(),
                self._ax_trace_y.get_ylim(),
            )
            xl, yl = nanlim(trace_data_x, default_x), nanlim(trace_data_y, default_y)
            xls, yls = nanlim_small(trace_data_x, default_x), nanlim_small(
                trace_data_y, default_y
            )

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
        location: "list[float]",
        frame_number: "int | None" = None,
        label: "str | None" = None,
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

    def remove_annotation(self, event: "Any | None" = None) -> None:
        """remove annotation at the current frame if it exists"""
        self.ann.remove(self._current_label, self._current_idx)
        self.update()

    def _get_nearest_annotated_frame(self, label: "str | None" = None) -> int:
        """Return the nearest frame (in either direction) number with the current label in the current annotation layer."""
        if label is None:
            label = self._current_label
        d = {abs(x - self._current_idx): x for x in self.ann.get_frames(label)}
        if not d:
            raise ValueError(
                f"No frames with label {label} in annotation layer {self._current_layer}."
            )
        return d[min(d)]

    def previous_frame_with_current_label(self, event: "Any | None" = None) -> None:
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

    def next_frame_with_current_label(self, event: "Any | None" = None) -> None:
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

    def previous_frame_with_any_label(self, event: "Any | None" = None) -> None:
        """Go to the previous frame with any label in the current annotation layer."""
        self._current_idx = max(
            [x for x in self.ann.frames if x < self._current_idx],
            default=self._current_idx,
        )
        self.update()

    def next_frame_with_any_label(self, event: "Any | None" = None) -> None:
        """Go to the next frame with any label in the current annotation layer."""
        self._current_idx = min(
            [x for x in self.ann.frames if x > self._current_idx],
            default=self._current_idx,
        )
        self.update()

    def previous_frame_of_interest(self, event: "Any | None" = None) -> None:
        """Go to the previous frame of interest."""
        self._current_idx = max(
            [x for x in self.frames_of_interest if x < self._current_idx],
            default=self._current_idx,
        )
        self.update()

    def next_frame_of_interest(self, event: "Any | None" = None) -> None:
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
        label_states = [
            str(int(x) % 10)
            for x in self.ann.labels
            if int(x)
            in range(
                int(self.statevariables["label_range"]._current_state_idx) * 10,
                (int(self.statevariables["label_range"]._current_state_idx) + 1) * 10,
            )
        ]
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

    def increment_if_unannotated(self, event: "Any | None" = None) -> None:
        """Advance the frame if the current frame doesn't have any annotations.
        Useful to pause at annotated frames when adding a new label.
        """
        if self._current_idx not in self.ann.frames:
            self.increment()

    def decrement_if_unannotated(self, event: "Any | None" = None) -> None:
        """Go to the previous frame if the current frame doesn't have any annotations.
        Useful to pause at annotated frames when adding a new label.
        """
        if self._current_idx not in self.ann.frames:
            self.decrement()

    def save(self) -> None:
        """Save current annotation layer json file.

        ``.h5`` layers (DLC predicted traces, ``labeled_data`` blocks)
        are not in-place editable through this key — the JSON-only
        contract on :meth:`VideoAnnotation.save` would raise mid-keypress
        and look like a silent no-op. Short-circuit with a clear
        printed message that points the user to manual layers or to
        ``Save annotation as...`` for explicit copy-out.
        """
        ann_fname = getattr(self.ann, "fname", None)
        if ann_fname is not None and Path(ann_fname).suffix.lower() == ".h5":
            layer_name = getattr(self.ann, "name", None) or Path(ann_fname).stem
            print(
                f"[save] Layer {layer_name!r} is a .h5 file; the 's' key "
                f"saves only manual .json layers. Switch to a manual layer "
                f"or use 'Save annotation as...' (sidebar) to copy this "
                f"layer to a new .json file."
            )
            return
        self.ann.save()

    def select_label_with_mouse(self, event: Any) -> None:
        """Select a label by clicking on it with the left mousebutton."""
        if event.mouseevent.button.name == "LEFT":
            picked_label = self.ann.labels[int(event.ind[0])]
            self.statevariables["label_range"]._current_state_idx = (
                int(picked_label) // 10
            )
            self.update_annotation_label_states()
            self.statevariables["annotation_label"].set_state(
                str(int(picked_label) % 10)
            )
            print(f"Picked label {picked_label} at frame {self._current_idx}")
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
        labels: "str | list[str]" = "all",
        start_frame: "int | None" = None,
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

    def get_selected_interval(self) -> "tuple[int, int]":
        start_frame, end_frame = (
            self.events["interp_with_lk"]
            ._data[(Path(self.fname).stem, self._current_layer, self._current_label)]
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

    def decimate_annotations_in_interval(self) -> None:
        """Prune incomplete frames, then drop every other complete frame in the selected interval.

        Prep step for training: ensures the surviving in-interval
        frames are (a) fully annotated across the required-label set
        and (b) half as dense (even-stride sampling). Frame-level --
        every label is removed at each dropped frame.

        Required-label set follows the same project-aware /
        project-unaware split the Train preflight uses
        (:func:`_preflight.scan_incomplete_frames`):

        * **DLC project loaded** -- required = project ``bodyparts``
          (mapped through :func:`_dlc_bodyparts_to_layer_labels`).
          Stray non-bodypart labels on the layer are ignored from the
          required check, but any frame they sit on is still part of
          ``all_frames`` -- if it's missing a required label it still
          counts as incomplete, and pruning cleans the stray with it.
        * **No DLC project** -- required = labels with at least one
          annotation. Empty labels are treated as UI placeholders
          (user created a slot and abandoned it). This is the best
          inference available without external truth and matches the
          project-unaware Train preflight rule.

        Starter form of the "general-model workflow" decimation
        feature; the DINOv3-feature farthest-point-sampling variant
        is deferred. Incomplete frames in the interval are always
        pruned. The every-other halving is a no-op when fewer than
        2 complete frames remain in the interval after pruning.
        """
        start_frame, end_frame = self.get_selected_interval()
        labels = list(self.ann.labels)
        target_labels: "list[str] | None" = None
        if self._dlcproject is not None:
            bodyparts = self._dlcproject.config.get("bodyparts") or []
            if bodyparts:
                target_labels = _dlc_bodyparts_to_layer_labels(bodyparts)
        incomplete_in_interval = sorted(
            f
            for f in _preflight.scan_incomplete_frames(
                self.ann.data, target_labels=target_labels,
            )
            if start_frame <= f <= end_frame
        )
        for frame_number in incomplete_in_interval:
            for label in labels:
                self.ann.remove(label, frame_number)
        complete_in_interval = sorted(
            {
                f
                for label in labels
                for f in self.ann.get_frames(label)
                if start_frame <= f <= end_frame
            }
        )
        for frame_number in complete_in_interval[1::2]:
            for label in labels:
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
                    target_ann.add(list(rstc_path[frame_count, 0, :]), label, frame_num)
                rstc_paths[label][start_frame : end_frame + 1, :] = np.squeeze(
                    rstc_path
                )
        self.update()

    def render(
        self, start_frame: int, end_frame: int, out_vid_name: "str | None" = None
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
