"""The interactive DUSTrack GUI class.

:class:`DUSTrack` is the user-facing point-tracking widget; it inherits
from :class:`dustrack.pointtracking._DUSTrackBase` (a
``datanavigator.VideoPointAnnotator`` subclass) and adds:

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
from .pointtracking import _DUSTrackBase
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



from .dlcinterface import (
    DLCProject,
    _find_dlc_config,
    _find_video_index,
    _is_dlc_config_yaml,
    _is_dlc_project_root,
    _resolve_multi_video_from_list,
    _session_inside_dlc_project,
)

class DUSTrack(_DUSTrackBase):
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

    def __init__(self, vid_name, annotation_names="iteration-0", *args,
                 clahe_clip=1.0, clahe_grid=8, gamma=1.0, brightness=0,
                 dark_mode=False, **kwargs):
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
                    im, self._clahe_clip, self._clahe_grid,
                    self._gamma, self._brightness
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

        kwargs['image_process_func'] = image_processor
        # DUSTrack defaults to datanavigator 1.5.0+ Tier 2 (Qt-native
        # video pane, ~3x speedup on real videos). Override with
        # ``DUSTrack(..., fast_render=False)`` only if a subclass needs
        # matplotlib Axes on the image region (no in-tree subclass
        # does today; this is forward-looking).
        kwargs.setdefault('fast_render', True)
        # Pin the QApplication palette before any widget is built so
        # appearance is reproducible across Qt bindings + OS themes.
        # See :func:`_pin_qt_palette` for the rationale.
        _pin_qt_palette(dark_mode)
        super().__init__(vid_name, annotation_names, *args, **kwargs)

        for ann in self.annotations:
            ann.__class__ = VideoAnnotation

        self._dlcproject = None
        self._ax_lims = {'state': False, 'x': [None, None], 'y_trace_x': [None, None], 'y_trace_y': [None, None]}

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
            self.buttons.add(text="Create DLC Project", action_func=self.create_dlc_project, style_tag="workflow")
            self.buttons.add(text="Train DLC model", action_func=self.process_dlc_project, style_tag="workflow")
            self.buttons.add(text="Apply manual corrections", action_func=self.apply_manual_corrections, style_tag="workflow")
            self.buttons.add(text="Reduce jitter", action_func=self.process_with_lk, style_tag="workflow")
        self.buttons.add(text="Save annotation as...", action_func=self.save_annotation_as, style_tag="workflow")
        self.buttons.add_separator(style="double")

        # --- Display / trace controls -----------------------------------
        # Image enhancement is driven by the EnhanceWidget sliders
        # (mounted below statevars by _add_enhance_widget). Sliders at
        # min = bypass; no separate Toggle enhance button.
        self.buttons.add(text="Discard unsaved annotations", action_func=self.discard_unsaved_annotations, style_tag="display")
        self.buttons.add_multi(
            dict(text="Trace: line", action_func=(lambda s, ev: s.ann.set_plot_type("line")).__get__(self), style_tag="display"),
            dict(text="Trace: dot",  action_func=(lambda s, ev: s.ann.set_plot_type("dot")).__get__(self), style_tag="display"),
        )
        self.buttons.add_multi(
            dict(text="Freeze plot axes",   action_func=self.freeze_plot_axes, style_tag="display"),
            dict(text="Unfreeze plot axes", action_func=self.unfreeze_plot_axes, style_tag="display"),
        )
        self.buttons.add_separator(style="double")

        # Niche operation; flagged for a future decision -- should this
        # button be replaced by a keyboard-only shortcut to reclaim the
        # vertical slot? Track usage before removing.
        self.buttons.add(text="Replace existing from overlay", action_func=self.copy_existing_annotations_from_overlay, style_tag="niche")
        self.buttons.add(text="Remove layer", action_func=self.remove_current_layer, style_tag="niche")
        self.buttons.add_separator(style="double")

        # --- Utilities + Swap layers -----------------------------------
        # Refresh UI is normally installed by _DUSTrackBase's
        # _add_default_buttons hook; DUSTrack overrides that hook to
        # no-op (see _add_default_buttons below) so this slot keeps the
        # button next to Keyboard shortcuts as a "utility" pair just
        # above Swap layers.
        self.buttons.add(text="Refresh UI", action_func=self.refresh, style_tag="utilities")
        self.buttons.add(text="Keyboard shortcuts", action_func=(lambda s, ev: s.show_key_bindings()).__get__(self), style_tag="utilities")
        self.buttons.add_separator(style="double")

        self.buttons.add(text="Swap annotation layers", action_func=self.swap_active_and_overlay, style_tag="swap")

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

        if self.__class__.__name__ == "DUSTrack":
            plt.show(block=False)
            self.update()
            plt.setp(self._ax_trace_x.get_xticklabels(), visible=False)
            plt.draw()

    def _add_default_buttons(self) -> None:
        """Override the parent's default-button hook to a no-op.

        _DUSTrackBase installs ``Refresh UI`` immediately at
        end of ``__init__``; DUSTrack's rc2 sidebar instead places it
        next to ``Keyboard shortcuts`` as a utility pair just above
        ``Swap layers`` (see button-add block in ``__init__``).
        """
        return

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
            "bg": "#cfdef3", "fg": "#2c3e50",
            "border": "#a8c0dd", "hover": "#bccfea", "pressed": "#a8c0dd",
        },
        "display": {   # pale mint -- cool green, analogous step from blue
            "bg": "#d4ebd4", "fg": "#2c3e50",
            "border": "#aed4ae", "hover": "#c1dfc1", "pressed": "#aed4ae",
        },
        "niche": {     # pale apricot -- warm shift, "use sparingly"
            "bg": "#f5d9c0", "fg": "#2c3e50",
            "border": "#d9b88a", "hover": "#eaca9f", "pressed": "#d9b88a",
        },
        "utilities": {  # pale sand -- neutral warm
            "bg": "#ece6d5", "fg": "#2c3e50",
            "border": "#d4cdb8", "hover": "#e0d9c5", "pressed": "#d4cdb8",
        },
        "swap": {      # pale silver -- matches statevars
            "bg": "#e0e4e8", "fg": "#2c3e50",
            "border": "#c0c5cb", "hover": "#d0d4d9", "pressed": "#c0c5cb",
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
        insert_at = (col.statevars_slot_index + 1) if col.statevars_widget is not None else col.statevars_slot_index
        col.outer_layout.insertWidget(insert_at, widget)
        self._enhance_widget = widget

    def _paint_statevars_widget(self) -> None:
        """Paint the statevars widget bg/fg to match the rc2 sidebar palette.

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
        """Re-evaluate workflow gates once the lazy DLC import resolves.

        Poll-based: starts a 250 ms ``QTimer`` on the Qt main thread
        that watches :func:`_dlc_load_state`; when state transitions
        out of ``"pending"`` / ``"loading"`` the timer fires
        :meth:`_refresh_workflow_button_state` once and stops itself.
        Polling (vs. a cross-thread signal hop from the loader thread)
        keeps every Qt touch on the main thread and mirrors the
        ``_run_with_overlay`` pattern already in this file.

        No-op on the mpl-fallback path (no Qt window) and on the
        ``HAS_DLC=False`` path (Workflow buttons aren't created).
        """
        if not HAS_DLC:
            return
        if _dlc_load_state() in ("done", "missing"):
            # Already resolved (e.g. tests pre-bound state, or the
            # user spent >7 s in the picker on a warm cache). Gates
            # were already in their final state when
            # ``_refresh_workflow_button_state()`` ran above.
            return
        try:
            from qtpy.QtCore import QTimer
        except Exception:  # noqa: BLE001 -- mpl-only / pre-Qt teardown
            return

        qt_window = self._find_qt_window()
        if qt_window is None:
            return

        timer = QTimer(qt_window)
        timer.setInterval(250)

        def _tick():
            if _dlc_load_state() in ("done", "missing"):
                timer.stop()
                try:
                    self._refresh_workflow_button_state()
                except Exception:  # noqa: BLE001 -- defensive; never break the event loop.
                    traceback.print_exc()

        timer.timeout.connect(_tick)
        timer.start()
        # Keep a reference so Qt doesn't garbage-collect the timer
        # mid-poll. Same convention as the overlay-worker timer above.
        self._dlc_load_gate_timer = timer

    def _refresh_workflow_button_state(self) -> None:
        """Enable / disable Workflow-group buttons based on session state.

        Three gates run here:

        - **Create DLC Project** is disabled when the session already
          sits inside a DLC project (either ``self._dlcproject`` is
          set, or the video path resolves into a project tree). DLC's
          ``copy_videos`` would otherwise scaffold a nested project at
          ``<project>/videos/`` whose downstream paths nothing handles.
        - **Train DLC model** is disabled when ``self._dlcproject`` is
          None. Replaces the click-time ``ValueError("DLCProject not
          created. ...")`` with a greyed-out button + tooltip.
        - **Apply manual corrections** is disabled when there's no
          overlay layer set, or when the active layer is already the
          corrections output. Mirrors the two
          :class:`ValueError` paths in :meth:`apply_manual_corrections`.

        ``Reduce jitter`` is intentionally *not* gated here: its real
        precondition is "every frame in the active layer is fully
        annotated", which is a data property rather than a name-pattern
        property. The cheap name-based proxy
        (:func:`_is_dense_layer_name`) is correct for rendering style
        but would false-disable a fully-annotated manual layer; the
        gate is deferred until a cheap data-side check exists. See
        :func:`lk_moving_average_filter`'s sparse-labels guard for the
        canonical run-time check.

        Qt-only: walks ``self.buttons`` and writes ``setEnabled`` +
        ``setToolTip`` on each Button's ``_qt_btn`` attribute. The
        helper no-ops on any button whose Qt handle is missing (the
        legacy mpl-fallback path; no longer supported as a first-class
        deployment, kept working for users on pinned older versions).
        """
        if not HAS_DLC:
            # The Workflow group's DLC buttons aren't added when
            # deeplabcut is missing; nothing to gate.
            return
        gates = self._evaluate_workflow_gates()
        for label, (enabled, tooltip) in gates.items():
            if label not in self.buttons:
                continue
            btn = self.buttons[label]
            qt_btn = getattr(btn, "_qt_btn", None)
            if qt_btn is None:
                continue
            qt_btn.setEnabled(enabled)
            qt_btn.setToolTip(tooltip)

    def _evaluate_workflow_gates(self) -> dict:
        """Compute ``{button_label: (enabled, tooltip)}`` for the gated buttons.

        Pulled out of :meth:`_refresh_workflow_button_state` so the
        gate semantics can be unit-tested without standing up a Qt
        window. Reads only ``self._dlcproject`` /
        ``self._current_overlay`` / ``self.ann`` / ``self.fname`` --
        the same state the click handlers themselves consult.
        """
        gates: dict = {}

        # --- Create DLC Project --------------------------------------
        proj_root = _session_inside_dlc_project(self)
        dlc_state = _dlc_load_state()
        if proj_root is not None:
            # "Inside a project" wins over "still loading" -- the click
            # would refuse on that ground first.
            gates["Create DLC Project"] = (
                False,
                f"Already inside DLC project {proj_root.name!r} — "
                "use Train DLC model to extend it.",
            )
        elif dlc_state in ("pending", "loading"):
            # The bg preload (``_ensure_dlc_loaded_async`` fired from
            # ``dustrack.open()``) hasn't finished yet. Subtle "we're
            # not ready" signal via greyed-out button + tooltip; flips
            # to enabled when ``register_dlc_load_callback`` fires the
            # post-load gate refresh in ``__init__``.
            gates["Create DLC Project"] = (
                False,
                "Loading DeepLabCut… (this button enables once the "
                "import completes -- typically a few seconds after "
                "DUSTrack launches).",
            )
        elif dlc_state == "missing":
            # ``find_spec`` said yes but the import raised. Edge case
            # (broken torch / dependency conflict / partial DLC
            # install); ``HAS_DLC=False`` would have skipped button
            # creation entirely.
            gates["Create DLC Project"] = (
                False,
                "DeepLabCut failed to load. Check the launching "
                "terminal for the import error.",
            )
        else:
            gates["Create DLC Project"] = (True, "")

        # --- Train DLC model -----------------------------------------
        if self._dlcproject is None:
            gates["Train DLC model"] = (
                False,
                "Create a DLC project first.",
            )
        else:
            gates["Train DLC model"] = (True, "")

        # --- Apply manual corrections --------------------------------
        ann = getattr(self, "ann", None)
        ann_name = getattr(ann, "name", None) if ann is not None else None
        if self._current_overlay is None:
            gates["Apply manual corrections"] = (
                False,
                "Set an overlay layer (typically a 'dlc_*' trace) "
                "first.",
            )
        elif ann_name == self.CORRECTIONS_LAYER_NAME:
            gates["Apply manual corrections"] = (
                False,
                "Switch the active layer back to your manual "
                f"annotations — {self.CORRECTIONS_LAYER_NAME!r} is the "
                "output, not the input.",
            )
        else:
            gates["Apply manual corrections"] = (True, "")

        return gates

    def _apply_dark_theme(self):
        """Apply dark theme to the GUI for better ultrasound visibility."""
        bg_color = '#1a1a1a'
        ax_color = '#2a2a2a'
        text_color = 'white'

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


    def create_dlc_project(self, event=None, name=None, path=None,
                           experimenter=_config.EXPERIMENTER,
                           seed_bundle_path=None) -> DLCProject:
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

        Returns:
            DLCProject: The newly created project instance on the sync
            path. ``None`` on the Qt async path -- read
            ``self._dlcproject`` after the Done button is clicked.

        Note:
            Project names must contain an underscore for proper DLC configuration handling.
        """
        if not HAS_DLC:
            raise ImportError('deeplabcut is not installed. Cannot create DLC project.')

        qt_window = self._find_qt_window()
        active_layer_empty = not any(self.ann.data.values())

        # Qt path with empty active layer + no explicit bundle: open
        # the seeding modal sequence. The user can still cancel out
        # at every step (intent -> folder pick -> confirm). Cancel
        # leaves the UI intact (returns None).
        if (
            active_layer_empty
            and seed_bundle_path is None
            and qt_window is not None
        ):
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
                    iteration_num=0, create_video=False,
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
            raise ImportError('deeplabcut is not installed. Cannot process DLC project.')
        if self._dlcproject is None:
            raise ValueError('DLCProject not created. Use create_dlc_project() to create it.')

        qt_window = self._find_qt_window()
        if qt_window is None:
            # Non-Qt fallback: no Training options modal possible, so
            # route through ``DLCProject.process()`` (auto-infer + sane
            # defaults). The Qt path uses ``train_iteration`` below with
            # explicit args supplied by the modal.
            kwargs.setdefault('create_video', False)
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
            if not self._prompt_unified_pre_flight(qt_window, issues):
                return self  # user cancelled -- UI left intact
            self._apply_pre_flight_remediations(issues)
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

    @staticmethod
    def _scan_incomplete_frames(
        data: dict, target_labels: list[str] | None = None,
    ) -> dict:
        """Find frames missing one or more required bodyparts in an
        annotation ``data`` dict (``{label: {frame: [x, y]}}``).

        Two modes:

        - ``target_labels=None`` (legacy, project-unaware): "required"
          = labels that have at least one annotation. Empty labels
          are treated as UI placeholders so they don't fail every
          frame. This is the right behavior for a session without a
          DLC project, where the user's declared label set may be the
          default ``[" 0"]`` bootstrap with no project bodyparts to
          anchor against.
        - ``target_labels=<list>`` (project-aware): "required" =
          exactly that list. Used when a ``DLCProject`` exists and
          its ``config['bodyparts']`` are the load-bearing label
          set -- training fails if a frame is missing any of them,
          regardless of whether the user has touched that label
          anywhere in the layer yet. Closes the case where a
          seeded project has bodyparts ``["point0", "point1"]``
          but the user annotated only the ``"0"`` label, and
          pre-flight wrongly reported the layer as complete.

        Returns ``{frame: [missing_label, ...]}`` for incomplete
        frames, frame-sorted with missing-labels lists in the same
        order as the required-label list. Empty dict iff every
        annotated frame has every required label.

        Pure data-in / data-out; testable from synthetic dicts.
        """
        if target_labels is None:
            required = [L for L, frames in data.items() if frames]
        else:
            required = list(target_labels)
        if not required:
            return {}
        all_frames: set = set()
        for L, frames in data.items():
            all_frames.update(frames.keys())
        incomplete: dict = {}
        for frame in sorted(all_frames):
            missing = [L for L in required if frame not in data.get(L, {})]
            if missing:
                incomplete[frame] = missing
        return incomplete

    @staticmethod
    def _build_dropped_incomplete_payload(data: dict, incomplete_frames: dict) -> dict:
        """Build the JSON payload for the dropped-incomplete sidecar.

        Each entry is ``{label: [x, y]}`` for the labels that *were*
        present at the dropped frame (the missing ones are the
        incompleteness). Frame keys are stringified for JSON compat.
        """
        payload: dict = {}
        for frame in incomplete_frames:
            present = {}
            for L, frames in data.items():
                if frame in frames:
                    present[L] = [float(x) for x in frames[frame]]
            payload[str(frame)] = present
        return payload

    @staticmethod
    def _build_dropped_incomplete_sidecar_name(ann_fname, ts: str) -> str:
        """``<fstem>.dustrack-dropped-incomplete-<ts>`` in the
        annotation's directory.

        Intentionally avoids `.json` so DUSTrack's annotation-discovery
        glob (``{video_stem}*_annotations*.json`` in
        :meth:`DLCProject.extract_frames`) does not re-ingest the
        sidecar on a subsequent training run.
        """
        p = Path(ann_fname)
        return str(p.parent / f"{p.stem}.dustrack-dropped-incomplete-{ts}")

    @staticmethod
    def _format_incomplete_breakdown(incomplete_frames: dict, max_rows: int = 200) -> str:
        """Multi-line per-bodypart breakdown for the pre-flight modal's
        detailed-text panel. Truncates very wide reports.
        """
        rows = []
        total = len(incomplete_frames)
        for i, (frame, missing) in enumerate(sorted(incomplete_frames.items())):
            if i >= max_rows:
                rows.append(f"... ({total - max_rows} more frames)")
                break
            rows.append(f"Frame {frame}: missing {', '.join(missing)}")
        return "\n".join(rows)

    def _save_dropped_incomplete_sidecar(self, ann, incomplete_frames: dict):
        """Persist the dropped-frame contents next to the given layer.

        Returns the sidecar path on success, ``None`` if the layer
        has no on-disk filename (in-memory only).
        """
        import datetime
        import json
        if ann.fname is None:
            return None
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        sidecar = self._build_dropped_incomplete_sidecar_name(ann.fname, ts)
        payload = self._build_dropped_incomplete_payload(ann.data, incomplete_frames)
        # Path.write_text rather than builtin open() because the
        # module-level open() (workflow entry point) shadows
        # builtins.open inside this file -- see _load_layer_disk_data.
        Path(sidecar).write_text(json.dumps(payload, indent=2))
        return sidecar

    @staticmethod
    def _is_manual_layer_name(
        ann_name: str,
        special_names: tuple = ("dlccorr", "buffer"),
    ) -> bool:
        """Name-only predicate for "is this a manual annotation layer?".

        Pure string check on the layer name -- excludes ``dlccorr``
        (terminal output of apply_manual_corrections), ``buffer``
        (workspace scratch), and any layer whose name starts with
        ``"dlc"`` (DLC trace + process_with_lk LK outputs). Symmetric
        with the name-side of :meth:`_is_manual_annotation_layer`,
        which adds an on-disk file-pattern check on top.

        Lives separately so callers that care about incomplete-frame
        scanning -- which only needs the in-memory ``ann.data`` --
        can include layers that aren't yet saved to disk (``ann.fname
        is None``). The Train pre-flight uses this for inclusion and
        then guards the disk-diff portion on ``ann.fname`` being set;
        save-on-close uses the stricter file-aware predicate because
        a layer with no disk file has nothing to diff against.
        """
        if ann_name in special_names:
            return False
        if ann_name.startswith("dlc"):
            return False
        return True

    @staticmethod
    def _is_manual_annotation_layer(
        video_fname,
        ann_fname,
        ann_name: str,
        special_names: tuple = ("dlccorr", "buffer"),
    ) -> bool:
        """Identify a manual annotation ``.json`` layer that feeds
        :meth:`DLCProject.extract_frames`.

        Rule: ``.json`` file alongside the video, matching the
        ``<video_stem>_annotations*.json`` pattern, AND the layer name
        passes :meth:`_is_manual_layer_name`. Excludes ``dlccorr`` /
        ``buffer`` / ``dlc*`` by name.

        File-based detection -- doesn't rely on the
        ``iteration-N`` naming convention, so a layer the user
        renamed to ``iter1`` or seeded with experimenter initials
        (e.g. ``pn``) is still picked up.
        """
        if ann_fname is None or video_fname is None:
            return False
        fname_path = Path(ann_fname)
        if fname_path.suffix != ".json":
            return False
        video_path = Path(video_fname)
        if fname_path.parent != video_path.parent:
            return False
        video_stem = video_path.stem
        stem = fname_path.stem
        if stem != f"{video_stem}_annotations" and not stem.startswith(
            f"{video_stem}_annotations_"
        ):
            return False
        return DUSTrack._is_manual_layer_name(ann_name, special_names)

    @staticmethod
    def _normalize_layer_data(data: dict) -> dict:
        """Canonical form for diff comparison: int frame keys, float
        ``[x, y]`` values, empty labels filtered.

        Empty-label filtering exists for *diff* symmetry: a label
        with no frames contributes no entries to added / removed /
        modified regardless of whether it's present on one side
        only. With dnav 1.4.0rc2's first-class-label schema, both
        on-disk JSON and in-memory data may legitimately carry
        ``"label": {}`` entries (whereas pre-rc2,
        :meth:`VideoAnnotation.save` pruned them on the way out); the
        diff still works correctly because both inputs are filtered
        the same way here.
        """
        out: dict = {}
        for label, frames in data.items():
            if not frames:
                continue
            out[label] = {
                int(frame): [float(x) for x in xy]
                for frame, xy in frames.items()
            }
        return out

    @staticmethod
    def _load_layer_disk_data(ann_fname) -> dict:
        """Read the on-disk JSON for a layer and return its data in
        canonical form. Empty dict if the file does not exist or
        cannot be parsed (treated as "fully unsaved").

        Uses ``Path.read_text`` rather than builtin ``open()`` because
        the module defines a top-level ``open`` (the workflow entry
        point) that shadows ``builtins.open`` inside this file -- same
        convention as ``DLCProject._read_trackermap``.
        """
        import json
        if ann_fname is None:
            return {}
        p = Path(ann_fname)
        if not p.exists():
            return {}
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return DUSTrack._normalize_layer_data(raw)

    @staticmethod
    def _diff_ann_vs_disk(mem_data: dict, disk_data: dict) -> dict:
        """Compare two canonical data dicts. Returns
        ``{"added": [...], "removed": [...], "modified": [...]}``
        where each list contains ``(label, frame)`` tuples in label-
        then frame-sorted order.

        Both inputs are assumed normalized via
        :meth:`_normalize_layer_data`.
        """
        added: list = []
        removed: list = []
        modified: list = []
        all_labels = set(mem_data) | set(disk_data)
        for label in sorted(all_labels):
            mem_frames = mem_data.get(label, {})
            disk_frames = disk_data.get(label, {})
            for frame in sorted(set(mem_frames) - set(disk_frames)):
                added.append((label, frame))
            for frame in sorted(set(disk_frames) - set(mem_frames)):
                removed.append((label, frame))
            for frame in sorted(set(mem_frames) & set(disk_frames)):
                if mem_frames[frame] != disk_frames[frame]:
                    modified.append((label, frame))
        return {"added": added, "removed": removed, "modified": modified}

    def _scan_unsaved_layers(self) -> dict:
        """Across every manual annotation layer, return
        ``{layer_name: diff}`` for layers whose in-memory data differs
        from disk. Sibling of :meth:`_scan_unsaved_and_incomplete`
        scoped to the data-loss concern only -- the close-event guard
        does not care about incomplete-frame quality (that surfaces
        next time the user trains).
        """
        unsaved: dict = {}
        for ann in self.annotations:
            if not self._is_manual_annotation_layer(
                self.fname, ann.fname, ann.name
            ):
                continue
            mem_data = self._normalize_layer_data(ann.data)
            disk_data = self._load_layer_disk_data(ann.fname)
            diff = self._diff_ann_vs_disk(mem_data, disk_data)
            if any(diff.values()):
                unsaved[ann.name] = diff
        return unsaved

    @staticmethod
    def _format_unsaved_summary(unsaved: dict) -> str:
        """Per-layer +added / -removed / ~modified counts for the
        save-on-close modal's informative text.
        """
        lines = []
        for layer_name, diff in unsaved.items():
            a = len(diff.get("added", []))
            r = len(diff.get("removed", []))
            m = len(diff.get("modified", []))
            pieces = []
            if a:
                pieces.append(f"+{a} added")
            if r:
                pieces.append(f"-{r} removed")
            if m:
                pieces.append(f"~{m} modified")
            lines.append(f"  {layer_name!r}: " + ", ".join(pieces))
        return "\n".join(lines)

    def _scan_unsaved_and_incomplete(self) -> dict:
        """Across every manual annotation layer in the session, find
        in-memory-vs-disk diffs AND/OR incomplete frames. Returns
        ``{layer_name: {"diff": ..., "incomplete": ...}}`` for
        layers with at least one issue; layers with neither are
        omitted.

        Inclusion is name-based (see :meth:`_is_manual_layer_name`)
        so layers that haven't been saved yet -- ``ann.fname is None``,
        the state after a user opens a fresh video, annotates a
        partial layer, and clicks Train without saving first -- are
        still scanned for incompleteness. The disk-diff portion is
        guarded on ``ann.fname`` being set AND matching the
        ``<video_stem>_annotations*.json`` pattern; without a disk
        file there's nothing to diff against. Pre-2026-05-21 the
        inclusion check required a file match too, which silently
        skipped unsaved-but-incomplete layers on first-time training.

        Project-aware incomplete scan (1.2.0a2): when ``self._dlcproject``
        is set, derives the required-label set from
        ``config['bodyparts']`` (mapped through
        :func:`_dlc_bodyparts_to_layer_labels`) and hands it to
        :meth:`_scan_incomplete_frames` as ``target_labels``. Catches
        the post-seeding case where the user has annotated, say, only
        the ``"0"`` label but the project bodyparts demand both
        ``"0"`` and ``"1"`` -- the legacy "active labels = labels
        with any annotation" rule wrongly reported those frames as
        complete.
        """
        target_labels: list[str] | None = None
        if self._dlcproject is not None:
            bodyparts = self._dlcproject.config.get("bodyparts") or []
            if bodyparts:
                target_labels = _dlc_bodyparts_to_layer_labels(bodyparts)

        issues: dict = {}
        for ann in self.annotations:
            if not self._is_manual_layer_name(ann.name):
                continue
            incomplete = self._scan_incomplete_frames(
                ann.data, target_labels=target_labels,
            )
            diff = {"added": [], "removed": [], "modified": []}
            if self._is_manual_annotation_layer(
                self.fname, ann.fname, ann.name
            ):
                mem_data = self._normalize_layer_data(ann.data)
                disk_data = self._load_layer_disk_data(ann.fname)
                diff = self._diff_ann_vs_disk(mem_data, disk_data)
            if any(diff.values()) or incomplete:
                issues[ann.name] = {"diff": diff, "incomplete": incomplete}
        return issues

    @staticmethod
    def _format_pre_flight_summary(
        issues: dict, max_incomplete_examples: int = 3
    ) -> str:
        """Per-layer breakdown for the unified pre-flight modal's
        detailed-text panel.
        """
        blocks = []
        for layer_name, info in issues.items():
            lines = [f"Layer {layer_name!r}:"]
            diff = info.get("diff", {})
            a = len(diff.get("added", []))
            r = len(diff.get("removed", []))
            m = len(diff.get("modified", []))
            if a or r or m:
                pieces = []
                if a:
                    pieces.append(f"+{a} added")
                if r:
                    pieces.append(f"-{r} removed")
                if m:
                    pieces.append(f"~{m} modified")
                lines.append("  Unsaved changes: " + ", ".join(pieces))
            else:
                lines.append("  (no unsaved changes)")
            incomplete = info.get("incomplete", {})
            if incomplete:
                n = len(incomplete)
                lines.append(f"  Incomplete frames: {n}")
                for i, (frame, missing) in enumerate(sorted(incomplete.items())):
                    if i >= max_incomplete_examples:
                        lines.append(
                            f"    ... ({n - max_incomplete_examples} more)"
                        )
                        break
                    lines.append(
                        f"    frame {frame}: missing {', '.join(missing)}"
                    )
            else:
                lines.append("  (no incomplete frames)")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def _prompt_training_options(self, qt_window):
        """Show the Training options modal and return kwargs ready to
        splat into :meth:`DLCProject.train_iteration`.

        Builds the initial state via
        :func:`_default_training_options` from the live ``DLCProject``,
        runs :class:`TrainingOptionsDialog` synchronously, and
        translates the user's choices via
        :func:`_training_options_to_train_iteration_kwargs`.

        Returns:
            dict | None: kwargs for ``train_iteration``, or ``None``
            if the user clicked Cancel (caller returns without
            kicking off training).
        """
        TrainingOptionsDialog = _make_training_options_class()
        initial_state = _default_training_options(self._dlcproject)
        options = TrainingOptionsDialog(
            qt_window, initial_state=initial_state,
        ).exec_()
        if options is None:
            return None
        return _training_options_to_train_iteration_kwargs(options)

    def _prompt_seed_bundle(self, qt_window) -> Optional[str]:
        """Multi-step modal sequence that fires when ``Create DLC Project``
        is clicked with an empty active manual layer.

        Two entry points depending on whether a seed-bundles root has
        been remembered:

        - **Root set + non-empty**: opens a list-picker
          (:class:`SeedBundlePickerDialog`) showing every valid bundle
          under the root with its name + bodyparts + description.
          Quick-select; no file-dialog navigation required.
        - **No root set / empty root**: opens the legacy
          :class:`ConfirmOverlay` -> ``QFileDialog`` -> confirm path.
          After a successful pick, offers to remember the picked
          bundle's parent as the bundles root so next session uses
          the picker.

        Returns the validated bundle folder path on Accept, or ``None``
        on any Cancel / invalid bundle path. Caller (``create_dlc_project``)
        treats ``None`` as "user bailed -- leave the UI alone".
        """
        # Loop so the picker's "Change bundles root" action can
        # re-open the dialog against the new root, and "Browse
        # elsewhere" can fall through to the file-dialog branch.
        while True:
            root = get_seed_bundles_root()
            if root is not None and root.is_dir():
                bundles = list_seed_bundles(root)
            else:
                bundles = []

            if bundles:
                action = self._pick_from_seed_bundles(qt_window, root, bundles)
                if action is None:
                    return None
                kind = action[0]
                if kind == "use":
                    info = action[1]
                    bundle_path = str(info["path"])
                    if self._confirm_seed_bundle(qt_window, bundle_path, info):
                        return bundle_path
                    return None
                if kind == "set_root":
                    set_seed_bundles_root(action[1])
                    continue  # re-list against the new root
                if kind == "browse":
                    # Fall through to legacy Browse flow below, but
                    # don't loop -- a Browse pick should either
                    # accept and return, or cancel out.
                    pass

            # Legacy flow: explain + Browse + validate + confirm.
            picked = self._browse_for_seed_bundle(qt_window)
            if picked is None:
                return None
            # First-time-Browse polite ask: remember the parent as
            # the root so the picker takes over next session. Skip
            # if a root is already configured (user already declined
            # or has a different setup in mind).
            if get_seed_bundles_root() is None:
                self._maybe_remember_seed_bundles_root(qt_window, picked)
            return picked

    def _pick_from_seed_bundles(self, qt_window, root, bundles):
        """Drive :class:`SeedBundlePickerDialog`. Returns the
        dialog's raw result tuple (or ``None`` on cancel)."""
        PickerDialog = _make_seed_bundle_picker_class()
        return PickerDialog(qt_window, root=root, bundles=bundles).exec_()

    def _browse_for_seed_bundle(self, qt_window) -> Optional[str]:
        """Legacy seed-bundle flow used when no bundles root is set
        (or the picker user clicked Browse elsewhere): intent
        overlay -> ``QFileDialog`` -> validate -> confirm. Returns
        the validated bundle path on accept, ``None`` on any cancel
        or invalid bundle."""
        from qtpy.QtWidgets import QFileDialog

        ConfirmOverlay = _make_confirm_overlay_class()

        # Only show the intent overlay when there's no remembered
        # root -- if the user got here via "Browse elsewhere" from
        # the picker, they already understand the situation.
        if get_seed_bundles_root() is None:
            result = ConfirmOverlay(
                qt_window,
                title="No annotations in active layer",
                message=(
                    f"Active layer {self.ann.name!r} has no labels. "
                    "To create a DLC project from this session, "
                    "either annotate frames manually first, or seed "
                    "iteration-0 from a pre-trained snapshot bundle "
                    "(a folder containing snapshot-*.pt + "
                    "pytorch_config.yaml + pose_cfg.yaml).\n\n"
                    "Inference from the bundled snapshot will run on "
                    "the current video and load as a dense reference "
                    "overlay; your manual refinements then become "
                    "iteration-1."
                ),
                buttons=[
                    ("Browse for seed bundle…", "primary"),
                    ("Cancel", "neutral"),
                ],
                default="Cancel",
                severity="warning",
            ).exec_()
            if result != "Browse for seed bundle…":
                return None

        bundle_dir = QFileDialog.getExistingDirectory(
            qt_window,
            "Choose seed bundle folder",
            "",
            QFileDialog.ShowDirsOnly,
        )
        if not bundle_dir:
            return None

        try:
            info = inspect_seed_bundle(bundle_dir)
        except (FileNotFoundError, ValueError) as exc:
            ConfirmOverlay(
                qt_window,
                title="Invalid seed bundle",
                message=(
                    f"The selected folder is not a usable seed bundle:\n\n"
                    f"{exc}\n\n"
                    "Re-click 'Create DLC Project' to try again."
                ),
                buttons=[("OK", "neutral")],
                default="OK",
                severity="error",
            ).exec_()
            return None

        if self._confirm_seed_bundle(qt_window, bundle_dir, info):
            return bundle_dir
        return None

    def _confirm_seed_bundle(self, qt_window, bundle_path, info) -> bool:
        """Final confirm-with-detected-info overlay shared by the
        picker and Browse paths. Returns True iff the user clicked
        ``Create and seed``.
        """
        ConfirmOverlay = _make_confirm_overlay_class()
        description = info.get("description") or "(no description)"
        result = ConfirmOverlay(
            qt_window,
            title="Confirm seed bundle",
            message=(
                f"Bundle: {bundle_path}\n"
                f"Snapshot: {info['snapshot'].name}\n"
                f"Bodyparts ({len(info['bodyparts'])}): {info['bodyparts']}\n"
                f"Net type: {info.get('net_type') or '(unset)'}\n"
                f"Description: {description}\n\n"
                "Create the project, install this snapshot as iteration-0, "
                "and run inference on the current video?"
            ),
            buttons=[
                ("Create and seed", "primary"),
                ("Cancel", "neutral"),
            ],
            default="Cancel",
            severity="info",
        ).exec_()
        return result == "Create and seed"

    def _maybe_remember_seed_bundles_root(self, qt_window, bundle_path) -> None:
        """After the first successful Browse pick, ask the user if
        they want to remember the bundle's parent folder as the
        seed-bundles root so the next session opens the list-picker
        directly. No-op if they say no (the next Browse will ask
        again)."""
        ConfirmOverlay = _make_confirm_overlay_class()
        parent = Path(bundle_path).parent
        result = ConfirmOverlay(
            qt_window,
            title="Remember bundles location?",
            message=(
                f"Use this folder as your seed-bundles root?\n\n"
                f"{parent}\n\n"
                "Next time you click Create DLC Project on an empty "
                "layer, DUSTrack will list every bundle in this "
                "folder so you don't have to browse."
            ),
            buttons=[
                ("Remember it", "primary"),
                ("Not now", "neutral"),
            ],
            default="Remember it",
            severity="info",
        ).exec_()
        if result == "Remember it":
            set_seed_bundles_root(parent)

    def _has_trainable_labels(self) -> bool:
        """True if the project has *any* source of labels training
        could consume: at least one non-empty manual annotation layer
        in the session, or at least one ``.h5`` under the project's
        ``labeled-data/`` folder (already-extracted labels from
        prior iterations).

        Pure predicate -- no side effects. Used by
        :meth:`process_dlc_project` to decide between hard-blocking
        the Train DLC click and falling through to the
        "Continue training without new data?" confirm.
        """
        for ann in self.annotations:
            if not self._is_manual_layer_name(ann.name):
                continue
            if any(ann.data.values()):
                return True
        if self._dlcproject is not None:
            labels_dir = Path(self._dlcproject.paths["labels"])
            if labels_dir.is_dir():
                for _ in labels_dir.rglob("*.h5"):
                    return True
        return False

    def _prompt_no_trainable_labels(self, qt_window) -> None:
        """Hard-block overlay for the Train DLC path when the active
        manual layer is empty AND no other source of labels exists in
        the project (no other non-empty manual layer, no
        ``labeled-data/*.h5``). Distinct from
        :meth:`_prompt_empty_layer_train_confirm` -- there's nothing
        to confirm, the user has to add labels before training can do
        anything.

        Typical trigger: freshly-seeded project, user clicks Train
        before annotating any iteration-1 frames.
        """
        ConfirmOverlay = _make_confirm_overlay_class()
        ConfirmOverlay(
            qt_window,
            title="No labels to train on",
            message=(
                f"Active layer {self.ann.name!r} has no labels, and no "
                "other annotation layer or 'labeled-data/' file in this "
                "project has any either. Training would have nothing "
                "to consume.\n\n"
                "Annotate some frames in the active layer first, or "
                "use 'Apply manual corrections' to convert a DLC "
                "prediction trace into a manual annotation layer."
            ),
            buttons=[("OK", "neutral")],
            default="OK",
            severity="error",
        ).exec_()

    def _prompt_empty_layer_train_confirm(self, qt_window) -> bool:
        """Modal that fires when Train DLC is clicked with an empty
        active manual layer. Returns True iff the user confirmed
        ``Continue training``.

        "Empty active layer" means no label has any frames -- the
        check at the call site is ``not any(self.ann.data.values())``,
        the same predicate used by ``_rewire_to_in_project_paths``
        to decide whether a layer needs an on-disk save. The empty
        layer itself is *not* saved by this path -- the pre-flight
        scan downstream only acts on layers with diffs or
        incompleteness, and an empty layer has neither.

        User intent in this state: "train for more iterations without
        new labels." Training will reuse whatever labels already
        exist in ``labeled-data/``; if none do (e.g. a freshly-seeded
        iteration-1), DLC will fail downstream with its own error.
        """
        ConfirmOverlay = _make_confirm_overlay_class()
        body = (
            f"Active layer {self.ann.name!r} has no labels.\n\n"
            "Training will reuse the labels already in "
            "'labeled-data/' from previous iterations. The empty "
            "active layer will not be saved.\n\n"
            "Continue without adding new labels?"
        )
        result = ConfirmOverlay(
            qt_window,
            title="No annotations in active layer",
            message=body,
            buttons=[
                ("Continue training", "primary"),
                ("Cancel", "neutral"),
            ],
            default="Cancel",
            severity="warning",
        ).exec_()
        return result == "Continue training"

    def _prompt_unified_pre_flight(self, qt_window, issues: dict) -> bool:
        """Single modal for the combined save-state + incompleteness
        pre-flight. Returns True iff the user picked
        *Save and clean*.

        Routes through :class:`ConfirmOverlay` (rc2) so the modal
        shares visual vocabulary with the new ``Discard unsaved`` /
        ``Remove layer`` confirms; pre-rc2 used ``QMessageBox``. The
        per-layer breakdown is shown inline rather than behind a
        collapsed "Show Details..." toggle -- the breakdown is the
        substance the user needs to decide on, not optional extra.
        """
        ConfirmOverlay = _make_confirm_overlay_class()
        n = len(issues)
        header = (
            f"{n} manual annotation layer{'s' if n != 1 else ''} "
            f"{'have' if n != 1 else 'has'} unsaved changes and/or "
            "incomplete frames."
        )
        breakdown = self._format_pre_flight_summary(issues)
        body = (
            f"{header}\n\n"
            f"{breakdown}\n\n"
            "Save and clean will:\n"
            " - save in-memory edits to disk for the listed layer(s),\n"
            " - drop frames missing one or more bodyparts (per-layer "
            "recovery sidecars written next to each annotation file),\n"
            " - then start training.\n\n"
            "Cancel returns to the UI without changes."
        )
        result = ConfirmOverlay(
            qt_window,
            title="Pre-flight issues",
            message=body,
            buttons=[
                ("Save and clean", "primary"),
                ("Cancel", "neutral"),
            ],
            default="Cancel",
            severity="warning",
        ).exec_()
        return result == "Save and clean"

    def _apply_pre_flight_remediations(self, issues: dict) -> None:
        """For each layer with issues, drop incomplete frames (with
        recovery sidecar) and save the (possibly trimmed) layer.

        Layers whose ``ann.fname`` is ``None`` (in-session unsaved
        layers, the first-time-training case) get a canonical fname
        derived from the video stem + layer name before save:
        ``<video_stem>_annotations_<layer_name>.json``. The recovery
        sidecar needs the same path resolved.
        """
        for layer_name, info in issues.items():
            ann = self.annotations[layer_name]
            if ann.fname is None:
                ann.fname = str(make_annotation_file_name(
                    Path(self.fname), annotation_suffix=ann.name
                ))
            incomplete = info.get("incomplete") or {}
            if incomplete:
                self._save_dropped_incomplete_sidecar(ann, incomplete)
                # Drop the incomplete frames directly using the scan's
                # output. The pre-fix pair (remove_empty_labels +
                # keep_overlapping_frames) silently failed in the
                # project-aware case: a required-but-empty label (the
                # user touched only "0" in a ["point0", "point1"]
                # project) got dropped by remove_empty_labels, and
                # keep_overlapping_frames then preserved every
                # incomplete frame because the now-single-label
                # schema trivially "overlapped" with itself. Routing
                # mutations through ``ann.remove(label, frame)`` keeps
                # the revision counter consistent (see
                # ``feedback_revision_counter_invalidation_pattern``)
                # and works in both project-aware and legacy modes.
                for frame in incomplete:
                    for label in list(ann.data.keys()):
                        if frame in ann.data[label]:
                            ann.remove(label, frame)
            ann.save()
        self.update()

    def _prompt_save_on_close(self, qt_window, unsaved) -> str:
        """Modal triggered by the save-on-close guard. Returns the user's
        choice as one of ``"save"`` / ``"discard"`` / ``"cancel"``.

        ``unsaved`` is either the legacy single-bundle shape
        (``{layer_name: diff}``) or, in 1.2.0a3+ multi-video sessions,
        the per-bundle shape ``{video_index: {"fname": Path,
        "layers": {layer_name: diff}}}``. The modal renders each
        bundle's layers in a separate block so users can see which
        video each diff belongs to.

        *Save* writes every layer with diffs and lets the window close;
        *Discard* lets the window close without writing; *Cancel* keeps
        the window open. ``Cancel`` is the default button so that
        accidental Enter / Esc do not silently lose data. Routes
        through :class:`ConfirmOverlay` (rc2); pre-rc2 used
        ``QMessageBox``.
        """
        ConfirmOverlay = _make_confirm_overlay_class()
        if unsaved and "fname" in next(iter(unsaved.values()), {}):
            # Multi-bundle shape -- one block per video.
            blocks = []
            total_layers = 0
            for video_index, info in unsaved.items():
                layers = info["layers"]
                if not layers:
                    continue
                total_layers += len(layers)
                blocks.append(
                    f"  {Path(info['fname']).name} (video {video_index + 1}):\n"
                    f"{self._format_unsaved_summary(layers)}"
                )
            breakdown = "\n\n".join(blocks)
            n_videos = sum(1 for info in unsaved.values() if info.get("layers"))
            header = (
                f"{total_layers} annotation layer"
                f"{'s' if total_layers != 1 else ''} across "
                f"{n_videos} video{'s' if n_videos != 1 else ''} "
                f"{'have' if total_layers != 1 else 'has'} unsaved changes."
            )
        else:
            # Legacy single-bundle shape.
            n = len(unsaved)
            header = (
                f"{n} annotation layer{'s' if n != 1 else ''} "
                f"{'have' if n != 1 else 'has'} unsaved changes."
            )
            breakdown = self._format_unsaved_summary(unsaved)
        body = (
            f"{header}\n\n"
            f"{breakdown}\n\n"
            "Save all writes the in-memory edits to disk before closing.\n"
            "Discard closes without writing -- changes are lost.\n"
            "Cancel keeps the window open."
        )
        result = ConfirmOverlay(
            qt_window,
            title="Unsaved annotations",
            message=body,
            buttons=[
                ("Save all", "primary"),
                ("Discard", "destructive"),
                ("Cancel", "neutral"),
            ],
            default="Cancel",
            severity="destructive",
        ).exec_()
        if result == "Save all":
            return "save"
        if result == "Discard":
            return "discard"
        return "cancel"

    def _save_unsaved_layers(self, unsaved) -> None:
        """Persist every layer with diffs. Called from the save-on-close
        guard when the user picks *Save all*.

        Accepts the legacy single-bundle shape
        (``{layer_name: diff}`` -- saves against ``self.annotations``)
        or the multi-bundle shape (``{video_index: {"fname": Path,
        "layers": {layer_name: diff}}}`` -- saves against each
        bundle's own ``annotations`` container).
        """
        if unsaved and "fname" in next(iter(unsaved.values()), {}):
            for video_index, info in unsaved.items():
                bundle = self._bundles[video_index]
                if bundle.annotations is None:
                    continue
                for layer_name in info["layers"]:
                    if layer_name in bundle.annotations.names:
                        bundle.annotations[layer_name].save()
            return
        for layer_name in unsaved:
            ann = self.annotations[layer_name]
            ann.save()

    def _scan_unsaved_layers_all_bundles(self) -> dict:
        """Sweep every ``ready`` bundle for in-memory-vs-disk diffs.

        Returns ``{video_index: {"fname": Path, "layers":
        {layer_name: diff}}}`` for bundles with at least one unsaved
        layer; empty when nothing is dirty. Pending / hydrating /
        failed bundles are skipped (their data isn't in memory, so the
        on-disk state IS the only state -- nothing to lose).

        The active bundle's scan is identical to single-bundle
        :meth:`_scan_unsaved_layers`, but reaches the same code path
        by temporarily binding ``self.annotations`` to each bundle's
        container during the scan. This avoids forking the diff logic.
        """
        result: dict = {}
        if not self._bundles:
            return result
        saved_annotations = self.annotations
        saved_fname = self.fname
        try:
            for bundle in self._bundles:
                if not bundle.is_ready or bundle.annotations is None:
                    continue
                # Re-point shell attrs so the existing per-layer scan
                # (which reads self.annotations + self.fname for the
                # is-manual-layer predicate) sees this bundle's state.
                self.annotations = bundle.annotations
                self.fname = str(bundle.fname)
                layers = self._scan_unsaved_layers()
                if layers:
                    result[bundle.video_index] = {
                        "fname": bundle.fname,
                        "layers": layers,
                    }
        finally:
            self.annotations = saved_annotations
            self.fname = saved_fname
        return result

    def _install_close_guard(self) -> None:
        """Patch the QMainWindow ``closeEvent`` so window close triggers
        the unsaved-diffs scan + modal.

        Monkey-patch rather than subclass because the QMainWindow is
        constructed inside matplotlib's Qt backend; intercepting it
        without owning the type means patching the instance. The
        original ``closeEvent`` is chained at the end so any
        backend-internal cleanup still runs.

        No-op on the mpl fallback path (no Qt window to hook).
        """
        qt_window = self._find_qt_window()
        if qt_window is None:
            return
        if getattr(qt_window, "_dustrack_close_guard_installed", False):
            return  # idempotent: a second __init__ pass (e.g. subclass) must not stack

        original_close_event = qt_window.closeEvent
        dustrack_self = self

        def closeEvent(event):
            # Seed-session short-circuit (1.2.0a3): the seed-modal
            # launch path mounts the modal on a synthetic seed video
            # and tears it down on dismiss / cancel. There are no
            # user edits worth prompting about, and the seed asset
            # path must not land in ``recent_sessions``. Skip every
            # tail step that would survey state or write history.
            if getattr(dustrack_self, "_is_seed_session", False):
                original_close_event(event)
                return
            try:
                # Multi-bundle (1.2.0a3): sweep every ready bundle's
                # unsaved diffs, not just the active one. The
                # single-bundle case (most users) ends up with a
                # one-entry dict pointing at the active bundle, so
                # the modal renders identically to the pre-1.2.0a3
                # path.
                unsaved = dustrack_self._scan_unsaved_layers_all_bundles()
            except Exception:
                # If the scan itself fails (e.g. annotation list mutated
                # mid-shutdown), don't block the close -- the guard is a
                # safety net, not a hard gate.
                unsaved = {}
            if unsaved:
                choice = dustrack_self._prompt_save_on_close(qt_window, unsaved)
                if choice == "cancel":
                    event.ignore()
                    return
                if choice == "save":
                    dustrack_self._save_unsaved_layers(unsaved)
                # "discard" falls through to the original closeEvent
            # History write happens after the unsaved-diff gate so a
            # cancelled close does NOT pollute the recent list. Wrapped
            # because a config-write failure (read-only home, disk full)
            # must never strand the user with an un-closeable window.
            try:
                dustrack_self._record_session_in_history()
            except Exception:
                pass
            original_close_event(event)

        qt_window.closeEvent = closeEvent
        qt_window._dustrack_close_guard_installed = True

    def _record_session_in_history(self) -> None:
        """Write the current session's full bundle list to the unified
        ``recent_sessions`` store.

        Called from the close-guard. Single-video sessions write a
        1-element entry; multi-video sessions write the full bundle
        list in queue order (bundle 0 first, then the tail). The
        active video is always the first element so a click-to-reopen
        from the picker lands on the same video the user was on.

        Skipped for seed sessions (the modal-host launch path sets
        ``_is_seed_session = True`` on the temporary seed-tracker;
        recording its synthetic asset path would pollute the recent
        list with an entry that's never useful to reopen).

        Robust to missing attributes (``fname`` is the only required
        signal); the close-guard caller already wraps in try/except.
        """
        if getattr(self, "_is_seed_session", False):
            return
        fname = getattr(self, "fname", None)
        if not fname:
            return
        bundles = getattr(self, "_bundles", None) or []
        if bundles:
            paths = [b.fname for b in bundles]
        else:
            paths = [fname]
        try:
            _config.record_recent_session(paths)
        except Exception:
            # Best-effort -- if the JSON store is unwritable, drop
            # the entry but don't fail the close.
            pass

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
            target=_worker, name="dustrack-overlay-worker", daemon=True,
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
                sys.__stderr__.write(f"{title} failed: {type(exc).__name__}: {exc_str}\n")
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
                    bundle, project, new_suffix,
                )
            except Exception:  # noqa: BLE001 - never abort the post-train UX
                sys.__stderr__.write(
                    f"[dustrack] post-train refresh failed for bundle "
                    f"{bundle.video_index} ({bundle.fname}):\n"
                    f"{traceback.format_exc()}\n"
                )

    def _add_new_dlc_layers_to_bundle(
        self, bundle: _BundleState, project, new_suffix: str,
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
            name: path for name, path in all_layers.items()
            if name not in existing
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
            dustrack.postprocess.lk_moving_average_filter: The filtering algorithm.
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

        removable = [
            n for n in self.annotations.names if n != "buffer"
        ]
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
                self.ann.add(overlay_ann.data[current_label][frame_num], current_label, frame_num)
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
        if old_fname is not None and Path(old_fname).exists() and str(old_fname) != str(new_fname):
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
        suggested_name = Path(
            make_annotation_file_name(self.fname, layer_name)
        ).name
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
        ret = super().update()
        if self._ax_lims['state']:
            if self._ax_lims['x'][0] is not None:
                self._ax_trace_x.set_xlim(self._ax_lims['x'])
                self._ax_trace_y.set_xlim(self._ax_lims['x'])
            if self._ax_lims['y_trace_x'][0] is not None:
                self._ax_trace_x.set_ylim(self._ax_lims['y_trace_x'])
            if self._ax_lims['y_trace_y'][0] is not None:
                self._ax_trace_y.set_ylim(self._ax_lims['y_trace_y'])
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
        """Populate :attr:`_bundles` from the just-constructed shell +
        a list of queued video paths.

        Called by :func:`dustrack.open` (and friends) after the
        active-bundle ``DUSTrack`` is constructed. The just-built
        annotations / VideoReader become bundle 0 (``ready`` from the
        start); pending bundles are scaffolded for each path in
        ``video_paths[1:]`` and hydrated by the background worker
        (1.2.0a3 Slice 2). The user can start work on bundle 0
        immediately; the tail loads in parallel so swap-to is fast
        when the user reaches it.

        Args:
            project: The shared :class:`DLCProject` for all bundles, or
                ``None`` for the (single-bundle) Phase 1 path.
            video_paths: Ordered video paths. ``video_paths[0]`` must
                match ``self.fname``; the rest become pending bundles.
        """
        if len(video_paths) == 0:
            raise ValueError("_init_bundles: video_paths cannot be empty")

        # Bundle 0 -- snapshot the just-constructed shell into the
        # ready bundle's heavy + lightweight fields.
        active_bundle = _BundleState(
            fname=Path(video_paths[0]),
            video_index=0,
            project=project,
            reader=self.data,
            annotations=self.annotations,
            current_idx=self._current_idx,
            ax_lims=dict(self._ax_lims),
            image_view_state=self._get_image_view_state(),
            frames_of_interest=list(self.frames_of_interest),
            hydration_state=HYDRATION_READY,
        )
        active_bundle.selections = self._capture_statevar_selections()
        self._bundles = [active_bundle]

        # Pending bundles for the tail. All share the same project as
        # bundle 0 (1.2.0a3 multi-video contract: same-project only).
        for i, vp in enumerate(video_paths[1:], start=1):
            self._bundles.append(_BundleState(
                fname=Path(vp), video_index=i,
                project=project,
                hydration_state=HYDRATION_PENDING,
            ))

        self._active_index = 0
        # Back-compat: ``_video_queue`` (set since 1.2.0a2 by
        # :func:`dustrack.open`) remains as the tail-of-paths
        # observability attribute. Kept in sync with the bundle list
        # so legacy consumers / tests don't break.
        self._video_queue = [b.fname for b in self._bundles[1:]]

        self._hydration_worker = None
        if project is not None and len(self._bundles) > 1:
            self._hydration_worker = _BgHydrationWorker(
                self, project, self._bundles[1:],
            )
            self._hydration_worker.start()

        # Slice 3: broadcast statevar changes across every bundle.
        # ``annotation_label`` / ``label_range`` / ``number_keys`` are
        # the UI-mode-flavoured statevars that almost always carry
        # across same-project videos -- when the user toggles them
        # on bundle k via dropdown / key cycle, write the new value
        # into every bundle's snapshot so swap-in restores it.
        self._install_broadcast_statevar_hooks()

        self._refresh_nav_buttons()

        # Schedule a synchronous paint of the figure that fires AFTER
        # the Qt event loop starts. Calling ``canvas.draw()`` inline
        # here doesn't help -- ``dustrack.open()`` returns to
        # ``dustrack/cli.py``, which then calls
        # ``plt.show(block=True)`` to start the event loop. Any
        # synchronous paint we do here renders into a buffer that
        # the window hasn't yet been told to display, so the user
        # sees a stale frame until they trigger a repaint manually
        # (window resize, zoom tool, dropdown click, swap-and-back).
        # ``QTimer.singleShot(0, ...)`` posts a deferred event onto
        # the Qt main thread, which fires on the very first idle
        # AFTER the event loop is running -- so the synchronous
        # ``canvas.draw()`` lands after the window is on screen and
        # the trace pane reflects the post-construction state on
        # first open without needing a swap-and-back to "warm up"
        # the canvas. Same root cause + fix shape as the tail of
        # :meth:`swap_to`.
        try:
            from qtpy.QtCore import QTimer
            QTimer.singleShot(0, self.figure.canvas.draw)
        except Exception:  # noqa: BLE001
            # mpl-fallback / no Qt: best-effort sync paint.
            try:
                self.figure.canvas.draw()
            except Exception:  # noqa: BLE001
                pass

    def _install_broadcast_statevar_hooks(self) -> None:
        """Wire ``add_on_change`` callbacks on every broadcast
        statevar so user-driven mutations propagate to every bundle's
        ``selections`` dict (including pending bundles that haven't
        hydrated yet -- they'll honour the value when their initial
        selection is derived).

        Idempotent guard via ``_broadcast_hooks_installed`` so a
        subclass re-entering ``__init__`` doesn't stack callbacks.
        """
        if getattr(self, "_broadcast_hooks_installed", False):
            return
        for sv_name in self._BROADCAST_STATEVARS:
            if sv_name not in self.statevariables.names:
                continue
            sv = self.statevariables[sv_name]
            sv.add_on_change(
                # Bind by default-arg so each closure captures its
                # own name (Python late-binding gotcha).
                lambda _name=sv_name: self._broadcast_statevar(_name),
            )
        self._broadcast_hooks_installed = True

    def _broadcast_statevar(self, sv_name: str) -> None:
        """Write the shell's current value for ``sv_name`` into every
        bundle's ``selections`` dict.

        Called from the ``add_on_change`` hook installed by
        :meth:`_install_broadcast_statevar_hooks`. Silent-restore in
        :meth:`_restore_statevar_selections` bypasses the on_change
        callback chain, so this fires only on genuine user mutations
        (combo box pick, key cycle) -- not on swap-in restores. That
        bidirectional split is what keeps swap-in from triggering a
        broadcast that would then overwrite every bundle's
        just-restored value with the active bundle's.
        """
        if sv_name not in self.statevariables.names:
            return
        new_value = self.statevariables[sv_name].current_state
        for bundle in self._bundles:
            bundle.selections[sv_name] = new_value

    def _await_hydration(self, bundle: _BundleState) -> bool:
        """Block (pumping the Qt event loop) until ``bundle`` reaches a
        terminal state.

        Returns ``True`` if the bundle is ready, ``False`` if it
        failed. Used by :meth:`swap_to` when the user clicks ahead of
        the background hydration worker.

        Pumps :meth:`QCoreApplication.processEvents` so the UI stays
        responsive while waiting -- the window can still receive
        paints / wheel-zoom / close events. No overlay in Slice 2
        (added in a follow-up if the wait becomes noticeable; per-
        bundle hydration is ~2-3 s on typical pia02 videos and
        usually finishes long before the user clicks anywhere).
        """
        if bundle.is_terminal:
            return bundle.is_ready
        try:
            from qtpy.QtCore import QCoreApplication
            qt_pump = QCoreApplication.processEvents
        except Exception:  # noqa: BLE001
            qt_pump = None
        import time as _time
        deadline_per_tick = 0.02  # 50 Hz poll
        while not bundle.is_terminal:
            if qt_pump is not None:
                qt_pump()
            _time.sleep(deadline_per_tick)
        return bundle.is_ready

    def _hydrate_bundle_data_only(
        self, bundle: _BundleState, project,
    ) -> dict:
        """Off-thread half of bundle hydration. Returns a payload the
        Qt-thread half (:meth:`_finalise_bundle_artists`) consumes.

        Touches: filesystem (VideoReader open, JSON / h5 reads),
        numpy / pandas (vectorised DLC trace decoding), VideoAnnotation
        construction with EMPTY axis lists so the artist setup
        downstream is a no-op. Does NOT touch Qt or matplotlib --
        safe to call from a daemon thread.

        State machine: PENDING -> HYDRATING (set on entry). Caller
        flips to READY / FAILED based on the rest of the pipeline.
        """
        bundle.hydration_state = HYDRATION_HYDRATING
        # Resolve this bundle's position in the project's video list
        # (DLC keys by canonical path; stem fallback handles drive-
        # letter / UNC drift).
        video_index = _find_video_index(project, bundle.fname)
        if video_index is None:
            raise ValueError(
                f"bundle video {bundle.fname} is not in DLC project "
                f"{project.config_path}"
            )
        in_project_path = Path(project.video_list[video_index])

        # Compute the next-iteration suffix exactly the way
        # ``DLCProject.annotate`` does, so a Train DLC run cuts the
        # same fresh layer regardless of which bundle was active when
        # the user clicked Train.
        if project.latest_iteration_is_trained():
            new_iteration_num = project.latest_iteration + 1
        else:
            new_iteration_num = project.latest_iteration
        new_annotation_suffix = f"iteration-{new_iteration_num}"

        fm = VideoFileManager(project, video_index)
        ann_name_to_fname = fm.get_all_annotation_layers(new_annotation_suffix)
        ann_name_to_fname["buffer"] = fm.get_new_json("buffer")

        # Open a dedicated VideoReader for this bundle. Each bundle
        # owns its own reader -- one open file per video in the
        # queue. Thread-safe to construct (each instance is
        # independent).
        with builtins_open(str(in_project_path), "rb") as f:
            reader = VideoReader(f)

        # Fresh VideoAnnotations container. ``parent=self`` lets
        # downstream callers reach the shell. The container holds the
        # per-layer artist handles after the Qt-thread half runs;
        # for now the per-annotation ``plot_handles`` are empty.
        # VideoAnnotation.__init__'s ``load()`` reads DLC .h5 files
        # via pandas.read_hdf -> PyTables, which is NOT thread-safe.
        # Serialise the entire VideoAnnotation construction loop
        # behind the module-level HDF5 lock so concurrent bundle
        # hydrations + main-thread h5 reads can't race.
        container = VideoAnnotations(parent=self)
        with _HDF5_LOCK:
            for name, fname in ann_name_to_fname.items():
                # EMPTY ax_list_scatter / ax_list_trace_x / ax_list_trace_y:
                # VideoAnnotation.__init__ calls setup_display(), which
                # iterates the (empty) ax lists and skips artist creation.
                # The Qt-thread half attaches real axes + builds artists.
                ann = container.add(
                    name=name,
                    fname=fname,
                    vname=str(in_project_path),
                    video=reader,
                    ax_list_scatter=[],
                    ax_list_trace_x=[],
                    ax_list_trace_y=[],
                    palette_name="Set2",
                    n_labels=1,
                )
                ann.__class__ = VideoAnnotation

        # Union of declared labels across the layers so every layer
        # presents the same label rotation -- mirrors
        # ``_DUSTrackBase.add_annotation_layers``. ``re_setup_display``
        # at the tail is a no-op because the ax lists are empty.
        all_labels = sorted(
            {label for ann in container._list for label in ann.labels}
        )
        if not all_labels:
            all_labels = ["0"]
        for ann in container._list:
            for label in all_labels:
                if label not in ann.labels:
                    ann.add_label(label)
            ann.sort_labels()
            ann.re_setup_display()

        return {
            "reader": reader,
            "container": container,
            "in_project_path": in_project_path,
        }

    def _hydrate_phase1_bundle_data(
        self, bundle: _BundleState, *, layer_name: str = "iteration-0",
    ) -> dict:
        """Off-thread half of Phase 1 (bare-video, no DLC project)
        bundle hydration. Returns a payload the Qt-thread half
        (:meth:`_finalise_bundle_artists`) consumes.

        Mirrors :meth:`_hydrate_bundle_data_only`'s contract, except
        the layer set is the canonical Phase 1 pair (``layer_name``
        + ``buffer``) with paths derived from the bundle's video
        stem -- no ``VideoFileManager`` / project lookup. Used by
        :meth:`add_video` when appending a bare-video bundle to a
        live tracker (notably the seed-modal flow).

        Touches: filesystem (VideoReader open, JSON reads if the
        layer .json sits next to the video), VideoAnnotation
        construction with EMPTY axis lists so the artist setup
        downstream is a no-op. Does NOT touch Qt or matplotlib --
        safe to call from a daemon thread.

        State machine: PENDING -> HYDRATING (set on entry). Caller
        flips to READY / FAILED based on the rest of the pipeline.
        """
        bundle.hydration_state = HYDRATION_HYDRATING

        vname = str(bundle.fname)
        # Phase 1 layer-fname derivation mirrors
        # ``_DUSTrackBase._get_fname_annotations``: alongside the
        # video, ``<stem>_annotations_<layer>.json``.
        def _ann_path(name: str) -> str:
            stem = bundle.fname.stem
            suffix_part = f"_{name}" if name else ""
            return str(bundle.fname.parent / f"{stem}_annotations{suffix_part}.json")

        ann_name_to_fname = {
            layer_name: _ann_path(layer_name),
            "buffer": _ann_path("buffer"),
        }

        with builtins_open(vname, "rb") as f:
            reader = VideoReader(f)

        # Empty ax lists -- ``_finalise_bundle_artists`` wires real
        # ax lists in the Qt-thread half. Same _HDF5_LOCK pattern as
        # the Phase 2 path for symmetry; Phase 1 .json reads aren't
        # HDF5-backed but the lock is cheap when uncontended and
        # keeps the data-half contract consistent across phases.
        container = VideoAnnotations(parent=self)
        with _HDF5_LOCK:
            for name, fname in ann_name_to_fname.items():
                ann = container.add(
                    name=name,
                    fname=fname,
                    vname=vname,
                    video=reader,
                    ax_list_scatter=[],
                    ax_list_trace_x=[],
                    ax_list_trace_y=[],
                    palette_name="Set2",
                    n_labels=1,
                )
                ann.__class__ = VideoAnnotation

        # Union of declared labels across layers, mirroring
        # ``_DUSTrackBase.add_annotation_layers``.
        all_labels = sorted(
            {label for ann in container._list for label in ann.labels}
        )
        if not all_labels:
            all_labels = ["0"]
        for ann in container._list:
            for label in all_labels:
                if label not in ann.labels:
                    ann.add_label(label)
            ann.sort_labels()
            ann.re_setup_display()

        return {
            "reader": reader,
            "container": container,
            "in_project_path": bundle.fname,
        }

    def _finalise_bundle_artists(
        self, bundle: _BundleState, payload: dict, project,
    ) -> None:
        """Qt-thread half of bundle hydration. Wires each annotation's
        artists into the shell's axes (per-layer marker group on the
        image pane, shared trace axes), then hides every artist so the
        bundle is parked invisible until the user swaps to it.

        MUST run on the Qt thread: ``_image_pane.add_marker_group()``
        modifies the QGraphicsScene, which is not thread-safe.

        On success: bundle ``hydration_state`` flips to ``ready``,
        ``selections`` seeded to the canonical fresh-load state, and
        ``_refresh_nav_buttons`` is called so the position indicator
        + arrow enable states update.
        """
        container = payload["container"]
        reader = payload["reader"]
        # Wire artists for each annotation. Tier 2 builds a per-layer
        # marker group on the image pane; Tier 1 reuses the shell's
        # image axis directly. Call ``setup_display`` directly (NOT
        # ``re_setup_display``) because the latter clears existing
        # handles first, and the data-only init never populated any
        # ``labels_in_ax*`` keys to clear (empty ax_list_scatter ->
        # empty plot_handles -> KeyError inside clear_display when
        # the new ax_list_scatter has length > 0).
        for ann in container._list:
            if self._fast_render:
                ax_list_scatter = [self._image_pane.add_marker_group()]
            else:
                ax_list_scatter = [self._ax_image]
            ann.setup_display(
                ax_list_scatter=ax_list_scatter,
                ax_list_trace_x=[self._ax_trace_x],
                ax_list_trace_y=[self._ax_trace_y],
            )
            # Apply the per-annotation plot-type convention that the
            # active bundle's construction path gets for free via
            # ``DLCProject.annotate`` -> ``_normalize_dlc_layer_display``
            # and ``_DUSTrackBase.__init__``'s ``buffer.plot_type =
            # "line"`` line. Without this, every bundle-k+1 dense
            # layer (DLC traces, lkmovavg outputs, dlccorr, buffer)
            # defaults to ``setup_display``'s ``set_plot_type(self.plot_type)``
            # tail which reads the constructor's ``_plot_type="dot"``
            # default. Hits the user as "swap to bundle 2 -> traces
            # are dots not lines, and Trace:line button doesn't help
            # because it acts on the (empty) active layer".
            if ann.name == "buffer" or _is_dense_layer_name(ann.name):
                try:
                    ann.set_plot_type("line", draw=False)
                except Exception:  # noqa: BLE001
                    pass
            # Invalidate the trace cache so the first ``update_display_trace``
            # against the freshly-bound handles repopulates ydata.
            ann.invalidate_caches()
            ann.hide(draw=False)

        bundle.reader = reader
        bundle.annotations = container
        # Derive the canonical fresh-load selections (active = latest
        # manual, overlay = latest dlc_*) for this bundle.
        derived = self._derive_initial_bundle_selections(
            container, project=project,
        )
        # If the user toggled a broadcast statevar
        # (annotation_label / label_range / number_keys) while this
        # bundle was pending, the broadcast wrote into
        # ``bundle.selections`` BEFORE hydration completed. Preserve
        # those user-driven values; only the per-video statevars
        # (annotation_layer / annotation_overlay) come from the
        # derived canonical defaults.
        existing = bundle.selections or {}
        for sv_name in self._BROADCAST_STATEVARS:
            if sv_name in existing:
                derived[sv_name] = existing[sv_name]
        bundle.selections = derived
        bundle.hydration_state = HYDRATION_READY
        bundle.hydration_error = None
        try:
            self._refresh_nav_buttons()
        except Exception:  # noqa: BLE001
            pass

    def _derive_initial_bundle_selections(
        self, container: VideoAnnotations, project=None,
    ) -> dict:
        """First-time statevar selections for a freshly-hydrated bundle.

        Picks the canonical fresh-load state: latest manual layer as
        active, latest ``dlc_*`` layer as overlay (or None), first
        label / its label_range as the active bodypart, current
        shell's ``number_keys`` mode (so the cross-bundle UI mode
        carries from the start).
        """
        names = container.names
        # Latest manual layer = active. Manual layers are everything
        # except buffer / dense (dlc_* / dlccorr* / lkmovavg).
        manuals = [
            n for n in names
            if n != "buffer" and not _is_dense_layer_name(n)
        ]
        # The new ``iteration-{N+1}`` layer (just created by
        # ``get_all_annotation_layers``) lands at the tail of the
        # manuals block -- match ``DLCProject.annotate``'s convention
        # by picking the LAST manual as active.
        active_layer = manuals[-1] if manuals else (names[0] if names else None)
        dlc_layers = [n for n in names if n.startswith("dlc_")]
        overlay = dlc_layers[-1] if dlc_layers else None
        # Active label / label_range -- derive from the active layer.
        ann = container[active_layer] if active_layer else None
        if ann and ann.labels:
            first_label = ann.labels[0]
            try:
                label_range_idx = int(first_label) // 10
                label_range_value = f"{label_range_idx*10}-{label_range_idx*10+9}"
            except (TypeError, ValueError):
                label_range_value = None
        else:
            first_label = None
            label_range_value = None
        # number_keys carries the shell's current mode (broadcast
        # default).
        nk = None
        if "number_keys" in self.statevariables.names:
            nk = self.statevariables["number_keys"].current_state
        return {
            "annotation_layer": active_layer,
            "annotation_overlay": overlay,
            "annotation_label": first_label,
            "label_range": label_range_value,
            "number_keys": nk,
        }

    def _capture_statevar_selections(self) -> dict:
        """Snapshot the shell's current statevar selections (5 names)
        for the active bundle. Names absent from the container are
        omitted (mpl-fallback path may skip some Qt-only statevars).
        """
        out: dict = {}
        for sv in self._ALL_TRACKED_STATEVARS:
            if sv in self.statevariables.names:
                out[sv] = self.statevariables[sv].current_state
        return out

    def _restore_statevar_selections(
        self, selections: dict, layer_names: list,
    ) -> None:
        """Rewrite each statevar's ``states`` list to the new bundle's
        rotation, restore the snapshotted selection silently (bypass
        on_change callbacks so the per-statevar cascade doesn't fire
        during the restore), then refresh the Qt sidebar widgets in
        one ``_text.update()`` call.

        Silent restore matters because on_change callbacks include
        :meth:`_refresh_workflow_button_state` and
        :meth:`_on_active_label_change` -- firing them mid-restore
        would re-read partially-rebuilt state and either thrash the
        gates or trigger an erroneous label change. The pattern
        (direct ``_current_state_idx = ...`` + manual
        ``_text.update()``) mirrors :meth:`select_label_with_mouse`'s
        existing bypass (see ``StateVariable`` callback note).
        """
        # 1. Rewrite the rotations from the new bundle's layer list.
        if "annotation_layer" in self.statevariables.names:
            sv = self.statevariables["annotation_layer"]
            sv.states = list(layer_names)
        if "annotation_overlay" in self.statevariables.names:
            sv = self.statevariables["annotation_overlay"]
            sv.states = [None] + list(layer_names)

        # 2. Restore each snapshotted selection. annotation_layer /
        # annotation_overlay must come BEFORE annotation_label so the
        # active layer is set when we re-derive the label rotation.
        for sv_name in self._ALL_TRACKED_STATEVARS:
            if sv_name not in self.statevariables.names:
                continue
            if sv_name not in selections:
                continue
            sv = self.statevariables[sv_name]
            value = selections[sv_name]
            try:
                idx = sv.states.index(value)
            except ValueError:
                # Snapshot value isn't valid for this bundle (e.g.
                # broadcast label that doesn't exist in this layer's
                # rotation). Fall back to the first state.
                idx = 0
            sv._current_state_idx = idx
            # For annotation_label specifically, also refresh the
            # label rotation against the new active layer before
            # locking in the selection. The base class
            # ``update_annotation_label_states`` reads
            # ``self.ann.labels`` -- which is now the new bundle's
            # active layer.
            if sv_name == "annotation_layer":
                self.update_annotation_label_states()

        # 3. Sync the Qt sidebar widgets (combo boxes / toggle button
        # group) to the new states + selections.
        try:
            if self.statevariables._text is not None:
                self.statevariables._text.update()
        except Exception:  # noqa: BLE001 - mpl-fallback / pre-teardown
            pass

    # ------------------------------------------------------------------
    # Image-pane viewport snapshot / restore (Tier 1 + Tier 2 dispatch)
    # ------------------------------------------------------------------

    def _get_image_view_state(self):
        """Snapshot the current image pane's zoom / pan state.

        Returns an opaque blob the matching :meth:`_set_image_view_state`
        understands. Tier 2 (Qt-native) wraps QGraphicsView's
        transform + scrollbar positions; Tier 1 (matplotlib) wraps
        the image axis's xlim / ylim. ``None`` = no viewport saved /
        nothing rendered yet (caller restores to fit-frame).
        """
        if self._fast_render:
            pane = self._image_pane
            getter = getattr(pane, "get_view_state", None)
            if getter is None:
                return None
            try:
                return getter()
            except Exception:  # noqa: BLE001 - defensive
                return None
        # Tier 1: mpl Axes. Read xlim/ylim; treat axis-defaults as
        # "no snapshot" so the next swap-in stays at fit-frame.
        ax = self._ax_image
        if ax is None:
            return None
        try:
            xlim = tuple(ax.get_xlim())
            ylim = tuple(ax.get_ylim())
        except Exception:  # noqa: BLE001
            return None
        return {"kind": "mpl", "xlim": xlim, "ylim": ylim}

    def _set_image_view_state(self, state) -> None:
        """Restore a previously-snapshotted viewport. ``None`` falls
        back to fit-frame on Tier 2 (pane's ``reset_view``) or a
        no-op autoscale on Tier 1.
        """
        if self._fast_render:
            pane = self._image_pane
            setter = getattr(pane, "set_view_state", None)
            if setter is not None:
                try:
                    setter(state)
                except Exception:  # noqa: BLE001
                    pass
            elif state is None:
                reset = getattr(pane, "reset_view", None)
                if reset is not None:
                    try:
                        reset()
                    except Exception:  # noqa: BLE001
                        pass
            return
        ax = self._ax_image
        if ax is None:
            return
        if state is None or state.get("kind") != "mpl":
            try:
                ax.relim()
                ax.autoscale_view()
            except Exception:  # noqa: BLE001
                pass
            return
        try:
            ax.set_xlim(state["xlim"])
            ax.set_ylim(state["ylim"])
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Swap entry points
    # ------------------------------------------------------------------

    def swap_to(self, index: int) -> bool:
        """Switch the active video to ``self._bundles[index]``.

        Implements the swap contract from ``specs/dustrack.md``
        Roadmap *Next 1.2.0* item 3:

        1. Snapshot the active bundle's per-video state (frame, axis
           limits, image pane viewport, statevar selections, frames
           of interest).
        2. Park the leaving bundle's artists (``ann.hide(draw=False)``
           on every annotation -- data + ``_revision`` survive
           untouched in memory, so a swap back is instant).
        3. Rebind shell attributes (``fname``, ``data``,
           ``annotations``, ``_current_idx``, ``_ax_lims``,
           ``frames_of_interest``) onto the arriving bundle.
        4. Show the arriving bundle's artists.
        5. Restore the arriving bundle's statevar selections (silent
           callback bypass) and image pane viewport.
        6. Repaint once via :meth:`update`.

        Returns ``True`` on a successful swap (or no-op when
        ``index`` is already active), ``False`` when the swap was
        rejected (out-of-bounds, non-ready bundle, or a Cancel from
        the future loading overlay).
        """
        if not (0 <= index < len(self._bundles)):
            return False
        if index == self._active_index:
            return True
        target = self._bundles[index]
        if target.hydration_state == HYDRATION_FAILED:
            self._notify_bundle_failure(target)
            return False
        if not target.is_ready:
            # Bundle is still being hydrated by the bg worker. Wait
            # (pumping Qt events so the UI stays responsive) for it
            # to reach a terminal state. _await_hydration returns
            # True iff the bundle is now READY; FAILED returns False.
            ready = self._await_hydration(target)
            if not ready:
                self._notify_bundle_failure(target)
                return False

        # 1. Snapshot leaving bundle.
        self._snapshot_active_bundle()

        # 2. Park leaving bundle's artists.
        leaving = self._bundles[self._active_index]
        self._park_bundle_artists(leaving)

        # 3. Rebind shell onto arriving bundle.
        self._attach_bundle(target)

        # 4. Show arriving bundle's artists.
        self._show_bundle_artists(target)

        # 4b. Restore (or first-time-fit) the trace axes view. Same
        # contract as the image pane viewport: if the arriving
        # bundle has a captured ``trace_view_state`` (it's been
        # swapped-out from before), restore the exact xlim/ylim the
        # user left. If not (first visit to this bundle), apply a
        # default fit (xlim = (0, n_frames), autoscale-y on) so the
        # trace pane shows the new video's data at the right scale
        # instead of inheriting bundle 0's axis range.
        if not self._ax_lims["state"]:
            if target.trace_view_state is not None:
                self._set_trace_view_state(target.trace_view_state)
                # The marker-cache keys on (current_label, per-ann
                # revisions, FOI). After a restore, the per-ann
                # revisions don't match the leaving bundle's cache,
                # so the next paint recomputes anyway -- but clear
                # defensively in case two bundles' revision tuples
                # happened to collide.
                self._frame_marker_cache = None
            else:
                # First visit -- ``setup_display_trace`` only claims
                # ``set_xlim`` while autoscalex_on is True, and the
                # leaving bundle's setup turned that off. Force-fit
                # to the arriving bundle's frame count + re-enable
                # autoscale-y so ``update_frame_marker`` refits ylim
                # on the next paint.
                n_frames = len(target.reader)
                self._ax_trace_x.set_xlim(0, n_frames)
                self._ax_trace_x.set_autoscalex_on(True)
                self._ax_trace_x.set_autoscaley_on(True)
                self._ax_trace_y.set_autoscaley_on(True)
                self._frame_marker_cache = None

        # 5. Restore statevars + image viewport + enhance state.
        self._restore_statevar_selections(
            target.selections, target.annotations.names,
        )
        self._set_image_view_state(target.image_view_state)
        self._set_enhance_state(target.enhance_state)

        # 6. Repaint. ``DUSTrack.update`` calls
        # ``canvas.flush_events()`` after the base update so the
        # deferred paint actually runs. Note the multi-video
        # interactive-render limitation documented in
        # ``DUSTrack.update``: the user may need to flip videos
        # once after open for ``draw_idle`` to start firing.
        self._active_index = index
        self._refresh_nav_buttons()
        try:
            self._refresh_workflow_button_state()
        except Exception:  # noqa: BLE001
            pass
        self.update()
        return True

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
        self, path_or_paths, *, layer_name=None, set_active=False,
        **dustrack_kwargs,
    ) -> list[int]:
        """Append one or more videos to this tracker's bundle list.

        Validates and hydrates the picked path(s), appends the new
        bundle(s) to :attr:`_bundles`, optionally swaps to the first
        new bundle. Mirrors :func:`dustrack.open`'s validation: a
        scalar path resolves to Phase 1 (bare video) or Phase 2
        (video in a DLC project) by scanning for a ``config.yaml``
        next to the file; a list of paths must all belong to the
        same DLC project (Phase 2 multi only).

        Args:
            path_or_paths: A single video path (``str`` /
                :class:`Path`), or a list/tuple of such paths.
            layer_name: Annotation layer name for the new bundle.
                Phase 1 default: ``'iteration-0'``. Phase 2: ignored
                (the project's ``iteration-{N+1}`` convention wins).
            set_active: If True, swap to the first new bundle after
                hydration. Used by :meth:`replace_active_with`; the
                public default is False so callers can add bundles
                in the background without disturbing the user's
                current view.
            **dustrack_kwargs: Reserved for future per-bundle
                construction options. Today only the
                :func:`dustrack.open` kwarg set is forwarded, but
                bundles built post-construction inherit the shell's
                kwargs (``fast_render``, ``dark_mode``, etc.) so
                this list is effectively empty.

        Returns:
            list[int]: Indices of the newly-appended bundles in
            :attr:`_bundles`, in queue order.

        Raises:
            FileNotFoundError: A path doesn't exist.
            ValueError: Empty sequence, multi-video list with
                mixed / missing DLC projects, etc. -- same shape
                as :func:`dustrack.open`'s validation errors.
            ImportError: Phase 2 entry on a system without
                ``deeplabcut`` installed.
        """
        # Normalise to (project, [paths]) -- same logic as
        # :func:`dustrack.open` post-validation but split out so
        # the existing top-level dispatch and add_video share it.
        project, video_paths = self._validate_bundle_paths(path_or_paths)

        # Hydrate the first new bundle synchronously so a swap-to
        # immediately after add_video is a no-wait. The tail (for
        # multi-video adds) goes PENDING and the bg worker takes
        # over -- same shape as ``_init_bundles``.
        base_index = len(self._bundles)
        new_bundles: list[_BundleState] = []
        first = _BundleState(
            fname=Path(video_paths[0]),
            video_index=base_index,
            project=project,
            hydration_state=HYDRATION_PENDING,
        )
        self._hydrate_bundle_sync(first)
        if first.hydration_state == HYDRATION_FAILED:
            raise RuntimeError(
                f"add_video: hydration failed for {first.fname}: "
                f"{first.hydration_error}"
            )
        new_bundles.append(first)
        for i, vp in enumerate(video_paths[1:], start=1):
            new_bundles.append(_BundleState(
                fname=Path(vp), video_index=base_index + i,
                project=project,
                hydration_state=HYDRATION_PENDING,
            ))
        self._bundles.extend(new_bundles)
        # Sync the legacy back-compat ``_video_queue`` attribute (set
        # by :func:`dustrack.open` since 1.2.0a2) so observers see the
        # extended queue.
        self._video_queue = [b.fname for b in self._bundles[1:]]
        # Kick off the bg worker for any PENDING tail bundles. Same
        # contract as :meth:`_init_bundles`: same-project across the
        # batch, daemon thread, Qt poller drains finalisations.
        pending_tail = [b for b in new_bundles[1:] if b.hydration_state == HYDRATION_PENDING]
        if pending_tail and project is not None:
            worker = _BgHydrationWorker(self, project, pending_tail)
            worker.start()
            # Track most-recent worker for diagnostics; tests can poke it.
            self._hydration_worker = worker
        self._refresh_nav_buttons()
        new_indices = [b.video_index for b in new_bundles]
        if set_active:
            self.swap_to(new_indices[0])
        return new_indices

    def remove_video(self, index: int) -> bool:
        """Drop a bundle from the tracker's bundle list.

        If ``index == self._active_index``, swaps to another bundle
        first (prefers ``index + 1``, falls back to ``index - 1``).
        Refuses to empty the bundle list -- a tracker without any
        bundles is undefined; callers wanting "reset to empty" should
        close the window instead.

        Args:
            index: 0-based bundle index in :attr:`_bundles`.

        Returns:
            bool: ``True`` on success, ``False`` if the index is
            out of bounds or the removal was refused (would empty
            the list).

        Notes:
            Renumbering: every surviving bundle's ``video_index`` is
            re-assigned to its post-removal position, so external
            consumers holding indices need to refresh.
        """
        if not (0 <= index < len(self._bundles)):
            return False
        if len(self._bundles) <= 1:
            return False
        if index == self._active_index:
            # Swap-first: prefer next; fall back to previous when at
            # the tail. The fallback index uses the pre-removal layout,
            # so a swap to ``index - 1`` lands on the bundle that will
            # end up at ``index - 1`` post-removal.
            if index + 1 < len(self._bundles):
                target = index + 1
            else:
                target = index - 1
            if not self.swap_to(target):
                return False
        # After the swap, ``self._active_index`` no longer equals
        # ``index`` (or never did). Park the leaving bundle's artists.
        leaving = self._bundles[index]
        self._park_bundle_artists(leaving)
        # Drop the bundle. If the removed bundle was below the active
        # index, the active index shifts down by one.
        del self._bundles[index]
        if index < self._active_index:
            self._active_index -= 1
        # Renumber surviving bundles so ``video_index`` matches the
        # new list position. Consumers that walk the bundle list
        # (nav widget, bg worker batches) read ``video_index``.
        for new_idx, bundle in enumerate(self._bundles):
            bundle.video_index = new_idx
        # Refresh observable attributes.
        self._video_queue = [b.fname for b in self._bundles[1:]]
        self._refresh_nav_buttons()
        return True

    def replace_active_with(
        self, path_or_paths, *, layer_name=None, **dustrack_kwargs,
    ) -> list[int]:
        """Swap the active bundle for one (or more) newly-picked
        video(s); drop the previously-active bundle.

        The 1.2.0a3 seed-modal flow uses this to transition from the
        synthetic seed video to whatever the user picked: it adds
        the picked bundle(s), swaps to the first new bundle, then
        removes the seed bundle. Generalizes to "I want a different
        video / set of videos as my active session, keeping every
        other bundle in place" -- the parked tail bundles survive
        the replace.

        Args:
            path_or_paths: Same shape as :meth:`add_video`.
            layer_name: Forwarded to :meth:`add_video`.
            **dustrack_kwargs: Forwarded to :meth:`add_video`.

        Returns:
            list[int]: Final indices of the new bundles after the
            old active bundle is removed.

        Raises:
            FileNotFoundError, ValueError, ImportError: Forwarded
                from :meth:`add_video`'s validation.
            RuntimeError: Hydration of the active picked bundle
                failed; the tracker is left unchanged (the failed
                bundle is rolled back).
        """
        old_active_bundle = self._bundles[self._active_index]
        try:
            new_indices = self.add_video(
                path_or_paths,
                layer_name=layer_name,
                set_active=True,
                **dustrack_kwargs,
            )
        except Exception:
            # Roll back any failed-bundle artifacts the hydration
            # left in place. add_video raises before mutating
            # _bundles on hydration failure, so this is a safety net
            # for the validation-error path.
            self._video_queue = [b.fname for b in self._bundles[1:]]
            self._refresh_nav_buttons()
            raise
        # Find old_active_bundle's *current* index post-add (it may
        # have shifted if swap_to renumbered something, though today
        # it doesn't). Identity match by object identity, not __eq__,
        # so dataclass field equality doesn't confuse the lookup.
        old_idx = next(
            (i for i, b in enumerate(self._bundles) if b is old_active_bundle),
            None,
        )
        if old_idx is None:
            # Defensive: shouldn't happen since add_video appends.
            return new_indices
        n_new = len(new_indices)
        # Remove the now-non-active old bundle. remove_video renumbers
        # everything below the removed position; after removal the new
        # bundles sit at the tail of self._bundles.
        self.remove_video(old_idx)
        total = len(self._bundles)
        return list(range(total - n_new, total))

    def _validate_bundle_paths(self, path_or_paths) -> tuple:
        """Resolve ``path_or_paths`` into ``(project_or_None, [Path...])``.

        Mirrors the validation logic at the top of :func:`dustrack.open`
        but extracted so :meth:`add_video` can reuse it without
        re-running the full dispatch. The contract:

        - Single path -> Phase 1 (project=None) if no ``config.yaml``
          is found beside it, else Phase 2 (project resolved).
        - List of paths -> Phase 2 multi (must all belong to one
          shared DLC project). Bare-video entries raise.
        - Project folder -> Phase 2 multi (queue every video in the
          project).
        - ``config.yaml`` -> Phase 2 single on the first project
          video.
        """
        if isinstance(path_or_paths, (list, tuple)):
            if len(path_or_paths) == 0:
                raise ValueError("add_video: empty path sequence")
            paths = [Path(p) for p in path_or_paths]
            for p in paths:
                if not p.exists():
                    raise FileNotFoundError(
                        f"add_video: path does not exist: {p}"
                    )
            if len(paths) == 1:
                return self._validate_bundle_paths(paths[0])
            return _resolve_multi_video_from_list(paths)

        p = Path(path_or_paths)
        if not p.exists():
            raise FileNotFoundError(
                f"add_video: path does not exist: {p}"
            )
        if _is_dlc_config_yaml(p):
            # Mirror dustrack.open's config.yaml dispatch (1.2.0a3
            # follow-up): queue every video in the project, in
            # config['video_sets'] order. DLCProject.__init__ runs
            # rebase_to_config so a renamed project folder self-
            # heals before we enumerate.
            if not HAS_DLC:
                raise ImportError(
                    f"add_video: {p} is a DLC config.yaml but "
                    "deeplabcut is not installed."
                )
            project = DLCProject(str(p))
            video_paths = [Path(v) for v in project.video_list]
            if not video_paths:
                raise ValueError(
                    f"add_video: DLC project at {p.parent} has no videos."
                )
            return project, video_paths
        if p.is_dir():
            if not _is_dlc_project_root(p):
                raise ValueError(
                    f"add_video: {p!s} is a directory but doesn't look "
                    "like a DLC project."
                )
            if not HAS_DLC:
                raise ImportError(
                    f"add_video: detected a DLC project at {p}, but "
                    "deeplabcut is not installed."
                )
            project = DLCProject(str(p / "config.yaml"))
            video_paths = [Path(v) for v in project.video_list]
            if not video_paths:
                raise ValueError(
                    f"add_video: DLC project at {p} has no videos."
                )
            return project, video_paths

        # File. Phase 1 vs Phase 2 split by config.yaml discovery.
        config_path = _find_dlc_config(p)
        if config_path is None:
            return None, [p]
        if not HAS_DLC:
            raise ImportError(
                f"add_video: detected a DLC project at {config_path.parent}, "
                "but deeplabcut is not installed."
            )
        return DLCProject(str(config_path)), [p]

    def _hydrate_bundle_sync(self, bundle: _BundleState, project=None) -> None:
        """Populate ``bundle``'s heavy state synchronously, dispatching
        on ``bundle.project`` (Phase 1 vs Phase 2).

        Pre-1.2.0a3: the ``project`` arg was required and only Phase 2
        was supported. The new signature is backwards-compatible
        (``project`` defaults to None, falling back to
        ``bundle.project``); callers that already pass a project keep
        working, and new callers can rely on the bundle's own project
        field.

        Used by single-video / single-bundle entry paths and by the
        worker's failure-path tests; the multi-video happy path goes
        through the worker.
        """
        # Per-bundle project takes precedence over the legacy arg
        # so cross-Phase batches stay coherent. ``project`` is kept
        # for back-compat with the pre-1.2.0a3 call-site.
        eff_project = bundle.project if bundle.project is not None else project
        try:
            if eff_project is None:
                payload = self._hydrate_phase1_bundle_data(bundle)
            else:
                payload = self._hydrate_bundle_data_only(bundle, eff_project)
        except Exception as exc:  # noqa: BLE001
            bundle.hydration_state = HYDRATION_FAILED
            bundle.hydration_error = f"{type(exc).__name__}: {exc}"
            sys.__stderr__.write(
                f"[dustrack] bundle {bundle.video_index} "
                f"({bundle.fname}) hydration failed:\n{traceback.format_exc()}\n"
            )
            return
        try:
            self._finalise_bundle_artists(bundle, payload, eff_project)
        except Exception as exc:  # noqa: BLE001
            bundle.hydration_state = HYDRATION_FAILED
            bundle.hydration_error = f"{type(exc).__name__}: {exc}"
            sys.__stderr__.write(
                f"[dustrack] bundle {bundle.video_index} "
                f"({bundle.fname}) artist setup failed:\n{traceback.format_exc()}\n"
            )

    def _snapshot_active_bundle(self) -> None:
        """Write the shell's current per-video UI state back to the
        active bundle. Called at the top of every swap so the next
        swap back lands the user where they were.
        """
        if not self._bundles:
            return
        active = self._bundles[self._active_index]
        active.current_idx = self._current_idx
        active.ax_lims = dict(self._ax_lims)
        # Deep copies on the inner lists so subsequent shell mutations
        # don't leak back into the snapshot.
        for k in ("x", "y_trace_x", "y_trace_y"):
            if k in active.ax_lims and isinstance(active.ax_lims[k], list):
                active.ax_lims[k] = list(active.ax_lims[k])
        active.image_view_state = self._get_image_view_state()
        active.trace_view_state = self._get_trace_view_state()
        active.enhance_state = self._get_enhance_state()
        active.frames_of_interest = list(self.frames_of_interest)
        active.selections = self._capture_statevar_selections()

    def _get_enhance_state(self) -> dict:
        """Snapshot the shell's current CLAHE / gamma / brightness
        values so a swap-out can preserve them per-bundle. The
        EnhanceWidget sliders bind to these shell attributes; on
        swap-in :meth:`_set_enhance_state` pushes the restored values
        back into the widget so the sliders move to match.
        """
        return {
            "clahe_clip": float(self._clahe_clip),
            "gamma": float(self._gamma),
            "brightness": float(self._brightness),
        }

    def _set_enhance_state(self, state) -> None:
        """Restore a previously-snapshotted enhance state, or reset to
        construction-time defaults on a first-visit (``state is None``).

        Pre-fix this method returned early on first-visit, which meant
        the new bundle inherited the leaving bundle's slider positions
        (e.g. set gamma=1.5 on V1, swap to V2 first-visit, V2's
        sliders showed 1.5 instead of the construction default).
        Restoring to ``_initial_enhance_state`` on first-visit gives
        each bundle a clean baseline; user changes still persist via
        the per-bundle snapshot taken on swap-out.

        Pushes new slider positions into the EnhanceWidget if it's
        mounted (Tier 2 / Qt path) so the visible slider knobs match
        the restored values.
        """
        if state is None:
            state = getattr(self, "_initial_enhance_state", None)
        if state is None:
            # Pre-construction fallback (subclass calling out of order
            # / missing init snapshot). Hold the line.
            return
        self._clahe_clip = float(state["clahe_clip"])
        self._gamma = float(state["gamma"])
        self._brightness = float(state.get("brightness", 0))
        widget = getattr(self, "_enhance_widget", None)
        if widget is None:
            return
        # The EnhanceWidget exposes a sync helper that updates the
        # slider knobs + numeric labels in one go without triggering
        # the per-slider on-change cascade (so this restore doesn't
        # re-write what we just wrote).
        sync = getattr(widget, "sync_from_shell", None)
        if sync is None:
            return
        try:
            sync()
        except Exception:  # noqa: BLE001
            pass

    def _get_trace_view_state(self) -> dict:
        """Snapshot the trace axes' current xlim / ylim so a swap-out
        can preserve the user's pan/zoom on the trace pane the same
        way :meth:`_get_image_view_state` does for the image pane.

        Captures both trace axes (x and y) -- the marker, FOI ticks,
        and the per-label trace lines all share these two axes, and
        a returning swap should land back on the exact view the user
        left.
        """
        return {
            "trace_x_xlim": tuple(self._ax_trace_x.get_xlim()),
            "trace_x_ylim": tuple(self._ax_trace_x.get_ylim()),
            "trace_y_ylim": tuple(self._ax_trace_y.get_ylim()),
        }

    def _set_trace_view_state(self, state) -> None:
        """Restore a previously-snapshotted trace axes view. ``None``
        means "first visit to this bundle" -- caller applies the
        default fit (xlim 0..n_frames, autoscale-y on) instead.
        """
        if state is None:
            return
        try:
            self._ax_trace_x.set_xlim(state["trace_x_xlim"])
            self._ax_trace_x.set_ylim(state["trace_x_ylim"])
            self._ax_trace_y.set_ylim(state["trace_y_ylim"])
        except Exception:  # noqa: BLE001
            pass

    def _attach_bundle(self, bundle: _BundleState) -> None:
        """Rebind shell attributes onto ``bundle``'s heavy state.

        Does not touch artists -- :meth:`_park_bundle_artists` and
        :meth:`_show_bundle_artists` handle visibility; this method
        just swaps the data pointers (fname, VideoReader, annotations
        container, DLC project) and the lightweight UI snapshot
        (frame, axis limits, frames of interest). Statevar /
        image-pane restore happens after this in :meth:`swap_to`.

        Cross-Phase contract (1.2.0a3 seed-modal cut): bundles can
        carry different ``project`` values (``None`` for Phase 1
        bare-video bundles, a ``DLCProject`` for Phase 2). The
        rebind below pushes the arriving bundle's project onto the
        shell so any Workflow-button gating or other project-aware
        code reads the right value on the next paint. Statevar
        rotation refresh (``annotation_layer`` / ``annotation_overlay``
        dropdowns) is handled by
        :meth:`_restore_statevar_selections` later in
        :meth:`swap_to`; this method only handles the data-pointer
        rebind.
        """
        self.fname = str(bundle.fname)
        self.data = bundle.reader
        self.annotations = bundle.annotations
        self._dlcproject = bundle.project
        self._current_idx = bundle.current_idx
        # Force a fresh dict so the shell's later mutations don't
        # alias the bundle snapshot.
        self._ax_lims = dict(bundle.ax_lims)
        for k in ("x", "y_trace_x", "y_trace_y"):
            if k in self._ax_lims and isinstance(self._ax_lims[k], list):
                self._ax_lims[k] = list(self._ax_lims[k])
        self.frames_of_interest = list(bundle.frames_of_interest)

    def _park_bundle_artists(self, bundle: _BundleState) -> None:
        """Hide every annotation artist owned by ``bundle``."""
        if bundle.annotations is None:
            return
        for ann in bundle.annotations._list:
            try:
                ann.hide(draw=False)
            except Exception:  # noqa: BLE001
                # Defensive: never strand the user mid-swap if one
                # artist's hide() raises.
                traceback.print_exc()

    def _show_bundle_artists(self, bundle: _BundleState) -> None:
        """Show every annotation artist owned by ``bundle``."""
        if bundle.annotations is None:
            return
        for ann in bundle.annotations._list:
            try:
                ann.show(draw=False)
            except Exception:  # noqa: BLE001
                traceback.print_exc()

    def _notify_bundle_failure(self, bundle: _BundleState) -> None:
        """Surface a hydration failure to the user.

        Slice 2 will route through a proper error overlay; for Slice 1
        we just print to stderr so the swap-failure case is at least
        observable in the terminal.
        """
        sys.__stderr__.write(
            f"[dustrack] cannot swap to bundle {bundle.video_index} "
            f"({bundle.fname}): {bundle.hydration_error}\n"
        )

    # ------------------------------------------------------------------
    # Sidebar nav row + key bindings
    # ------------------------------------------------------------------

    def _add_nav_widget(self) -> None:
        """Mount the ``◀ <video dropdown> ▶`` nav row at the TOP of the
        rc2 left column dock.

        The central widget is a :class:`QComboBox` listing every
        bundle's video as ``"i. <stem>"`` (1-based); the user can
        either click ◀ / ▶ (Alt+Left / Alt+Right) for sequential
        navigation or select directly from the dropdown to jump to
        an arbitrary video. Per-item tooltips carry the full path so
        the user can confirm which file a stem refers to on hover.

        Always rendered -- when ``N == 1`` (single-video session) the
        arrows are disabled and the dropdown shows a single entry, but
        the row stays visible so the affordance is discoverable when a
        multi-video session is opened later. No-op on the mpl-fallback
        path.
        """
        qt_window = self._find_qt_window()
        if qt_window is None:
            return
        col = getattr(qt_window, "_dnav_left_column", None)
        if col is None:
            return
        from qtpy.QtCore import Qt
        from qtpy.QtGui import QColor
        from qtpy.QtWidgets import (
            QComboBox, QFrame, QHBoxLayout, QSizePolicy, QToolButton, QWidget,
        )

        row = QWidget(col.host)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        prev_btn = QToolButton(row)
        prev_btn.setText("◀")  # ◀
        prev_btn.setFocusPolicy(Qt.NoFocus)
        prev_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        prev_btn.clicked.connect(lambda _checked=False: self.swap_prev())

        combo = QComboBox(row)
        combo.setFocusPolicy(Qt.NoFocus)
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # ``activated[int]`` fires only on user interaction (click /
        # keyboard selection) -- not on programmatic
        # ``setCurrentIndex``, which the post-swap sync uses. That's
        # the right signal here: a sync-after-swap would otherwise
        # recurse into ``swap_to``.
        combo.activated.connect(self._on_nav_combo_activated)

        next_btn = QToolButton(row)
        next_btn.setText("▶")  # ▶
        next_btn.setFocusPolicy(Qt.NoFocus)
        next_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        next_btn.clicked.connect(lambda _checked=False: self.swap_next())

        layout.addWidget(prev_btn)
        layout.addWidget(combo, stretch=1)
        layout.addWidget(next_btn)

        # Pale-blue palette echoing the Workflow group bg so the nav
        # row reads as "header above the workflow column" rather than
        # a stranded widget.
        row.setAutoFillBackground(True)
        pal = row.palette()
        pal.setColor(row.backgroundRole(), QColor("#cfdef3"))
        pal.setColor(row.foregroundRole(), QColor("#2c3e50"))
        row.setPalette(pal)
        # Hairline separator below so the row isn't visually fused
        # with the Workflow buttons.
        sep = QFrame(col.host)
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)

        col.outer_layout.insertWidget(0, row)
        col.outer_layout.insertWidget(1, sep)

        self._nav_widget = row
        self._nav_prev_btn = prev_btn
        self._nav_next_btn = next_btn
        self._nav_combo = combo

    def _on_nav_combo_activated(self, index: int) -> None:
        """User-triggered dropdown selection -> swap to that bundle.

        On a rejected swap (out-of-bounds, hydration-failed) we
        re-sync the combo back to the still-active index so the
        visible selection matches reality.
        """
        if index == self._active_index:
            return
        ok = self.swap_to(index)
        if not ok:
            self._refresh_nav_buttons()

    def _add_video_nav_key_bindings(self) -> None:
        """Register ``Alt+Left`` / ``Alt+Right`` for previous / next
        video. Verified unbound in dnav core key bindings -- bare
        arrows are taken for frame nav.
        """
        try:
            self.add_key_binding(
                "alt+left", self.swap_prev,
                "Previous video", group="0. Video navigation",
            )
            self.add_key_binding(
                "alt+right", self.swap_next,
                "Next video", group="0. Video navigation",
            )
        except Exception:  # noqa: BLE001 - older dnav signature / no method
            pass

    def _refresh_nav_buttons(self) -> None:
        """Sync the nav row's dropdown + enable states to
        ``self._bundles`` + ``self._active_index``. Idempotent;
        cheap; safe to call from any state-change site (swap, bundle
        init, bg-hydration progress tick).
        """
        if self._nav_widget is None:
            return
        n = max(len(self._bundles), 1)
        i = self._active_index
        combo = getattr(self, "_nav_combo", None)
        if combo is not None:
            self._sync_nav_combo(combo, n=n, active=i)
        if self._nav_prev_btn is not None:
            self._nav_prev_btn.setEnabled(i > 0)
        if self._nav_next_btn is not None:
            self._nav_next_btn.setEnabled(i < n - 1)

    @staticmethod
    def _format_nav_combo_item(bundle, idx: int) -> str:
        """Format one dropdown row as ``"i. <stem>"`` with a trailing
        marker for non-ready bundles. Exposed on the class (not as a
        free function) so the corresponding swap_to tests can target
        it directly without importing more module-level surface area.
        """
        stem = Path(bundle.fname).stem
        label = f"{idx + 1}. {stem}"
        state = bundle.hydration_state
        if state == HYDRATION_HYDRATING or state == HYDRATION_PENDING:
            return f"{label}  …"
        if state == HYDRATION_FAILED:
            return f"{label}  ✗"
        return label

    def _sync_nav_combo(self, combo, *, n: int, active: int) -> None:
        """Bring the dropdown's items + selection + tooltips in line
        with the current bundle list.

        Programmatic mutations are wrapped in ``blockSignals`` so the
        ``activated`` connection (user-only) is never re-entered from
        this path. When the bundle identity list is unchanged, only
        per-item suffixes + tooltips + the active selection are
        touched -- a hot path during bg-hydration progress ticks.
        """
        try:
            from qtpy.QtCore import Qt
            tooltip_role = Qt.ToolTipRole
        except Exception:  # noqa: BLE001 -- no qtpy in this env
            tooltip_role = 3  # Qt::ToolTipRole

        bundles = self._bundles
        # Snapshot the fname list so a count-only check below is
        # robust to in-place mutations of self._bundles.
        fnames = [str(b.fname) for b in bundles]
        signature = tuple(fnames)
        prior_signature = getattr(self, "_nav_combo_signature", None)

        combo.blockSignals(True)
        try:
            if signature != prior_signature:
                combo.clear()
                for j, b in enumerate(bundles):
                    combo.addItem(self._format_nav_combo_item(b, j))
                    combo.setItemData(j, fnames[j], tooltip_role)
                if not bundles:
                    # Placeholder for the (rare) zero-bundle stub
                    # state so the widget isn't empty.
                    combo.addItem("(no videos)")
                self._nav_combo_signature = signature
            else:
                # Same bundles, possibly different hydration states.
                for j, b in enumerate(bundles):
                    text = self._format_nav_combo_item(b, j)
                    if combo.itemText(j) != text:
                        combo.setItemText(j, text)
                    combo.setItemData(j, fnames[j], tooltip_role)
            if bundles:
                clamped = max(0, min(active, len(bundles) - 1))
                if combo.currentIndex() != clamped:
                    combo.setCurrentIndex(clamped)
                # Combo's own hover tooltip: full path of the
                # currently-displayed video.
                combo.setToolTip(fnames[clamped])
            else:
                combo.setToolTip("")
        finally:
            combo.blockSignals(False)

# Bind ``builtins.open`` under a private alias inside this module so
# the module-level ``def open(...)`` below doesn't shadow it for the
# few sites that still need to open a file handle directly (notably
# the per-bundle VideoReader construction in
# :meth:`DUSTrack._hydrate_bundle_sync`).
import builtins as _builtins
builtins_open = _builtins.open
