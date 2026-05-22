"""Tests for :meth:`DUSTrack.swap_to` + related multi-video helpers.

Pure-logic tests; the real DUSTrack constructor needs Qt + a video,
which is out of scope for unit tests. We invoke the swap methods
directly via ``D._method(stub, ...)`` against a hand-rolled stub that
mimics the shell-attribute surface the methods touch. Integration with
the real GUI is manual smoke (see plans/ archive for the session
notes).

Coverage:

- :meth:`swap_to` bounds + no-op + non-ready-bundle behavior.
- :meth:`_snapshot_active_bundle` round-trips per-video state.
- Statevar restoration uses the silent-bypass pattern (no on_change
  fan-out during the restore).
- Save-on-close multi-bundle sweep.
- Nav button refresh logic.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dustrack._bundle import (
    HYDRATION_FAILED,
    HYDRATION_PENDING,
    HYDRATION_READY,
    _BundleState,
)


# ---------------------------------------------------------------------
# Stub builders
# ---------------------------------------------------------------------


def _make_annotation(name, fname=None, labels=None):
    """Minimal VideoAnnotation duck-type: just the attrs swap_to /
    sweep / hide / show reach."""
    a = SimpleNamespace()
    a.name = name
    a.fname = str(fname) if fname else None
    a.labels = labels or ["0"]
    a.data = {label: {} for label in a.labels}
    a.hidden = False
    a.shown = False

    def hide(draw=False):
        a.hidden = True
    def show(draw=False):
        a.shown = True
    def save():
        a.saved = True

    a.hide = hide
    a.show = show
    a.save = save
    a.saved = False
    return a


class _StubAnnotationsContainer:
    """Tiny stand-in for VideoAnnotations: __getitem__ + names + _list."""

    def __init__(self, annotations):
        self._list = list(annotations)

    @property
    def names(self):
        return [a.name for a in self._list]

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._list[key]
        for a in self._list:
            if a.name == key:
                return a
        raise KeyError(key)

    def __contains__(self, name):
        return name in self.names


def _make_annotations_container(annotations):
    return _StubAnnotationsContainer(annotations)


class _StubStateVar:
    def __init__(self, name, states, initial=None):
        self.name = name
        self.states = list(states)
        self._current_state_idx = 0
        if initial is not None and initial in self.states:
            self._current_state_idx = self.states.index(initial)
        self.callbacks = []
        self.fired = 0

    @property
    def current_state(self):
        return self.states[self._current_state_idx]

    def set_state(self, value):
        if isinstance(value, int):
            self._current_state_idx = value
        else:
            self._current_state_idx = self.states.index(value)
        self._notify()

    def _notify(self):
        self.fired += 1
        for cb in self.callbacks:
            cb()

    def add_on_change(self, cb):
        self.callbacks.append(cb)


class _StubStateVars:
    def __init__(self, vars_):
        self._vars = {sv.name: sv for sv in vars_}
        self.names = list(self._vars)
        self._text = SimpleNamespace(update=lambda: None)

    def __getitem__(self, key):
        return self._vars[key]

    def __contains__(self, key):
        return key in self._vars


class _StubReader:
    """Mimics a VideoReader well enough for swap_to: only ``__len__``
    is consulted (for trace-axis xlim rescale)."""

    def __init__(self, n_frames=100):
        self.n_frames = n_frames

    def __len__(self):
        return self.n_frames


def _make_bundle(fname, video_index, annotations, selections=None,
                 current_idx=0, ax_lims=None, ready=True, n_frames=100):
    state = HYDRATION_READY if ready else HYDRATION_PENDING
    b = _BundleState(
        fname=Path(fname), video_index=video_index,
        hydration_state=state,
    )
    b.reader = _StubReader(n_frames=n_frames)
    b.annotations = _make_annotations_container(annotations)
    b.current_idx = current_idx
    b.selections = selections or {}
    if ax_lims is not None:
        b.ax_lims = ax_lims
    return b


class _StubTraceAxis:
    """Tiny stand-in for an mpl Axes: set_xlim / set_ylim /
    set_autoscale*_on + matching getters for the snapshot/restore
    flow."""

    def __init__(self):
        self.xlim = (0, 100)
        self.ylim = (0, 1)
        self.autoscalex_on = True
        self.autoscaley_on = True

    def get_xlim(self):
        return self.xlim

    def get_ylim(self):
        return self.ylim

    def set_xlim(self, *args):
        if len(args) == 1:
            self.xlim = tuple(args[0])
        else:
            self.xlim = tuple(args)
        self.autoscalex_on = False

    def set_ylim(self, *args):
        if len(args) == 1:
            self.ylim = tuple(args[0])
        else:
            self.ylim = tuple(args)
        self.autoscaley_on = False

    def set_autoscalex_on(self, value):
        self.autoscalex_on = bool(value)

    def set_autoscaley_on(self, value):
        self.autoscaley_on = bool(value)


# Internal helpers we bind onto the test stub so swap_to's `self.method`
# dispatch resolves. Class attributes (statevar tuples) get aliased too.
_BOUND_DUSTRACK_METHODS = (
    "swap_to", "swap_prev", "swap_next",
    "_snapshot_active_bundle", "_attach_bundle",
    "_park_bundle_artists", "_show_bundle_artists",
    "_capture_statevar_selections", "_restore_statevar_selections",
    "_get_image_view_state", "_set_image_view_state",
    "_refresh_nav_buttons", "_notify_bundle_failure",
    "_sync_nav_combo",
)
# Staticmethods on DUSTrack -- attach to the stub as bare functions
# (no ``__get__(shell)`` binding, which would inject ``shell`` as the
# first positional arg).
_STATIC_DUSTRACK_METHODS = (
    "_format_nav_combo_item",
)


def _make_stub_shell(bundles, active_index=0, statevars=None):
    """Build the stub shell the swap methods bind to."""
    from dustrack.dlcinterface import DUSTrack as D

    active = bundles[active_index]
    shell = SimpleNamespace(
        _bundles=bundles,
        _active_index=active_index,
        fname=str(active.fname),
        data=active.reader,
        annotations=active.annotations,
        _current_idx=active.current_idx,
        _ax_lims=dict(active.ax_lims),
        frames_of_interest=list(active.frames_of_interest),
        _fast_render=False,
        _ax_image=None,
        _ax_trace_x=_StubTraceAxis(),
        _ax_trace_y=_StubTraceAxis(),
        _frame_marker_cache=None,
        _image_pane=None,
        _nav_widget=None,
        _nav_prev_btn=None,
        _nav_next_btn=None,
        _nav_combo=None,
        _nav_combo_signature=None,
        statevariables=statevars or _StubStateVars([]),
        _ALL_TRACKED_STATEVARS=D._ALL_TRACKED_STATEVARS,
        _BROADCAST_STATEVARS=D._BROADCAST_STATEVARS,
    )
    # Helper noops so the methods don't crash on these stubs.
    shell.update = lambda: None
    shell.update_annotation_label_states = lambda: None
    shell._refresh_workflow_button_state = lambda: None
    # Bind the DUSTrack methods onto the stub so ``self.method``
    # dispatch resolves to the real implementation when we call
    # ``D.swap_to(stub, ...)``. Without this, the recursive
    # ``self._snapshot_active_bundle()`` calls inside swap_to fail
    # with AttributeError on the SimpleNamespace.
    for name in _BOUND_DUSTRACK_METHODS:
        setattr(shell, name, getattr(D, name).__get__(shell))
    for name in _STATIC_DUSTRACK_METHODS:
        setattr(shell, name, getattr(D, name))
    # Trace view-state snapshot/restore methods (1.2.0a3 follow-up:
    # per-bundle trace axes pan/zoom preservation).
    for name in ("_get_trace_view_state", "_set_trace_view_state"):
        setattr(shell, name, getattr(D, name).__get__(shell))
    # Enhance-state snapshot/restore (1.2.0a3 follow-up: per-bundle
    # CLAHE / gamma / brightness).
    for name in ("_get_enhance_state", "_set_enhance_state"):
        setattr(shell, name, getattr(D, name).__get__(shell))
    shell._clahe_clip = 1.0
    shell._gamma = 1.0
    shell._brightness = 0.0
    shell._enhance_widget = None
    return shell


# ---------------------------------------------------------------------
# Bounds / dispatch
# ---------------------------------------------------------------------


class TestSwapToBounds:
    def test_swap_to_current_index_is_noop(self):
        from dustrack.dlcinterface import DUSTrack as D

        bundles = [
            _make_bundle("/v0.mp4", 0, [_make_annotation("layer0")]),
            _make_bundle("/v1.mp4", 1, [_make_annotation("layer1")]),
        ]
        shell = _make_stub_shell(bundles, active_index=1)
        # Already on index 1 -- swap_to(1) returns True without doing
        # any work.
        assert D.swap_to(shell, 1) is True
        # Active didn't change; nothing was hidden / shown.
        assert shell._active_index == 1
        for ann in bundles[0].annotations._list:
            assert ann.hidden is False
            assert ann.shown is False

    def test_swap_to_out_of_bounds_returns_false(self):
        from dustrack.dlcinterface import DUSTrack as D

        bundles = [_make_bundle("/v0.mp4", 0, [_make_annotation("a")])]
        shell = _make_stub_shell(bundles)
        assert D.swap_to(shell, 5) is False
        assert D.swap_to(shell, -1) is False

    def test_swap_to_pending_bundle_waits_until_failed(self, monkeypatch):
        """When the user clicks ▶ to a still-hydrating bundle, swap_to
        blocks on ``_await_hydration`` which pumps the Qt loop until
        the bundle reaches a terminal state. If hydration eventually
        fails, swap_to surfaces the failure and returns False."""
        from dustrack.dlcinterface import DUSTrack as D

        b0 = _make_bundle("/v0.mp4", 0, [_make_annotation("a")])
        b1 = _BundleState(fname=Path("/v1.mp4"), video_index=1,
                          hydration_state=HYDRATION_PENDING)
        shell = _make_stub_shell([b0, b1])

        # Patch _await_hydration to simulate a worker flipping b1 to
        # FAILED. The real method would pump events until the bundle
        # is terminal; the patch short-circuits to keep the test fast.
        def _fake_await(bundle):
            bundle.hydration_state = HYDRATION_FAILED
            bundle.hydration_error = "ValueError: synthetic failure"
            return False
        shell._await_hydration = _fake_await

        assert D.swap_to(shell, 1) is False
        assert b1.hydration_state == HYDRATION_FAILED
        # Active didn't change.
        assert shell._active_index == 0

    def test_swap_to_pending_bundle_completes_when_ready(self, monkeypatch):
        """Symmetric to the failure case: if the bg worker finishes
        hydration mid-await, swap_to proceeds normally."""
        from dustrack.dlcinterface import DUSTrack as D

        a0 = _make_annotation("a")
        a1 = _make_annotation("b")
        b0 = _make_bundle("/v0.mp4", 0, [a0],
                          selections={"annotation_layer": "a",
                                      "annotation_overlay": None,
                                      "annotation_label": "0",
                                      "label_range": "0-9",
                                      "number_keys": "select"})
        # Build b1 with READY-shape state but flip the state to
        # PENDING so swap_to triggers _await_hydration. The patched
        # await then flips it back to READY (mimicking what the bg
        # worker would do).
        b1 = _make_bundle("/v1.mp4", 1, [a1],
                          selections={"annotation_layer": "b",
                                      "annotation_overlay": None,
                                      "annotation_label": "0",
                                      "label_range": "0-9",
                                      "number_keys": "select"})
        b1.hydration_state = HYDRATION_PENDING
        sv = _StubStateVars([
            _StubStateVar("annotation_layer", ["a"], initial="a"),
            _StubStateVar("annotation_overlay", [None, "a"], initial=None),
            _StubStateVar("annotation_label", ["0"], initial="0"),
            _StubStateVar("label_range", ["0-9"], initial="0-9"),
            _StubStateVar("number_keys", ["select", "place"], initial="select"),
        ])
        shell = _make_stub_shell([b0, b1], statevars=sv)

        def _fake_await(bundle):
            bundle.hydration_state = HYDRATION_READY
            return True
        shell._await_hydration = _fake_await

        assert D.swap_to(shell, 1) is True
        assert shell._active_index == 1

    def test_swap_to_failed_bundle_returns_false(self):
        from dustrack.dlcinterface import DUSTrack as D

        b0 = _make_bundle("/v0.mp4", 0, [_make_annotation("a")])
        b1 = _BundleState(
            fname=Path("/v1.mp4"), video_index=1,
            hydration_state=HYDRATION_FAILED,
            hydration_error="ValueError: file missing",
        )
        shell = _make_stub_shell([b0, b1])
        assert D.swap_to(shell, 1) is False


# ---------------------------------------------------------------------
# Successful swap: snapshot, attach, park/show, statevar restore
# ---------------------------------------------------------------------


class TestSwapToBehavior:
    def test_successful_swap_rebinds_shell(self):
        from dustrack.dlcinterface import DUSTrack as D

        a0 = _make_annotation("manual0", fname="/v0_annotations_manual.json")
        a1 = _make_annotation("manual1", fname="/v1_annotations_manual.json")
        b0 = _make_bundle("/v0.mp4", 0, [a0],
                          selections={"annotation_layer": "manual0",
                                      "annotation_overlay": None,
                                      "annotation_label": "0",
                                      "label_range": "0-9",
                                      "number_keys": "select"},
                          current_idx=42)
        b1 = _make_bundle("/v1.mp4", 1, [a1],
                          selections={"annotation_layer": "manual1",
                                      "annotation_overlay": None,
                                      "annotation_label": "0",
                                      "label_range": "0-9",
                                      "number_keys": "select"},
                          current_idx=123)
        sv = _StubStateVars([
            _StubStateVar("annotation_layer", ["manual0"], initial="manual0"),
            _StubStateVar("annotation_overlay", [None, "manual0"], initial=None),
            _StubStateVar("annotation_label", ["0"], initial="0"),
            _StubStateVar("label_range", ["0-9"], initial="0-9"),
            _StubStateVar("number_keys", ["select", "place"], initial="select"),
        ])
        shell = _make_stub_shell([b0, b1], active_index=0, statevars=sv)
        shell._current_idx = 42  # active state matches bundle 0

        # Track on_change calls -- the silent-bypass restore must NOT
        # fire them.
        for v in sv._vars.values():
            v.fired = 0

        assert D.swap_to(shell, 1) is True
        assert shell._active_index == 1
        # Shell attrs rebind to b1.
        assert Path(shell.fname) == Path("/v1.mp4")
        assert shell.annotations is b1.annotations
        assert shell._current_idx == 123
        # b0's artists hidden, b1's shown.
        assert a0.hidden is True
        assert a1.shown is True
        # No on_change callback fired during the silent restore.
        for sv_name, v in sv._vars.items():
            assert v.fired == 0, f"{sv_name} fired {v.fired} time(s)"

    def test_round_trip_swap_preserves_per_bundle_state(self):
        from dustrack.dlcinterface import DUSTrack as D

        a0 = _make_annotation("layer0")
        a1 = _make_annotation("layer1")
        b0 = _make_bundle("/v0.mp4", 0, [a0], current_idx=42)
        b1 = _make_bundle("/v1.mp4", 1, [a1], current_idx=123)
        sv = _StubStateVars([
            _StubStateVar("annotation_layer", ["layer0"], initial="layer0"),
            _StubStateVar("annotation_overlay", [None], initial=None),
            _StubStateVar("annotation_label", ["0"], initial="0"),
            _StubStateVar("label_range", ["0-9"], initial="0-9"),
            _StubStateVar("number_keys", ["select", "place"], initial="select"),
        ])
        b0.selections = {sv_name: sv[sv_name].current_state
                         for sv_name in sv._vars}
        b1.selections = {sv_name: sv[sv_name].current_state
                         for sv_name in sv._vars}
        # Pre-populate b1's selections to have annotation_layer="layer1"
        # (the value that exists in its rotation).
        b1.selections["annotation_layer"] = "layer1"

        shell = _make_stub_shell([b0, b1], active_index=0, statevars=sv)
        shell._current_idx = 42

        # User scrolls bundle 0 to frame 99 and freezes axes; this is
        # the kind of state the snapshot must preserve.
        shell._current_idx = 99
        shell._ax_lims["state"] = True
        shell._ax_lims["x"] = [10.0, 200.0]
        shell.frames_of_interest = [5, 17, 29]

        assert D.swap_to(shell, 1) is True
        # Bundle 0's snapshot captured the modified state.
        assert b0.current_idx == 99
        assert b0.ax_lims["state"] is True
        assert b0.ax_lims["x"] == [10.0, 200.0]
        assert b0.frames_of_interest == [5, 17, 29]

        # Swap back -- shell should land on the captured state.
        assert D.swap_to(shell, 0) is True
        assert shell._current_idx == 99
        assert shell._ax_lims["state"] is True
        assert shell._ax_lims["x"] == [10.0, 200.0]
        assert shell.frames_of_interest == [5, 17, 29]


# ---------------------------------------------------------------------
# Nav button refresh
# ---------------------------------------------------------------------


class _StubCombo:
    """In-memory QComboBox stand-in -- mirrors just enough of the
    QComboBox API for :meth:`DUSTrack._sync_nav_combo` to drive it.
    """

    def __init__(self):
        self.items = []  # list of [text, tooltip]
        self._current = 0
        self.tooltip = ""
        self._blocked = False

    def blockSignals(self, flag):
        prev = self._blocked
        self._blocked = bool(flag)
        return prev

    def clear(self):
        self.items = []
        self._current = 0

    def addItem(self, text):
        self.items.append([text, None])

    def count(self):
        return len(self.items)

    def itemText(self, idx):
        return self.items[idx][0]

    def setItemText(self, idx, text):
        self.items[idx][0] = text

    def setItemData(self, idx, value, role):  # role ignored in the stub
        self.items[idx][1] = value

    def itemData(self, idx, role=None):
        return self.items[idx][1]

    def currentIndex(self):
        return self._current

    def setCurrentIndex(self, idx):
        self._current = idx

    def setToolTip(self, text):
        self.tooltip = text


class TestRefreshNavButtons:
    def test_nav_combo_format_single_video(self):
        from dustrack.dlcinterface import DUSTrack as D

        b0 = _make_bundle("/v0.mp4", 0, [_make_annotation("a")])
        shell = _make_stub_shell([b0])
        combo = _StubCombo()
        prev_btn = SimpleNamespace(enabled=None,
                                   setEnabled=lambda v: setattr(prev_btn, "enabled", v))
        next_btn = SimpleNamespace(enabled=None,
                                   setEnabled=lambda v: setattr(next_btn, "enabled", v))
        shell._nav_widget = object()
        shell._nav_combo = combo
        shell._nav_combo_signature = None
        shell._nav_prev_btn = prev_btn
        shell._nav_next_btn = next_btn

        D._refresh_nav_buttons(shell)
        assert combo.count() == 1
        assert combo.itemText(0) == "1. v0"
        assert combo.itemData(0) == str(Path("/v0.mp4"))
        assert combo.tooltip == str(Path("/v0.mp4"))
        assert combo.currentIndex() == 0
        assert prev_btn.enabled is False
        assert next_btn.enabled is False

    def test_nav_combo_format_multi_video_all_ready(self):
        from dustrack.dlcinterface import DUSTrack as D

        bundles = [_make_bundle(f"/dir/v{i}.mp4", i, [_make_annotation(f"a{i}")])
                   for i in range(5)]
        shell = _make_stub_shell(bundles, active_index=2)
        combo = _StubCombo()
        prev_btn = SimpleNamespace(enabled=None,
                                   setEnabled=lambda v: setattr(prev_btn, "enabled", v))
        next_btn = SimpleNamespace(enabled=None,
                                   setEnabled=lambda v: setattr(next_btn, "enabled", v))
        shell._nav_widget = object()
        shell._nav_combo = combo
        shell._nav_combo_signature = None
        shell._nav_prev_btn = prev_btn
        shell._nav_next_btn = next_btn

        D._refresh_nav_buttons(shell)
        assert combo.count() == 5
        for i in range(5):
            assert combo.itemText(i) == f"{i + 1}. v{i}"
            assert combo.itemData(i) == str(Path(f"/dir/v{i}.mp4"))
        assert combo.currentIndex() == 2
        assert combo.tooltip == str(Path("/dir/v2.mp4"))
        assert prev_btn.enabled is True
        assert next_btn.enabled is True

    def test_nav_combo_format_partial_hydration_marks_pending(self):
        from dustrack.dlcinterface import DUSTrack as D

        ready_bundle = _make_bundle("/v0.mp4", 0, [_make_annotation("a")])
        pending = _BundleState(fname=Path("/v1.mp4"), video_index=1,
                               hydration_state=HYDRATION_PENDING)
        bundles = [ready_bundle, pending]
        shell = _make_stub_shell(bundles)
        combo = _StubCombo()
        prev_btn = SimpleNamespace(enabled=None,
                                   setEnabled=lambda v: setattr(prev_btn, "enabled", v))
        next_btn = SimpleNamespace(enabled=None,
                                   setEnabled=lambda v: setattr(next_btn, "enabled", v))
        shell._nav_widget = object()
        shell._nav_combo = combo
        shell._nav_combo_signature = None
        shell._nav_prev_btn = prev_btn
        shell._nav_next_btn = next_btn

        D._refresh_nav_buttons(shell)
        assert combo.itemText(0) == "1. v0"
        assert combo.itemText(1) == "2. v1  …"

    def test_nav_combo_marks_failed_bundle(self):
        from dustrack.dlcinterface import DUSTrack as D

        ready_bundle = _make_bundle("/v0.mp4", 0, [_make_annotation("a")])
        failed = _BundleState(fname=Path("/v1.mp4"), video_index=1,
                              hydration_state=HYDRATION_FAILED,
                              hydration_error="boom")
        shell = _make_stub_shell([ready_bundle, failed])
        combo = _StubCombo()
        prev_btn = SimpleNamespace(enabled=None,
                                   setEnabled=lambda v: setattr(prev_btn, "enabled", v))
        next_btn = SimpleNamespace(enabled=None,
                                   setEnabled=lambda v: setattr(next_btn, "enabled", v))
        shell._nav_widget = object()
        shell._nav_combo = combo
        shell._nav_combo_signature = None
        shell._nav_prev_btn = prev_btn
        shell._nav_next_btn = next_btn

        D._refresh_nav_buttons(shell)
        assert combo.itemText(1) == "2. v1  ✗"

    def test_nav_combo_resync_after_hydration_progress_does_not_rebuild(self):
        """When a pending bundle flips to ready, the per-item suffix
        updates but the items list isn't cleared and rebuilt -- the
        signature-based fast path is taken."""
        from dustrack.dlcinterface import DUSTrack as D

        ready_bundle = _make_bundle("/v0.mp4", 0, [_make_annotation("a")])
        pending = _BundleState(fname=Path("/v1.mp4"), video_index=1,
                               hydration_state=HYDRATION_PENDING)
        shell = _make_stub_shell([ready_bundle, pending])
        combo = _StubCombo()
        prev_btn = SimpleNamespace(enabled=None,
                                   setEnabled=lambda v: setattr(prev_btn, "enabled", v))
        next_btn = SimpleNamespace(enabled=None,
                                   setEnabled=lambda v: setattr(next_btn, "enabled", v))
        shell._nav_widget = object()
        shell._nav_combo = combo
        shell._nav_combo_signature = None
        shell._nav_prev_btn = prev_btn
        shell._nav_next_btn = next_btn

        D._refresh_nav_buttons(shell)
        # The id of the inner item list mirrors the QComboBox model;
        # the fast path mutates entries in place, the slow path
        # rebuilds.
        first_pass_items = combo.items
        # Bundle 1 now reports ready -- flip it without changing the
        # fnames list.
        pending.hydration_state = HYDRATION_READY
        D._refresh_nav_buttons(shell)
        assert combo.items is first_pass_items  # same list object
        assert combo.itemText(1) == "2. v1"

    def test_no_nav_widget_is_silent_noop(self):
        from dustrack.dlcinterface import DUSTrack as D

        shell = _make_stub_shell([_make_bundle("/v0.mp4", 0, [])])
        shell._nav_widget = None
        # Must not raise.
        D._refresh_nav_buttons(shell)


# ---------------------------------------------------------------------
# Statevar capture / restore
# ---------------------------------------------------------------------


class TestStatevarRoundTrip:
    def test_capture_returns_current_state_values(self):
        from dustrack.dlcinterface import DUSTrack as D

        sv = _StubStateVars([
            _StubStateVar("annotation_layer", ["x", "y"], initial="y"),
            _StubStateVar("annotation_label", ["0", "1"], initial="1"),
        ])
        shell = SimpleNamespace(
            statevariables=sv,
            _ALL_TRACKED_STATEVARS=D._ALL_TRACKED_STATEVARS,
        )
        out = D._capture_statevar_selections(shell)
        assert out == {"annotation_layer": "y", "annotation_label": "1"}

    def test_restore_uses_silent_bypass(self):
        from dustrack.dlcinterface import DUSTrack as D

        sv = _StubStateVars([
            _StubStateVar("annotation_layer", ["a"], initial="a"),
            _StubStateVar("annotation_overlay", [None, "a"], initial=None),
            _StubStateVar("annotation_label", ["0", "1", "2"], initial="0"),
            _StubStateVar("label_range", ["0-9"], initial="0-9"),
        ])
        for v in sv._vars.values():
            v.fired = 0
        shell = SimpleNamespace(
            statevariables=sv,
            _ALL_TRACKED_STATEVARS=D._ALL_TRACKED_STATEVARS,
            update_annotation_label_states=lambda: None,
        )
        D._restore_statevar_selections(
            shell,
            {"annotation_layer": "a", "annotation_overlay": None,
             "annotation_label": "2", "label_range": "0-9"},
            layer_names=["a"],
        )
        # Final state is what we asked for.
        assert sv["annotation_label"].current_state == "2"
        # No on_change callback fired (silent bypass).
        for sv_name, v in sv._vars.items():
            assert v.fired == 0, f"{sv_name} fired {v.fired} time(s)"

    def test_restore_clamps_invalid_selection_to_first_state(self):
        from dustrack.dlcinterface import DUSTrack as D

        sv = _StubStateVars([
            _StubStateVar("annotation_layer", ["new1", "new2"], initial="new1"),
            _StubStateVar("annotation_label", ["0"], initial="0"),
        ])
        shell = SimpleNamespace(
            statevariables=sv,
            _ALL_TRACKED_STATEVARS=D._ALL_TRACKED_STATEVARS,
            update_annotation_label_states=lambda: None,
        )
        D._restore_statevar_selections(
            shell,
            # "old_value" was valid on the previous bundle, but
            # ["new1", "new2"] is the new rotation -- doesn't contain
            # the value, so fall back to first state.
            {"annotation_layer": "old_value"},
            layer_names=["new1", "new2"],
        )
        assert sv["annotation_layer"].current_state == "new1"


# ---------------------------------------------------------------------
# Enhance-state first-visit defaults (1.2.0a3 follow-up 2026-05-22)
#
# Regression: pre-fix _set_enhance_state(None) returned early, so a
# first-visit swap to a bundle with no saved enhance_state inherited
# the leaving bundle's slider positions. The fix snapshots construction
# defaults into _initial_enhance_state in __init__ and restores those
# on first-visit. Returning visits still use the saved state.
# ---------------------------------------------------------------------


class TestEnhanceStateFirstVisitDefaults:
    def _make_shell_with_initial(self, initial):
        """Build a minimal shell stub that has ``_initial_enhance_state``
        + the un-bound ``_set_enhance_state`` method bound to it."""
        from dustrack.dlcinterface import DUSTrack as D
        shell = SimpleNamespace(
            _clahe_clip=99.0,  # something a default restore must overwrite
            _gamma=99.0,
            _brightness=99.0,
            _enhance_widget=None,
            _initial_enhance_state=initial,
        )
        shell._set_enhance_state = D._set_enhance_state.__get__(shell)
        return shell

    def test_first_visit_resets_to_construction_defaults(self):
        initial = {"clahe_clip": 1.0, "gamma": 1.0, "brightness": 0.0}
        shell = self._make_shell_with_initial(initial)
        # Simulate the leaving bundle leaving its slider positions on
        # the shell (1.5 / 2.0 / 5) before swap.
        shell._clahe_clip = 2.5
        shell._gamma = 1.5
        shell._brightness = 5.0
        # First-visit: state=None -> defaults restored.
        shell._set_enhance_state(None)
        assert shell._clahe_clip == 1.0
        assert shell._gamma == 1.0
        assert shell._brightness == 0.0

    def test_returning_visit_uses_saved_state_not_defaults(self):
        initial = {"clahe_clip": 1.0, "gamma": 1.0, "brightness": 0.0}
        shell = self._make_shell_with_initial(initial)
        saved = {"clahe_clip": 3.0, "gamma": 1.4, "brightness": 12.0}
        shell._set_enhance_state(saved)
        assert shell._clahe_clip == 3.0
        assert shell._gamma == pytest.approx(1.4)
        assert shell._brightness == 12.0

    def test_missing_initial_snapshot_is_safe_noop(self):
        from dustrack.dlcinterface import DUSTrack as D
        # Subclass calling out of order / construction snapshot missing.
        shell = SimpleNamespace(
            _clahe_clip=2.5,
            _gamma=1.5,
            _brightness=5.0,
            _enhance_widget=None,
        )
        shell._set_enhance_state = D._set_enhance_state.__get__(shell)
        # No _initial_enhance_state attr -> safe fallback: hold values.
        shell._set_enhance_state(None)
        assert shell._clahe_clip == 2.5
        assert shell._gamma == 1.5
        assert shell._brightness == 5.0
