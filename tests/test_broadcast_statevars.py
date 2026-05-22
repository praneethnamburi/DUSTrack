"""Tests for the 1.2.0a3 broadcast-statevar hook + post-train
multi-bundle dlc-refresh.

Broadcast statevars (``annotation_label`` / ``label_range`` /
``number_keys``) propagate from the active bundle to every bundle's
``selections`` dict when the user mutates them via UI / keybinding.
Swap-in then restores per-bundle from these dicts, so the user's
"working on bodypart 1" carries across the queue without any
swap-time logic.

The post-train refresh is the multi-bundle analog of
``_refresh_dlc_layers``: after Train DLC writes ``<video>_DLC*.h5``
for every video in the project, the active bundle reloads via
``add_annotation_layers`` and the non-active ready bundles get the
same layers added (artists parked invisible).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dustrack._bundle import (
    HYDRATION_PENDING,
    HYDRATION_READY,
    _BundleState,
)


# Reuse the stub builders from test_swap_to so the test shapes line up.
from tests.test_swap_to import (
    _StubAnnotationsContainer,
    _StubStateVar,
    _StubStateVars,
    _make_annotation,
    _make_annotations_container,
    _make_bundle,
    _make_stub_shell,
)


# ---------------------------------------------------------------------
# Broadcast hook installation + propagation
# ---------------------------------------------------------------------


class TestBroadcastHooks:
    def test_install_wires_callbacks_on_broadcast_statevars_only(self):
        from dustrack.dlcinterface import DUSTrack as D

        sv = _StubStateVars([
            _StubStateVar("annotation_layer", ["a"], initial="a"),
            _StubStateVar("annotation_overlay", [None, "a"], initial=None),
            _StubStateVar("annotation_label", ["0"], initial="0"),
            _StubStateVar("label_range", ["0-9"], initial="0-9"),
            _StubStateVar("number_keys", ["select", "place"], initial="select"),
        ])
        shell = _make_stub_shell(
            [_make_bundle("/v0.mp4", 0, [_make_annotation("a")])],
            statevars=sv,
        )
        shell._BROADCAST_STATEVARS = D._BROADCAST_STATEVARS
        shell._install_broadcast_statevar_hooks = (
            D._install_broadcast_statevar_hooks.__get__(shell)
        )
        shell._broadcast_statevar = D._broadcast_statevar.__get__(shell)

        shell._install_broadcast_statevar_hooks()

        # Three broadcast statevars got exactly one callback each.
        for sv_name in D._BROADCAST_STATEVARS:
            assert len(sv._vars[sv_name].callbacks) == 1
        # Non-broadcast statevars (annotation_layer/_overlay) have no
        # callback registered by the broadcast hook (other code may
        # register their own, that's fine).
        assert len(sv._vars["annotation_layer"].callbacks) == 0
        assert len(sv._vars["annotation_overlay"].callbacks) == 0

    def test_install_is_idempotent(self):
        from dustrack.dlcinterface import DUSTrack as D

        sv = _StubStateVars([
            _StubStateVar("annotation_label", ["0", "1"], initial="0"),
            _StubStateVar("label_range", ["0-9"], initial="0-9"),
            _StubStateVar("number_keys", ["select"], initial="select"),
        ])
        shell = _make_stub_shell(
            [_make_bundle("/v0.mp4", 0, [_make_annotation("a")])],
            statevars=sv,
        )
        shell._BROADCAST_STATEVARS = D._BROADCAST_STATEVARS
        shell._install_broadcast_statevar_hooks = (
            D._install_broadcast_statevar_hooks.__get__(shell)
        )
        shell._broadcast_statevar = D._broadcast_statevar.__get__(shell)

        shell._install_broadcast_statevar_hooks()
        shell._install_broadcast_statevar_hooks()  # second call: no stacking

        for sv_name in D._BROADCAST_STATEVARS:
            assert len(sv._vars[sv_name].callbacks) == 1

    def test_broadcast_writes_to_every_bundle(self):
        """When the user toggles ``number_keys`` (the only broadcast
        statevar), every bundle's selections dict picks up the new
        value so swap-in restores it consistently."""
        from dustrack.dlcinterface import DUSTrack as D

        sv = _StubStateVars([
            _StubStateVar("annotation_label", ["0", "1", "2"], initial="0"),
            _StubStateVar("label_range", ["0-9"], initial="0-9"),
            _StubStateVar("number_keys", ["select", "place"], initial="select"),
        ])
        bundles = [
            _make_bundle(f"/v{i}.mp4", i, [_make_annotation(f"a{i}")],
                        selections={"annotation_label": "0",
                                    "label_range": "0-9",
                                    "number_keys": "select"})
            for i in range(3)
        ]
        shell = _make_stub_shell(bundles, statevars=sv)
        shell._BROADCAST_STATEVARS = D._BROADCAST_STATEVARS
        shell._install_broadcast_statevar_hooks = (
            D._install_broadcast_statevar_hooks.__get__(shell)
        )
        shell._broadcast_statevar = D._broadcast_statevar.__get__(shell)
        shell._install_broadcast_statevar_hooks()

        # User toggles number_keys -> on_change fires -> broadcast
        # writes to every bundle's selections.
        sv["number_keys"].set_state("place")
        for b in bundles:
            assert b.selections["number_keys"] == "place"

    def test_label_change_does_NOT_broadcast(self):
        """``annotation_label`` is per-bundle now (1.2.0a3 follow-up):
        switching the active label on bundle 0 must not change
        bundle 1's stored selection. Otherwise the user reports
        "set label 1 on video 1, swap to video 2, see label 1
        instead of label 0"."""
        from dustrack.dlcinterface import DUSTrack as D

        sv = _StubStateVars([
            _StubStateVar("annotation_label", ["0", "1", "2"], initial="0"),
            _StubStateVar("label_range", ["0-9"], initial="0-9"),
            _StubStateVar("number_keys", ["select", "place"], initial="select"),
        ])
        bundles = [
            _make_bundle(f"/v{i}.mp4", i, [_make_annotation(f"a{i}")],
                        selections={"annotation_label": "0"})
            for i in range(3)
        ]
        shell = _make_stub_shell(bundles, statevars=sv)
        shell._BROADCAST_STATEVARS = D._BROADCAST_STATEVARS
        shell._install_broadcast_statevar_hooks = (
            D._install_broadcast_statevar_hooks.__get__(shell)
        )
        shell._broadcast_statevar = D._broadcast_statevar.__get__(shell)
        shell._install_broadcast_statevar_hooks()

        # User changes annotation_label on bundle 0 -- bundles 1+
        # must NOT receive this change.
        sv["annotation_label"].set_state("2")
        # Bundle 0's selection in the dict is unchanged because
        # _snapshot_active_bundle hasn't run yet; the live shell
        # statevar is what's "2".
        assert bundles[1].selections["annotation_label"] == "0"
        assert bundles[2].selections["annotation_label"] == "0"

    def test_broadcast_does_not_fire_on_non_broadcast_statevar(self):
        from dustrack.dlcinterface import DUSTrack as D

        sv = _StubStateVars([
            _StubStateVar("annotation_layer", ["a", "b"], initial="a"),
            _StubStateVar("annotation_label", ["0"], initial="0"),
            _StubStateVar("label_range", ["0-9"], initial="0-9"),
            _StubStateVar("number_keys", ["select"], initial="select"),
        ])
        bundles = [
            _make_bundle("/v0.mp4", 0, [_make_annotation("a")],
                        selections={"annotation_layer": "a"}),
            _make_bundle("/v1.mp4", 1, [_make_annotation("b")],
                        selections={"annotation_layer": "b"}),
        ]
        shell = _make_stub_shell(bundles, statevars=sv)
        shell._BROADCAST_STATEVARS = D._BROADCAST_STATEVARS
        shell._install_broadcast_statevar_hooks = (
            D._install_broadcast_statevar_hooks.__get__(shell)
        )
        shell._broadcast_statevar = D._broadcast_statevar.__get__(shell)
        shell._install_broadcast_statevar_hooks()

        # Mutate annotation_layer (NOT a broadcast statevar). Bundle 1's
        # selection should NOT change to match -- per-bundle layers are
        # data-shaped and must stay independent.
        sv["annotation_layer"].set_state("b")
        # Bundle 0's selection was "a"; broadcast didn't fire; still "a".
        assert bundles[0].selections["annotation_layer"] == "a"
        # Bundle 1's selection unchanged too.
        assert bundles[1].selections["annotation_layer"] == "b"

    def test_pending_bundle_receives_broadcast(self):
        """A bundle that's still PENDING when the user mutates a
        broadcast statevar (``number_keys``) gets the new value in
        its ``selections`` dict. When it later hydrates, the value
        is preserved (not clobbered by the derived initial)."""
        from dustrack.dlcinterface import DUSTrack as D

        sv = _StubStateVars([
            _StubStateVar("annotation_label", ["0", "1"], initial="0"),
            _StubStateVar("label_range", ["0-9"], initial="0-9"),
            _StubStateVar("number_keys", ["select", "place"], initial="select"),
        ])
        b0 = _make_bundle("/v0.mp4", 0, [_make_annotation("a0")],
                          selections={"annotation_label": "0",
                                      "label_range": "0-9",
                                      "number_keys": "select"})
        b1 = _BundleState(fname=Path("/v1.mp4"), video_index=1,
                          hydration_state=HYDRATION_PENDING)
        shell = _make_stub_shell([b0, b1], statevars=sv)
        shell._BROADCAST_STATEVARS = D._BROADCAST_STATEVARS
        shell._install_broadcast_statevar_hooks = (
            D._install_broadcast_statevar_hooks.__get__(shell)
        )
        shell._broadcast_statevar = D._broadcast_statevar.__get__(shell)
        shell._install_broadcast_statevar_hooks()

        sv["number_keys"].set_state("place")
        # Pending bundle picks up the broadcast.
        assert b1.selections["number_keys"] == "place"


# ---------------------------------------------------------------------
# Post-train multi-bundle refresh
# ---------------------------------------------------------------------


class TestPostTrainRefresh:
    def test_refresh_skips_active_and_non_ready_bundles(self, monkeypatch):
        from dustrack.dlcinterface import DUSTrack as D

        b0 = _make_bundle("/v0.mp4", 0, [_make_annotation("a0")])
        b1 = _make_bundle("/v1.mp4", 1, [_make_annotation("a1")])
        b2 = _BundleState(fname=Path("/v2.mp4"), video_index=2,
                          hydration_state=HYDRATION_PENDING)
        bundles = [b0, b1, b2]

        project = SimpleNamespace(
            latest_iteration_is_trained=lambda: True,
            latest_iteration=1,
        )
        shell = _make_stub_shell(bundles, active_index=0)
        shell._dlcproject = project
        # Bind the methods.
        shell._refresh_dlc_layers_other_bundles = (
            D._refresh_dlc_layers_other_bundles.__get__(shell)
        )

        called = []
        def _fake_add(bundle, proj, suffix):
            called.append(bundle.video_index)
        shell._add_new_dlc_layers_to_bundle = _fake_add

        shell._refresh_dlc_layers_other_bundles()
        # Only the ready non-active bundle (b1) is processed.
        # b0 is active -> skipped. b2 is pending -> skipped.
        assert called == [1]

    def test_refresh_continues_after_per_bundle_exception(self, monkeypatch):
        from dustrack.dlcinterface import DUSTrack as D

        b0 = _make_bundle("/v0.mp4", 0, [_make_annotation("a0")])
        b1 = _make_bundle("/v1.mp4", 1, [_make_annotation("a1")])
        b2 = _make_bundle("/v2.mp4", 2, [_make_annotation("a2")])
        bundles = [b0, b1, b2]

        project = SimpleNamespace(
            latest_iteration_is_trained=lambda: True,
            latest_iteration=1,
        )
        shell = _make_stub_shell(bundles, active_index=0)
        shell._dlcproject = project
        shell._refresh_dlc_layers_other_bundles = (
            D._refresh_dlc_layers_other_bundles.__get__(shell)
        )

        called = []
        def _fake_add(bundle, proj, suffix):
            if bundle.video_index == 1:
                raise RuntimeError("synthetic h5 corruption on v1")
            called.append(bundle.video_index)
        shell._add_new_dlc_layers_to_bundle = _fake_add

        # Must not raise even with b1's exception.
        shell._refresh_dlc_layers_other_bundles()
        # b2 still processed.
        assert called == [2]

    def test_refresh_no_op_without_dlcproject(self):
        from dustrack.dlcinterface import DUSTrack as D

        bundles = [_make_bundle("/v0.mp4", 0, [_make_annotation("a")])]
        shell = _make_stub_shell(bundles)
        shell._dlcproject = None
        shell._refresh_dlc_layers_other_bundles = (
            D._refresh_dlc_layers_other_bundles.__get__(shell)
        )
        # Tracker that fake_add was never called.
        shell._add_new_dlc_layers_to_bundle = lambda *a, **k: pytest.fail(
            "must not be called when no DLC project is bound"
        )
        # Must not raise.
        shell._refresh_dlc_layers_other_bundles()

    def test_refresh_no_op_single_bundle(self):
        from dustrack.dlcinterface import DUSTrack as D

        bundles = [_make_bundle("/v0.mp4", 0, [_make_annotation("a")])]
        project = SimpleNamespace(
            latest_iteration_is_trained=lambda: True,
            latest_iteration=1,
        )
        shell = _make_stub_shell(bundles)
        shell._dlcproject = project
        shell._refresh_dlc_layers_other_bundles = (
            D._refresh_dlc_layers_other_bundles.__get__(shell)
        )
        shell._add_new_dlc_layers_to_bundle = lambda *a, **k: pytest.fail(
            "must not be called when only one bundle exists"
        )
        shell._refresh_dlc_layers_other_bundles()
