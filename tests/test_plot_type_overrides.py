"""Tests for the opt-in ``plot_type_overrides`` display-style pin.

The override map lets a caller pin a layer's plot type by NAME
(``{"M00": "line", "outliers": "dot"}``) explicitly, winning over the
density-based default (:func:`_is_dense_layer_name`). It is applied on
the active bundle at construction and re-applied to every background
bundle as it hydrates, so the style survives multi-video swaps without
renaming layers to game the name convention.

These tests bind the real ``DUSTrack`` methods onto a light stub so the
application logic is exercised without spinning up a Qt GUI.
"""
from __future__ import annotations

from dustrack.dlcinterface import DUSTrack


class _FakeAnn:
    def __init__(self, name):
        self.name = name
        self.plot_calls = []

    def set_plot_type(self, type_, draw=True):
        self.plot_calls.append(type_)


class _FakeContainer:
    def __init__(self, anns):
        self._d = {a.name: a for a in anns}
        self._list = list(anns)

    @property
    def names(self):
        return list(self._d)

    def __iter__(self):
        return iter(self._list)

    def __getitem__(self, key):
        return self._d[key]


class _FakeSV:
    def __init__(self):
        self.state = None

    def set_state(self, s):
        self.state = s


class _Stub:
    """Bare object the real DUSTrack methods operate on."""

    # Bind the real implementations so the tests exercise actual logic
    # (mirrors the binding pattern in test_bundle_api.py).
    _apply_plot_type_overrides = DUSTrack._apply_plot_type_overrides
    _normalize_dlc_layer_display = DUSTrack._normalize_dlc_layer_display


def _stub(anns, overrides):
    t = _Stub()
    t.annotations = _FakeContainer(anns)
    t.plot_type_overrides = overrides
    t.statevariables = {"annotation_overlay": _FakeSV()}
    return t


def test_overrides_pin_style_by_name():
    anns = [_FakeAnn("M00"), _FakeAnn("outliers"), _FakeAnn("buffer")]
    t = _stub(anns, {"M00": "line", "outliers": "dot"})
    DUSTrack._apply_plot_type_overrides(t, draw=False)
    assert t.annotations["M00"].plot_calls == ["line"]
    assert t.annotations["outliers"].plot_calls == ["dot"]
    # A layer not in the map is left untouched.
    assert t.annotations["buffer"].plot_calls == []


def test_override_wins_over_density_default():
    # ``dlc_foo`` is dense -> the normalize pass sets it to a line; the
    # override then forces it to a dot, so the LAST applied style wins.
    anns = [_FakeAnn("dlc_foo"), _FakeAnn("M00")]
    t = _stub(anns, {"dlc_foo": "dot", "M00": "line"})
    DUSTrack._normalize_dlc_layer_display(t)
    assert t.annotations["dlc_foo"].plot_calls == ["line", "dot"]
    assert t.annotations["M00"].plot_calls[-1] == "line"


def test_empty_overrides_is_a_noop():
    anns = [_FakeAnn("M00")]
    t = _stub(anns, {})
    DUSTrack._apply_plot_type_overrides(t, draw=False)
    assert t.annotations["M00"].plot_calls == []


def test_invalid_value_and_missing_layer_skipped():
    anns = [_FakeAnn("M00")]
    t = _stub(anns, {"M00": "squiggle", "ghost": "line"})
    DUSTrack._apply_plot_type_overrides(t, draw=False)
    # invalid style value -> skipped; name not present -> skipped
    assert t.annotations["M00"].plot_calls == []


def test_scope_restricts_application():
    anns = [_FakeAnn("M00"), _FakeAnn("outliers")]
    t = _stub(anns, {"M00": "line", "outliers": "dot"})
    DUSTrack._apply_plot_type_overrides(t, names=["M00"], draw=False)
    assert t.annotations["M00"].plot_calls == ["line"]
    assert t.annotations["outliers"].plot_calls == []  # out of scope
