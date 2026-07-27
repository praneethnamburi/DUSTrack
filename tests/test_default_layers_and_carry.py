"""Tests for two multi-video review conveniences:

* ``default_layers`` -- opt-in per-name default statevar selections applied
  to every freshly-hydrated bundle (e.g. land each video on M00 active +
  outliers overlay). Exercised through
  :func:`dustrack._bundle.derive_initial_bundle_selections`.

* the ``carry`` swap -- Ctrl+Alt+Right/Left carry the current video's
  layer/overlay/point into the next/prev video wherever those names exist.
  Exercised through :func:`dustrack._bundle_swap._merge_carry_selections`.

Both use light stubs so the real logic runs without a Qt GUI.
"""
from __future__ import annotations

from dustrack._bundle import derive_initial_bundle_selections
from dustrack._bundle_swap import _merge_carry_selections


class _Ann:
    def __init__(self, name, labels):
        self.name = name
        self.labels = labels


class _Container:
    def __init__(self, anns):
        self._d = {a.name: a for a in anns}

    @property
    def names(self):
        return list(self._d)

    def __getitem__(self, k):
        return self._d[k]


class _SV:
    def __init__(self, state):
        self.current_state = state


class _SVs:
    def __init__(self, d):
        self._d = d

    @property
    def names(self):
        return list(self._d)

    def __getitem__(self, k):
        return self._d[k]


class _Shell:
    def __init__(self, default_layers):
        self.default_layers = default_layers
        self.statevariables = _SVs({"number_keys": _SV("select")})


def _derive(default_layers, anns):
    return derive_initial_bundle_selections(_Shell(default_layers), _Container(anns))


# ---- Part 1: default_layers ----------------------------------------------

def test_default_layers_override_active_and_overlay():
    anns = [_Ann("M00", ["0", "1"]), _Ann("outliers", ["0", "1"]), _Ann("iteration-0", ["0", "1"])]
    sel = _derive({"annotation_layer": "M00", "annotation_overlay": "outliers"}, anns)
    assert sel["annotation_layer"] == "M00"
    assert sel["annotation_overlay"] == "outliers"


def test_default_layers_absent_name_falls_back_to_derived():
    anns = [_Ann("M00", ["0", "1"]), _Ann("iteration-0", ["0", "1"])]
    sel = _derive({"annotation_layer": "ghost"}, anns)
    assert sel["annotation_layer"] == "iteration-0"      # last manual = derived default


def test_default_label_override_drags_label_range():
    anns = [_Ann("M00", ["0", "1"])]
    sel = _derive({"annotation_layer": "M00", "annotation_label": "1"}, anns)
    assert sel["annotation_label"] == "1"
    assert sel["label_range"] == "0-9"


def test_no_defaults_uses_derived():
    anns = [_Ann("M00", ["0", "1"]), _Ann("iteration-0", ["0", "1"])]
    sel = _derive({}, anns)
    assert sel["annotation_layer"] == "iteration-0"
    assert sel["annotation_overlay"] is None


# ---- Part 2: carry swap ---------------------------------------------------

class _Target:
    def __init__(self, anns, selections):
        self.annotations = _Container(anns)
        self.selections = selections


def test_carry_applies_when_names_present():
    t = _Target([_Ann("M00", ["0", "1"]), _Ann("M2", ["0", "1"])],
                {"annotation_layer": "M00", "annotation_overlay": None, "annotation_label": "0"})
    merged = _merge_carry_selections(
        t, {"annotation_layer": "M2", "annotation_overlay": "M00", "annotation_label": "1"})
    assert merged["annotation_layer"] == "M2"
    assert merged["annotation_overlay"] == "M00"
    assert merged["annotation_label"] == "1"
    assert merged["label_range"] == "0-9"


def test_carry_absent_layer_keeps_target_default():
    t = _Target([_Ann("M00", ["0", "1"])],
                {"annotation_layer": "M00", "annotation_overlay": None, "annotation_label": "0"})
    merged = _merge_carry_selections(t, {"annotation_layer": "M2"})   # M2 not here
    assert merged["annotation_layer"] == "M00"


def test_carry_label_guarded_by_active_layer_labels():
    t = _Target([_Ann("M00", ["0", "1"])],
                {"annotation_layer": "M00", "annotation_label": "0"})
    merged = _merge_carry_selections(t, {"annotation_label": "5"})     # 5 not a label
    assert merged["annotation_label"] == "0"


def test_carry_overlay_none_is_honored():
    t = _Target([_Ann("M00", ["0", "1"])],
                {"annotation_layer": "M00", "annotation_overlay": "M00"})
    merged = _merge_carry_selections(t, {"annotation_overlay": None})
    assert merged["annotation_overlay"] is None
