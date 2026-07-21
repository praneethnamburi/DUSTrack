"""Tests for :meth:`DUSTrack.predict_range_with_dlc` -- the ``z z e`` action.

Same approach as ``test_decimation.py``: the method touches a small,
well-defined slice of GUI state, so a ``SimpleNamespace`` fake driving
the unbound method is enough -- no GUI session, no model, no GPU.

The behaviours worth pinning here are the ones that would be silently
wrong rather than loudly broken:

* predictions land in the **overlay**, not the primary layer (the whole
  point is comparing old against new -- writing over the primary
  destroys the thing you asked to see);
* the scratch layer stays **in memory** (``extract_frames`` globs
  ``{video_stem}*_annotations*.json`` and filters only ``_dlccorr``, so
  a prediction JSON in the project videos folder would become training
  input for the next model);
* the cached model is **dropped after training**, since a stale
  predictor answers the exact question being asked with the previous
  weights and looks perfectly healthy doing it.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import dustrack
from dustrack import DUSTrack
from dustrack.predict import SCORER


def _prediction_df(frames, bodyparts=("point0", "point1")):
    """A DLC-format prediction frame, as ``RangePredictor`` returns."""
    cols = pd.MultiIndex.from_product(
        [[SCORER], list(bodyparts), ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    rows = []
    for f in frames:
        row = []
        for i, _ in enumerate(bodyparts):
            row += [float(f), float(f) + 100.0 * (i + 1), 0.9]
        rows.append(row)
    return pd.DataFrame(rows, columns=cols, index=pd.Index(list(frames), name="frame"))


class FakePredictor:
    """Stands in for RangePredictor -- records calls, returns a frame."""

    def __init__(self, frames=None, bodyparts=("point0", "point1")):
        self.frames = list(frames) if frames is not None else []
        self.bodyparts = list(bodyparts)
        self.snapshot_path = r"M:\models\proj\train\snapshot-best-270.pt"
        self.calls: list[tuple] = []
        self.closed = False

    def predict_range(self, video_path, start, end, annotation=True, **kw):
        self.calls.append((video_path, start, end, annotation))
        frames = self.frames or list(range(start, end + 1))
        return _prediction_df(frames, self.bodyparts)

    def close(self):
        self.closed = True


class FakeStateVar:
    def __init__(self):
        self.state = None
        self.calls: list = []

    def set_state(self, s):
        self.calls.append(s)
        self.state = s


class _FakeContainer(dict):
    """Minimal AssetContainer stand-in: dict + a ``names`` property."""

    @property
    def names(self):
        return list(self.keys())


def _make_fake(
    interval=(10, 14),
    predictor=None,
    dlcproject=SimpleNamespace(config_path="proj/config.yaml"),
    existing_layer=None,
):
    """Bind the action to a fake exposing only what it reads."""
    predictor = predictor if predictor is not None else FakePredictor()
    # The container itself is the shared store -- building it from a
    # plain dict would copy, and the add path would then write somewhere
    # the action never reads.
    annotations = _FakeContainer()
    if existing_layer is not None:
        annotations[DUSTrack.PREDICT_LAYER_NAME] = existing_layer

    added: list = []
    overlay = FakeStateVar()

    def _interval():
        if interval is None:
            raise KeyError("no interval selected")
        return interval

    fake = SimpleNamespace(
        _dlcproject=dlcproject,
        _range_predictor=predictor,
        data=SimpleNamespace(fname="video.mp4"),
        fname="video.mp4",
        get_selected_interval=_interval,
        statevariables={"annotation_overlay": overlay},
        annotations=annotations,
        update=lambda: None,
        updates=[],
        PREDICT_LAYER_NAME=DUSTrack.PREDICT_LAYER_NAME,
        _added_layers=added,
    )
    fake._get_range_predictor = lambda: predictor
    fake._ensure_predict_layer = (
        lambda: DUSTrack._ensure_predict_layer(fake)  # exercise the real one
    )

    def _add_annotation_layers(spec, n_labels=1):
        added.append(spec)
        name = list(spec)[0]
        ann = dustrack.VideoAnnotation(n_labels=1)
        ann.name = name
        ann.fname = f"video_annotations_{name}.json"
        ann.fstem = f"video_annotations_{name}"
        ann.set_plot_type = lambda *a, **k: None
        annotations[name] = ann

    fake.add_annotation_layers = _add_annotation_layers
    fake._overlay = overlay
    return fake


# --------------------------------------------------------------------- #
# Destination: overlay, not primary                                     #
# --------------------------------------------------------------------- #
def test_predictions_go_to_the_predict_layer():
    fake = _make_fake(interval=(10, 12))
    DUSTrack.predict_range_with_dlc(fake)
    layer = fake.annotations[DUSTrack.PREDICT_LAYER_NAME]
    assert set(layer.data["0"].keys()) == {10, 11, 12}
    assert layer.data["0"][10] == [10.0, 110.0]


def test_predict_layer_is_set_as_overlay():
    """The comparison is the feature -- it has to render alongside."""
    fake = _make_fake()
    DUSTrack.predict_range_with_dlc(fake)
    assert fake._overlay.state == DUSTrack.PREDICT_LAYER_NAME


def test_primary_layer_is_never_written():
    """`ann` is untouched; nothing in the action may reference it."""
    fake = _make_fake()
    sentinel = object()
    fake.ann = sentinel  # any mutation would need attribute access
    DUSTrack.predict_range_with_dlc(fake)
    assert fake.ann is sentinel


def test_layer_name_is_dlc_prefixed():
    """The ``dlc`` prefix is what keeps this layer out of the manual
    predicates -- close guard, Train pre-flight, empty-label rewrite."""
    from dustrack._layer_names import is_manual_layer_name

    assert DUSTrack.PREDICT_LAYER_NAME.startswith("dlc")
    assert not is_manual_layer_name(DUSTrack.PREDICT_LAYER_NAME)


# --------------------------------------------------------------------- #
# The scratch layer stays in memory                                     #
# --------------------------------------------------------------------- #
def test_scratch_layer_has_no_backing_file():
    """`fname` cleared so an accidental save raises rather than dropping
    a prediction JSON where extract_frames would find it."""
    fake = _make_fake()
    DUSTrack.predict_range_with_dlc(fake)
    layer = fake.annotations[DUSTrack.PREDICT_LAYER_NAME]
    assert layer.fname is None
    assert layer.fstem is None


def test_layer_created_once_and_reused():
    fake = _make_fake(interval=(10, 12))
    DUSTrack.predict_range_with_dlc(fake)
    DUSTrack.predict_range_with_dlc(fake)
    assert len(fake._added_layers) == 1


def test_second_prediction_accumulates():
    """Predicting a new range keeps the earlier one -- several probes
    around a trial build up rather than replacing each other."""
    pred = FakePredictor()
    fake = _make_fake(interval=(10, 12), predictor=pred)
    DUSTrack.predict_range_with_dlc(fake)
    fake.get_selected_interval = lambda: (50, 52)
    DUSTrack.predict_range_with_dlc(fake)
    layer = fake.annotations[DUSTrack.PREDICT_LAYER_NAME]
    assert set(layer.data["0"].keys()) == {10, 11, 12, 50, 51, 52}


# --------------------------------------------------------------------- #
# Guards                                                                #
# --------------------------------------------------------------------- #
def test_no_project_is_a_message_not_a_crash(capsys):
    fake = _make_fake(dlcproject=None)
    DUSTrack.predict_range_with_dlc(fake)
    assert "No DLC project" in capsys.readouterr().out


def test_no_interval_selected_is_a_message_not_a_traceback(capsys):
    """get_selected_interval raises when nothing is marked; the action
    must translate that into an instruction."""
    fake = _make_fake(interval=None)
    DUSTrack.predict_range_with_dlc(fake)
    assert "press z" in capsys.readouterr().out.lower()


def test_empty_prediction_is_reported(capsys):
    fake = _make_fake(predictor=FakePredictor(frames=[]))
    fake._get_range_predictor().frames = []

    class EmptyPredictor(FakePredictor):
        def predict_range(self, video_path, start, end, annotation=True, **kw):
            return _prediction_df([])

    fake._get_range_predictor = lambda: EmptyPredictor()
    DUSTrack.predict_range_with_dlc(fake)
    assert "No predictions" in capsys.readouterr().out


def test_reports_snapshot_so_a_stale_model_is_visible(capsys):
    """The printed snapshot is the cheap staleness guard -- after a
    retrain this line changes."""
    fake = _make_fake()
    DUSTrack.predict_range_with_dlc(fake)
    out = capsys.readouterr().out
    assert "snapshot-best-270.pt" in out


def test_passes_the_video_path_to_the_predictor():
    pred = FakePredictor()
    fake = _make_fake(interval=(7, 9), predictor=pred)
    DUSTrack.predict_range_with_dlc(fake)
    video_path, start, end, annotation = pred.calls[0]
    assert video_path == "video.mp4"
    assert (start, end) == (7, 9)
    assert annotation is False  # the action converts, so it wants the frame


# --------------------------------------------------------------------- #
# Stale-model invalidation                                              #
# --------------------------------------------------------------------- #
def test_invalidate_closes_and_clears():
    pred = FakePredictor()
    fake = SimpleNamespace(_range_predictor=pred)
    DUSTrack._invalidate_range_predictor(fake)
    assert pred.closed
    assert fake._range_predictor is None


def test_invalidate_is_safe_when_nothing_cached():
    fake = SimpleNamespace(_range_predictor=None)
    DUSTrack._invalidate_range_predictor(fake)
    assert fake._range_predictor is None


def test_refresh_dlc_layers_invalidates_the_predictor():
    """Training is what makes the cached model wrong, and
    _refresh_dlc_layers is the post-training hook."""
    import inspect

    src = inspect.getsource(DUSTrack._refresh_dlc_layers)
    assert "_invalidate_range_predictor" in src


# --------------------------------------------------------------------- #
# Keybinding                                                            #
# --------------------------------------------------------------------- #
def test_e_is_bound_to_the_action():
    """`e` is left-hand: the refine loop keeps the right hand on the
    mouse, and z / a / x in this group are left-hand too."""
    import re
    import pathlib

    src = pathlib.Path(dustrack.gui.__file__).read_text(encoding="utf-8")
    m = re.search(
        r'add_key_binding\(\s*"e",\s*self\.(\w+)',
        src,
    )
    assert m is not None, "no binding for 'e'"
    assert m.group(1) == "predict_range_with_dlc"


def test_e_does_not_collide():
    """Nothing else may claim 'e'."""
    import re
    import pathlib

    src = pathlib.Path(dustrack.gui.__file__).read_text(encoding="utf-8")
    keys = re.findall(r'add_key_binding\(\s*[\'"]([^\'"]+)[\'"]', src)
    assert keys.count("e") == 1
