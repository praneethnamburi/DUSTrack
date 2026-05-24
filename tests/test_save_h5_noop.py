"""Tests for the ``s``-key save behaviour on ``.h5`` annotation layers.

Previously, pressing ``s`` on a DLC trace / labeled_data ``.h5`` layer
silently no-op'd: :meth:`VideoAnnotation.save` raises
``ValueError("Supply a json file name.")`` for non-JSON suffixes, and
the exception was swallowed by matplotlib's key-event dispatcher --
giving the user "the key did nothing" UX.

The fix in :meth:`DUSTrack.save` detects the ``.h5`` case up front and
prints a clear, actionable message instead.

Surfaces from a pia02 mid-session UX gap (2026-05-23). See roadmap
item *Next (`1.3.0` polish -- save on `.h5` layers)* in
``pn-portfolio/specs/dustrack.md``.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from dustrack.gui import DUSTrack


# ---------------------------------------------------------------------
# DUSTrack.save is a method on the heavy GUI class; stub-self via
# SimpleNamespace per the test_save_on_close.py precedent so we don't
# spin up a real Qt session for a 3-line logic test.
# ---------------------------------------------------------------------


def _stub_ann(fname: str, name: str = "stub_layer"):
    """Minimal ``VideoAnnotation``-shaped stub exposing only the fields
    DUSTrack.save reads (``fname``, ``name``, ``save``)."""
    save_calls: list = []

    def save_fn():
        save_calls.append(True)

    return SimpleNamespace(fname=fname, name=name, save=save_fn), save_calls


def test_save_on_h5_layer_short_circuits_and_prints(capsys):
    """``.h5`` layer: ann.save() is NOT called; an actionable message prints."""
    ann, save_calls = _stub_ann(
        fname="C:/proj/videos/foo_DLC_Resnet50_pn24Oct24shuffle1_snapshot_300.h5",
        name="dlc_snapshot_300",
    )
    stub = SimpleNamespace(ann=ann)

    DUSTrack.save(stub)

    assert save_calls == [], "VideoAnnotation.save should not be invoked for .h5"
    captured = capsys.readouterr()
    # Message must mention the layer name and point at the fix.
    assert ".h5" in captured.out
    assert "dlc_snapshot_300" in captured.out
    assert (
        "manual layer" in captured.out.lower()
        or "save annotation as" in captured.out.lower()
    )


def test_save_on_json_layer_calls_through_to_ann_save(capsys):
    """``.json`` layer: legacy behaviour preserved -- ``ann.save()`` runs."""
    ann, save_calls = _stub_ann(
        fname="C:/proj/videos/foo_annotations_iteration-4.json",
        name="iteration-4",
    )
    stub = SimpleNamespace(ann=ann)

    DUSTrack.save(stub)

    assert save_calls == [True], "VideoAnnotation.save must be called for .json"
    captured = capsys.readouterr()
    # No short-circuit message; the real save handler prints its own
    # success line, but that path is mocked here so capture is empty.
    assert ".h5" not in captured.out


def test_save_on_h5_layer_case_insensitive(capsys):
    """Suffix check is case-insensitive (``.H5`` ~ ``.h5``)."""
    ann, save_calls = _stub_ann(
        fname="C:/proj/videos/foo_DLC.H5",
        name="dlc_caps",
    )
    stub = SimpleNamespace(ann=ann)

    DUSTrack.save(stub)

    assert save_calls == []
    captured = capsys.readouterr()
    assert ".h5" in captured.out


def test_save_on_none_fname_falls_through_to_ann_save():
    """``ann.fname is None``: the guard does not engage; legacy save() runs.

    Empty / freshly-created annotations have ``fname = None``;
    :meth:`VideoAnnotation.save` raises its own AssertionError in that
    case (covered upstream in test_pointtracking.py). Here we just verify
    the .h5 guard does not pre-empt that path.
    """
    save_calls: list = []

    def save_fn():
        save_calls.append(True)

    ann = SimpleNamespace(fname=None, name="noname", save=save_fn)
    stub = SimpleNamespace(ann=ann)

    DUSTrack.save(stub)

    assert save_calls == [True]
