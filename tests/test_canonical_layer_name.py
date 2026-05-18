"""Tests for :meth:`VideoFileManager.canonical_layer_name`.

This is the single source of truth for DUSTrack layer names derived
from filepaths -- exercised by the cold-open path
(:meth:`DLCProject.annotate`), the post-train refresh
(:meth:`DUSTrack._refresh_dlc_layers`), and the in-session adopt path
(:meth:`DUSTrack._adopt_layer`). The harmonization contract is that
all three paths name a given file identically; the unit tests below
pin the function's behaviour against the file-pattern matrix the
DUSTrack workflow produces.
"""
from pathlib import Path

import pytest

from dustrack.dlcinterface import HAS_DLC

if not HAS_DLC:
    pytest.skip("VideoFileManager requires deeplabcut", allow_module_level=True)

from dustrack.dlcinterface import VideoFileManager

canonical = VideoFileManager.canonical_layer_name


class TestManualAnnotationNames:
    """Files matching ``*_annotations[_<suffix>].json``: layer name is the
    suffix after ``_annotations`` (or empty string if absent)."""

    def test_simple_suffix(self):
        assert canonical("/proj/vid_annotations_brachialis.json") == "brachialis"

    def test_multi_token_suffix(self):
        assert (
            canonical("/proj/vid_annotations_brachialis_praneeth.json")
            == "brachialis_praneeth"
        )

    def test_iteration_suffix(self):
        assert canonical("/proj/vid_annotations_iteration-2.json") == "iteration-2"

    def test_buffer_suffix(self):
        assert canonical("/proj/vid_annotations_buffer.json") == "buffer"

    def test_lkmovavg_on_manual_source(self):
        # An LK pass over a manual layer produces <stem>_lkmovavg_X.json.
        # The stem still contains "_annotations", so the manual branch wins.
        assert (
            canonical("/proj/vid_annotations_iteration-2_lkmovavg_0.500.json")
            == "iteration-2_lkmovavg_0.500"
        )


class TestDLCTraceNames:
    """Files under ``videos/iteration-{N}/`` whose stem contains ``DLC``:
    layer name is ``dlc_iteration-{N}_<last underscore-token of stem>``."""

    def test_h5_trace(self):
        fname = "/proj/videos/iteration-0/vidDLC_resnet50_proj1_250000.h5"
        assert canonical(fname) == "dlc_iteration-0_250000"

    def test_json_trace(self):
        fname = "/proj/videos/iteration-1/vidDLC_resnet50_proj1_250000.json"
        assert canonical(fname) == "dlc_iteration-1_250000"

    def test_lkmovavg_on_dlc_source(self):
        # The motivating bug: pre-fix this layer was named "noname" in-session
        # but "dlc_iteration-2_0.500" on reload. The canonical name is the
        # reload-shaped one.
        fname = (
            "/proj/videos/iteration-2/vidDLC_resnet50_proj1_250000_lkmovavg_0.500.json"
        )
        assert canonical(fname) == "dlc_iteration-2_0.500"


class TestFallback:
    """Paths that match neither pattern fall back to the file stem."""

    def test_bare_file(self):
        assert canonical("/somewhere/just_a_file.json") == "just_a_file"

    def test_no_extension(self):
        assert canonical("/somewhere/no_ext") == "no_ext"


class TestPathTypeAcceptance:
    """The function takes ``str`` or ``Path`` -- ``add_annotation_layers``
    passes both shapes through it."""

    def test_accepts_pathlib_path(self):
        p = Path("/proj/vid_annotations_brachialis.json")
        assert canonical(p) == "brachialis"

    def test_accepts_string(self):
        assert canonical("/proj/vid_annotations_brachialis.json") == "brachialis"
