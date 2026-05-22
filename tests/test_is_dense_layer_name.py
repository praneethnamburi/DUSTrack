"""Tests for :func:`_is_dense_layer_name`.

The predicate drives the "render as line, not dots" default for
DLC-pipeline tracking output across all three layer-add paths --
:meth:`DLCProject.annotate` (cold open), :meth:`DUSTrack._refresh_dlc_layers`
(post-train refresh), and :meth:`DUSTrack._adopt_layer` (in-session,
including Reduce jitter on a non-DLC source). Pins the dispatch
matrix so future smoothing recipes (or a widened prefix list) can
extend the pattern data without regressing the existing matches.
"""
from dustrack._layer_names import _is_dense_layer_name


class TestDLCInferenceNames:
    """``dlc_*`` prefix -- the original predicate, preserved."""

    def test_h5_trace_token(self):
        assert _is_dense_layer_name("dlc_iteration-0_250000")

    def test_high_iteration_index(self):
        assert _is_dense_layer_name("dlc_iteration-12_500000")


class TestLKRSTCOutputs:
    """``lkmovavg`` substring -- broadens the predicate to catch
    Reduce-jitter output regardless of source layer."""

    def test_lkmovavg_on_dlc_source(self):
        # canonical_layer_name's DLC-stem branch -- name already starts
        # with dlc_, so either rule would catch it. Pinned here so the
        # substring rule never accidentally regresses below the prefix
        # rule.
        assert _is_dense_layer_name("dlc_iteration-2_0.500")
        assert _is_dense_layer_name("dlc_iteration-2_lkmovavg_0.500")

    def test_lkmovavg_on_dlccorr_source(self):
        # The motivating gap: Reduce-jitter on the dlccorr layer lands
        # at <video>_annotations_dlccorr_lkmovavg_<w>.json, which
        # canonical_layer_name's _annotations branch names
        # "dlccorr_lkmovavg_<w>". Pre-fix this skipped line-plot
        # treatment because the name doesn't start with dlc_.
        assert _is_dense_layer_name("dlccorr_lkmovavg_0.500")

    def test_lkmovavg_on_manual_iteration_source(self):
        # Reduce-jitter on a manual iteration-N layer:
        # <video>_annotations_iteration-2_lkmovavg_<w>.json ->
        # "iteration-2_lkmovavg_<w>".
        assert _is_dense_layer_name("iteration-2_lkmovavg_0.500")


class TestDLCCorrectionsLayer:
    """The ``dlccorr`` manual-corrections splice inherits per-frame
    coverage from the overlay DLC trace (manual edits replace a
    handful of frames; everything else is the overlay's data), so it
    renders as line like the trace it was spliced from."""

    def test_dlccorr_bare(self):
        assert _is_dense_layer_name("dlccorr")

    def test_dlccorr_lkmovavg_matches_via_both_rules(self):
        # Both the dlccorr prefix and the lkmovavg substring fire on
        # this name -- pinned so neither rule can silently regress
        # without the other catching it.
        assert _is_dense_layer_name("dlccorr_lkmovavg_0.500")


class TestSparseLayerNames:
    """Manual and placeholder layers must stay on the dnav default
    ("dot") -- forcing them to line would draw confusing connecting
    segments across frames with no data."""

    def test_manual_layer(self):
        assert not _is_dense_layer_name("manual")

    def test_named_manual_layer(self):
        assert not _is_dense_layer_name("brachialis_praneeth")

    def test_iteration_layer(self):
        assert not _is_dense_layer_name("iteration-2")

    def test_buffer_layer(self):
        assert not _is_dense_layer_name("buffer")

    def test_empty_name(self):
        assert not _is_dense_layer_name("")
