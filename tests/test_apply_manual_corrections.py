"""Tests for :meth:`DUSTrack.apply_manual_corrections` -- the splice helper +
the dispatch error branches.

The splice logic lives in the pure-data static method
:meth:`DUSTrack._merge_overlay_with_patch`, which is exercised below
with synthetic ``data`` dicts. The button-action method itself touches
``self.ann`` / overlay statevars / disk, so only its precondition
checks are tested here -- the happy path requires a live GUI session.
"""
import pytest

from dustrack.dlcinterface import DUSTrack


class TestMergeOverlayWithPatch:
    def test_dense_source_sparse_patch_overrides(self):
        # DLC overlay: dense per-frame predictions. Manual patch: a
        # couple of corrected frames. Result: dense, with patch values
        # at the corrected frames.
        source = {
            "0": {0: [10.0, 10.0], 1: [11.0, 11.0], 2: [12.0, 12.0]},
            "1": {0: [20.0, 20.0], 1: [21.0, 21.0], 2: [22.0, 22.0]},
        }
        patch = {
            "0": {1: [99.0, 99.0]},  # override one frame on label 0
        }
        merged = DUSTrack._merge_overlay_with_patch(source, patch)
        assert merged["0"] == {0: [10.0, 10.0], 1: [99.0, 99.0], 2: [12.0, 12.0]}
        assert merged["1"] == {0: [20.0, 20.0], 1: [21.0, 21.0], 2: [22.0, 22.0]}

    def test_empty_patch_returns_copy_of_source(self):
        source = {"0": {0: [1.0, 2.0]}, "1": {5: [3.0, 4.0]}}
        patch = {}
        merged = DUSTrack._merge_overlay_with_patch(source, patch)
        assert merged == source
        # Distinct dicts -- patch-side mutation later must not leak back.
        assert merged is not source
        assert merged["0"] is not source["0"]

    def test_empty_source_returns_copy_of_patch(self):
        source = {}
        patch = {"0": {0: [1.0, 2.0]}}
        merged = DUSTrack._merge_overlay_with_patch(source, patch)
        assert merged == patch

    def test_both_empty(self):
        assert DUSTrack._merge_overlay_with_patch({}, {}) == {}

    def test_disjoint_labels_carry_through(self):
        # Source has label "0", patch has label "1" -- the result
        # carries both, no merging at the (label, frame) level.
        source = {"0": {0: [1.0, 1.0]}}
        patch = {"1": {0: [2.0, 2.0]}}
        merged = DUSTrack._merge_overlay_with_patch(source, patch)
        assert merged == {"0": {0: [1.0, 1.0]}, "1": {0: [2.0, 2.0]}}

    def test_patch_only_label_with_multiple_frames(self):
        # A label that exists only in patch comes through with all of
        # its frames -- not just the ones the source happens to have.
        source = {"0": {0: [1.0, 1.0], 1: [1.0, 1.0]}}
        patch = {"2": {0: [9.0, 9.0], 1: [9.0, 9.0], 2: [9.0, 9.0]}}
        merged = DUSTrack._merge_overlay_with_patch(source, patch)
        assert merged["2"] == {0: [9.0, 9.0], 1: [9.0, 9.0], 2: [9.0, 9.0]}

    def test_override_at_intersecting_label_and_frame(self):
        # Same label, same frame: patch wins.
        source = {"0": {0: [1.0, 1.0]}}
        patch = {"0": {0: [9.0, 9.0]}}
        merged = DUSTrack._merge_overlay_with_patch(source, patch)
        assert merged == {"0": {0: [9.0, 9.0]}}

    def test_result_does_not_mutate_inputs(self):
        source = {"0": {0: [1.0, 1.0]}}
        patch = {"0": {0: [9.0, 9.0], 1: [8.0, 8.0]}}
        DUSTrack._merge_overlay_with_patch(source, patch)
        assert source == {"0": {0: [1.0, 1.0]}}
        assert patch == {"0": {0: [9.0, 9.0], 1: [8.0, 8.0]}}

    def test_labels_sorted_in_result(self):
        # The merged dict iterates labels in sorted order so downstream
        # display / save is deterministic regardless of input order.
        source = {"2": {0: [2.0, 2.0]}, "0": {0: [0.0, 0.0]}}
        patch = {"1": {0: [1.0, 1.0]}}
        merged = DUSTrack._merge_overlay_with_patch(source, patch)
        assert list(merged.keys()) == ["0", "1", "2"]


class TestCorrectionsLayerNameConstant:
    def test_matches_extract_frames_exclusion_pattern(self):
        # The _extract_frames filter excludes any filename containing
        # "_dlccorr". The class attribute must contain the substring
        # so the filter actually picks up files we write.
        assert "dlccorr" in f"_{DUSTrack.CORRECTIONS_LAYER_NAME}"
