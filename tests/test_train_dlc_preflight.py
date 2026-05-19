"""Tests for the "Train DLC model" pre-flight check.

The four ``DUSTrack._*_incomplete_*`` staticmethods are pure
data-in / data-out helpers: they decide which frames are missing
bodyparts in the active annotation layer, build the recovery
sidecar payload, and format the user-facing report. The instance-
method wiring (``_save_dropped_incomplete_sidecar``,
``_prompt_drop_or_cancel``, the new branch in
``process_dlc_project``) touches the GUI and a live DLC project, so
only the pure helpers are unit-tested here -- the happy-path
wiring requires a live session.
"""
from pathlib import Path

import pytest

from dustrack.dlcinterface import DUSTrack


class TestScanIncompleteFrames:
    def test_all_frames_complete_returns_empty(self):
        data = {
            "0": {0: [1.0, 1.0], 1: [2.0, 2.0]},
            "1": {0: [3.0, 3.0], 1: [4.0, 4.0]},
        }
        assert DUSTrack._scan_incomplete_frames(data) == {}

    def test_empty_data_returns_empty(self):
        assert DUSTrack._scan_incomplete_frames({}) == {}

    def test_all_labels_empty_returns_empty(self):
        # An annotation with placeholder labels but no points yet --
        # treat as complete, not "every frame missing every label".
        data = {"0": {}, "1": {}, "2": {}}
        assert DUSTrack._scan_incomplete_frames(data) == {}

    def test_single_incomplete_frame_lists_missing_label(self):
        data = {
            "0": {5: [1.0, 1.0], 6: [1.0, 1.0]},
            "1": {5: [2.0, 2.0]},  # frame 6 missing label "1"
        }
        assert DUSTrack._scan_incomplete_frames(data) == {6: ["1"]}

    def test_multiple_missing_labels_listed(self):
        data = {
            "0": {5: [1.0, 1.0], 7: [1.0, 1.0]},  # frame 6 missing "0"
            "1": {5: [2.0, 2.0], 6: [2.0, 2.0]},  # frame 7 missing "1"
            "2": {5: [3.0, 3.0]},                 # frames 6, 7 missing "2"
        }
        result = DUSTrack._scan_incomplete_frames(data)
        # Frame 5 is complete; frames 6 and 7 are incomplete.
        assert set(result.keys()) == {6, 7}
        assert set(result[6]) == {"0", "2"}
        assert set(result[7]) == {"1", "2"}

    def test_frames_sorted(self):
        data = {
            "0": {10: [1.0, 1.0], 2: [1.0, 1.0], 5: [1.0, 1.0]},
            "1": {2: [2.0, 2.0]},  # 10 and 5 missing "1"
        }
        result = DUSTrack._scan_incomplete_frames(data)
        assert list(result.keys()) == [5, 10]

    def test_empty_label_does_not_fail_every_frame(self):
        # Label "9" is an empty placeholder. Frames 0,1 are complete
        # across the two active labels ("0", "1"); nothing incomplete.
        data = {
            "0": {0: [1.0, 1.0], 1: [1.0, 1.0]},
            "1": {0: [2.0, 2.0], 1: [2.0, 2.0]},
            "9": {},
        }
        assert DUSTrack._scan_incomplete_frames(data) == {}


class TestBuildDroppedIncompletePayload:
    def test_payload_contains_present_labels_only(self):
        data = {
            "0": {5: [1.0, 1.0]},
            "1": {5: [2.0, 2.0], 6: [2.0, 2.0]},
        }
        incomplete = {6: ["0"]}  # frame 6 is missing label "0"
        payload = DUSTrack._build_dropped_incomplete_payload(data, incomplete)
        # Only one entry, keyed by stringified frame
        assert list(payload.keys()) == ["6"]
        # Frame 6 had label "1" only -- the dropped sidecar preserves
        # that. Label "0" is absent (it was the missing one).
        assert payload["6"] == {"1": [2.0, 2.0]}

    def test_payload_floats_not_ints(self):
        data = {"0": {5: [1, 2]}}
        incomplete = {5: ["1"]}  # arbitrary
        payload = DUSTrack._build_dropped_incomplete_payload(data, incomplete)
        # Values must be floats so the JSON dump is dnav-compatible.
        assert payload["5"]["0"] == [1.0, 2.0]
        assert all(isinstance(x, float) for x in payload["5"]["0"])

    def test_empty_incomplete_returns_empty_payload(self):
        data = {"0": {5: [1.0, 1.0]}}
        assert DUSTrack._build_dropped_incomplete_payload(data, {}) == {}


class TestBuildDroppedIncompleteSidecarName:
    def test_composite_suffix_no_json_extension(self):
        # The discovery glob is {video_stem}*_annotations*.json --
        # the composite suffix must not end in .json.
        name = DUSTrack._build_dropped_incomplete_sidecar_name(
            "/data/v1/myvideo_annotations_pn.json",
            "20260518T143000",
        )
        assert name.endswith(".dustrack-dropped-incomplete-20260518T143000")
        assert not name.endswith(".json")

    def test_sidecar_in_same_directory_as_annotation(self):
        ann_fname = str(Path("/data/v1/myvideo_annotations_pn.json"))
        name = DUSTrack._build_dropped_incomplete_sidecar_name(
            ann_fname, "20260518T143000"
        )
        assert Path(name).parent == Path(ann_fname).parent

    def test_sidecar_stem_derived_from_annotation_stem(self):
        name = DUSTrack._build_dropped_incomplete_sidecar_name(
            "/data/v1/myvideo_annotations_pn.json",
            "20260518T143000",
        )
        assert Path(name).name.startswith("myvideo_annotations_pn.")


class TestFormatIncompleteBreakdown:
    def test_single_frame_breakdown(self):
        out = DUSTrack._format_incomplete_breakdown({5: ["0", "2"]})
        assert out == "Frame 5: missing 0, 2"

    def test_multi_frame_sorted(self):
        out = DUSTrack._format_incomplete_breakdown(
            {10: ["1"], 2: ["0"], 5: ["2"]}
        )
        # Frame-sorted regardless of dict insertion order.
        assert out == "Frame 2: missing 0\nFrame 5: missing 2\nFrame 10: missing 1"

    def test_truncation_tail(self):
        # 500 incomplete frames, max_rows=3 -> 3 rows + truncation note.
        incomplete = {i: ["0"] for i in range(500)}
        out = DUSTrack._format_incomplete_breakdown(incomplete, max_rows=3)
        lines = out.splitlines()
        assert len(lines) == 4  # 3 rows + tail
        assert lines[0] == "Frame 0: missing 0"
        assert lines[-1] == "... (497 more frames)"
