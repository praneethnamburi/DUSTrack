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


class TestIsManualAnnotationLayer:
    VIDEO = "/data/v1/myvideo.mp4"

    def _check(self, ann_fname, ann_name):
        return DUSTrack._is_manual_annotation_layer(self.VIDEO, ann_fname, ann_name)

    def test_typical_iteration_layer_is_manual(self):
        assert self._check("/data/v1/myvideo_annotations_iteration-2.json", "iteration-2")

    def test_unsuffixed_annotations_file_is_manual(self):
        # The default <video>_annotations.json (no suffix; layer name is "noname"
        # in dnav but DUSTrack canonicalises it).
        assert self._check("/data/v1/myvideo_annotations.json", "")

    def test_renamed_layer_still_manual(self):
        # User closed UI and renamed iteration-1 to iter1 on disk.
        # File pattern still matches; layer name is irrelevant.
        assert self._check("/data/v1/myvideo_annotations_iter1.json", "iter1")

    def test_experimenter_initials_layer_is_manual(self):
        # Initial layer seeded with experimenter initials per DUSTrack convention.
        assert self._check("/data/v1/myvideo_annotations_pn.json", "pn")

    def test_dlccorr_excluded(self):
        # File matches the pattern but the layer name is the terminal
        # apply_manual_corrections output -- not a training source.
        assert not self._check(
            "/data/v1/myvideo_annotations_dlccorr.json", "dlccorr"
        )

    def test_buffer_excluded(self):
        # Workspace scratch layer.
        assert not self._check(
            "/data/v1/myvideo_annotations_buffer.json", "buffer"
        )

    def test_dlc_trace_layer_excluded_by_name(self):
        # LK output (process_with_lk) lands alongside the video as a
        # .json matching the pattern, but its layer name starts with
        # "dlc" so we exclude.
        assert not self._check(
            "/data/v1/myvideo_annotations_dlc_iteration-2_0.500.json",
            "dlc_iteration-2_0.500",
        )

    def test_h5_file_excluded(self):
        # DLC traces from h5 don't match the .json suffix.
        assert not self._check(
            "/data/v1/myvideo_iteration-2.h5", "dlc_iteration-2"
        )

    def test_different_directory_excluded(self):
        # .json with the right name but in a different folder
        # (e.g. a postprocess subdir or unrelated working dir).
        assert not self._check(
            "/data/v1/postprocess/myvideo_annotations_iteration-2.json",
            "iteration-2",
        )

    def test_wrong_stem_pattern_excluded(self):
        # Filename doesn't match <video_stem>_annotations*.
        assert not self._check("/data/v1/myvideo_notes_iteration-2.json", "iteration-2")

    def test_none_fname_excluded(self):
        # The file-aware predicate still excludes None fname -- callers
        # that need the disk file (save-on-close diff) require it. The
        # incomplete-frame scan uses _is_manual_layer_name below, which
        # does accept None fname (the first-time-training case).
        assert not self._check(None, "iteration-2")


class TestIsManualLayerName:
    """Name-only predicate: used by the Train pre-flight to decide
    whether to scan an in-session layer for incomplete frames, before
    the layer has ever been saved to disk (``ann.fname is None``)."""

    def test_typical_iteration_layer(self):
        assert DUSTrack._is_manual_layer_name("iteration-0")
        assert DUSTrack._is_manual_layer_name("iteration-7")

    def test_renamed_user_layer(self):
        assert DUSTrack._is_manual_layer_name("iter1")
        assert DUSTrack._is_manual_layer_name("pn")
        assert DUSTrack._is_manual_layer_name("manual")

    def test_dlccorr_excluded(self):
        assert not DUSTrack._is_manual_layer_name("dlccorr")

    def test_buffer_excluded(self):
        assert not DUSTrack._is_manual_layer_name("buffer")

    def test_dlc_prefix_excluded(self):
        assert not DUSTrack._is_manual_layer_name("dlc_iteration-2")
        assert not DUSTrack._is_manual_layer_name("dlc_iteration-2_0.500")


# ---------- _scan_unsaved_and_incomplete ----------
# Regression coverage for the first-time-training bug: a freshly-
# annotated layer with ``ann.fname is None`` (never saved yet) must
# still be flagged when it has incomplete frames. Pre-2026-05-21 the
# scan filtered such layers out via the file-aware predicate, so the
# user clicked Train and got no warning despite the missing bodypart.

from types import SimpleNamespace


def _make_ann_stub(name, fname, data):
    return SimpleNamespace(name=name, fname=(None if fname is None else str(fname)), data=data)


def _scan_with_stub_self(video_fname, ann_stubs):
    stub = SimpleNamespace(
        fname=str(video_fname),
        annotations=ann_stubs,
        _is_manual_layer_name=DUSTrack._is_manual_layer_name,
        _is_manual_annotation_layer=DUSTrack._is_manual_annotation_layer,
        _normalize_layer_data=DUSTrack._normalize_layer_data,
        _load_layer_disk_data=DUSTrack._load_layer_disk_data,
        _diff_ann_vs_disk=DUSTrack._diff_ann_vs_disk,
        _scan_incomplete_frames=DUSTrack._scan_incomplete_frames,
    )
    return DUSTrack._scan_unsaved_and_incomplete(stub)


class TestScanUnsavedAndIncomplete:
    def test_unsaved_layer_with_incomplete_frame_is_flagged(self, tmp_path):
        """The bug: first-time training, layer never saved, has an
        incomplete frame. Pre-2026-05-21 was silently skipped."""
        video = tmp_path / "v.mp4"
        data = {
            "0": {5: [1.0, 1.0], 6: [1.0, 1.0]},
            "1": {5: [2.0, 2.0]},  # frame 6 missing label "1"
        }
        ann = _make_ann_stub("iteration-0", fname=None, data=data)
        result = _scan_with_stub_self(video, [ann])
        assert "iteration-0" in result
        # Disk-diff is skipped on None fname; incomplete is populated.
        assert result["iteration-0"]["diff"] == {"added": [], "removed": [], "modified": []}
        assert result["iteration-0"]["incomplete"] == {6: ["1"]}

    def test_unsaved_layer_with_no_incomplete_omitted(self, tmp_path):
        """Don't false-positive on a clean unsaved layer."""
        video = tmp_path / "v.mp4"
        data = {"0": {5: [1.0, 1.0]}, "1": {5: [2.0, 2.0]}}
        ann = _make_ann_stub("iteration-0", fname=None, data=data)
        result = _scan_with_stub_self(video, [ann])
        assert result == {}

    def test_unsaved_dlccorr_layer_skipped(self, tmp_path):
        """The dlccorr layer name is excluded even when in-session
        unsaved -- it's not a manual training source."""
        video = tmp_path / "v.mp4"
        data = {
            "0": {5: [1.0, 1.0], 6: [1.0, 1.0]},
            "1": {5: [2.0, 2.0]},
        }
        ann = _make_ann_stub("dlccorr", fname=None, data=data)
        result = _scan_with_stub_self(video, [ann])
        assert result == {}

    def test_saved_layer_with_incomplete_still_flagged(self, tmp_path):
        """Regression guard: the existing (saved-layer) flow must keep
        working after the inclusion check was loosened."""
        import json
        video = tmp_path / "v.mp4"
        ann_path = tmp_path / "v_annotations_iteration-0.json"
        data = {
            "0": {5: [1.0, 1.0], 6: [1.0, 1.0]},
            "1": {5: [2.0, 2.0]},  # frame 6 missing "1"
        }
        with open(ann_path, "w") as f:
            json.dump({"0": {"5": [1.0, 1.0], "6": [1.0, 1.0]},
                       "1": {"5": [2.0, 2.0]}}, f)
        ann = _make_ann_stub("iteration-0", ann_path, data)
        result = _scan_with_stub_self(video, [ann])
        assert "iteration-0" in result
        assert result["iteration-0"]["incomplete"] == {6: ["1"]}

    def test_saved_layer_with_disk_diff_only(self, tmp_path):
        """Layer is complete in memory but disk differs: diff is
        populated, incomplete is empty."""
        import json
        video = tmp_path / "v.mp4"
        ann_path = tmp_path / "v_annotations_iteration-0.json"
        data = {"0": {5: [1.0, 1.0], 6: [2.0, 2.0]}}
        with open(ann_path, "w") as f:
            json.dump({"0": {"5": [1.0, 1.0]}}, f)
        ann = _make_ann_stub("iteration-0", ann_path, data)
        result = _scan_with_stub_self(video, [ann])
        assert "iteration-0" in result
        assert result["iteration-0"]["incomplete"] == {}
        assert ("0", 6) in result["iteration-0"]["diff"]["added"]


class TestNormalizeLayerData:
    def test_int_keys_float_values(self):
        # JSON loads frame keys as strings and xy as int/float mixed.
        raw = {"0": {"5": [1, 2], "6": [3.5, 4]}, "1": {"5": [10, 20]}}
        out = DUSTrack._normalize_layer_data(raw)
        assert out == {
            "0": {5: [1.0, 2.0], 6: [3.5, 4.0]},
            "1": {5: [10.0, 20.0]},
        }
        # Verify the types explicitly.
        for label, frames in out.items():
            for frame, xy in frames.items():
                assert isinstance(frame, int)
                assert all(isinstance(x, float) for x in xy)

    def test_drops_empty_labels(self):
        raw = {"0": {5: [1.0, 1.0]}, "1": {}, "2": {3: [2.0, 2.0]}}
        out = DUSTrack._normalize_layer_data(raw)
        assert set(out.keys()) == {"0", "2"}


class TestDiffAnnVsDisk:
    def test_identical_returns_no_diff(self):
        a = {"0": {5: [1.0, 1.0]}}
        b = {"0": {5: [1.0, 1.0]}}
        result = DUSTrack._diff_ann_vs_disk(a, b)
        assert result == {"added": [], "removed": [], "modified": []}

    def test_added_frames_detected(self):
        mem = {"0": {5: [1.0, 1.0], 6: [2.0, 2.0]}}
        disk = {"0": {5: [1.0, 1.0]}}
        result = DUSTrack._diff_ann_vs_disk(mem, disk)
        assert result["added"] == [("0", 6)]
        assert result["removed"] == []
        assert result["modified"] == []

    def test_removed_frames_detected(self):
        mem = {"0": {5: [1.0, 1.0]}}
        disk = {"0": {5: [1.0, 1.0]}, "1": {3: [9.0, 9.0]}}
        result = DUSTrack._diff_ann_vs_disk(mem, disk)
        assert result["added"] == []
        assert result["removed"] == [("1", 3)]
        assert result["modified"] == []

    def test_modified_xy_detected(self):
        mem = {"0": {5: [1.5, 1.0]}}
        disk = {"0": {5: [1.0, 1.0]}}
        result = DUSTrack._diff_ann_vs_disk(mem, disk)
        assert result["added"] == []
        assert result["removed"] == []
        assert result["modified"] == [("0", 5)]

    def test_disk_empty_treats_all_mem_as_added(self):
        # Fully unsaved layer: disk has no file yet, mem has data.
        mem = {"0": {5: [1.0, 1.0], 6: [2.0, 2.0]}}
        result = DUSTrack._diff_ann_vs_disk(mem, {})
        assert result["added"] == [("0", 5), ("0", 6)]
        assert result["removed"] == []
        assert result["modified"] == []

    def test_results_label_then_frame_sorted(self):
        mem = {
            "1": {5: [1.0, 1.0], 2: [1.0, 1.0]},
            "0": {10: [1.0, 1.0]},
        }
        result = DUSTrack._diff_ann_vs_disk(mem, {})
        # Labels sorted: "0" then "1"; within each label, frames sorted.
        assert result["added"] == [("0", 10), ("1", 2), ("1", 5)]


class TestFormatPreFlightSummary:
    def test_diffs_only_layer(self):
        issues = {
            "iteration-1": {
                "diff": {"added": [("0", 5)], "removed": [], "modified": []},
                "incomplete": {},
            }
        }
        out = DUSTrack._format_pre_flight_summary(issues)
        assert "Layer 'iteration-1'" in out
        assert "+1 added" in out
        assert "(no incomplete frames)" in out

    def test_incomplete_only_layer(self):
        issues = {
            "iteration-2": {
                "diff": {"added": [], "removed": [], "modified": []},
                "incomplete": {5: ["1"], 10: ["0", "2"]},
            }
        }
        out = DUSTrack._format_pre_flight_summary(issues)
        assert "(no unsaved changes)" in out
        assert "Incomplete frames: 2" in out
        assert "frame 5: missing 1" in out
        assert "frame 10: missing 0, 2" in out

    def test_both_kinds_of_issues(self):
        issues = {
            "pn": {
                "diff": {
                    "added": [("0", 5)],
                    "removed": [("1", 3)],
                    "modified": [("0", 7)],
                },
                "incomplete": {2: ["1"]},
            }
        }
        out = DUSTrack._format_pre_flight_summary(issues)
        assert "+1 added" in out
        assert "-1 removed" in out
        assert "~1 modified" in out
        assert "Incomplete frames: 1" in out

    def test_multiple_layers_separated_by_blank_line(self):
        issues = {
            "iteration-1": {
                "diff": {"added": [("0", 5)], "removed": [], "modified": []},
                "incomplete": {},
            },
            "iteration-2": {
                "diff": {"added": [], "removed": [], "modified": []},
                "incomplete": {7: ["0"]},
            },
        }
        out = DUSTrack._format_pre_flight_summary(issues)
        assert "\n\n" in out  # blank line between layer blocks
        assert "Layer 'iteration-1'" in out
        assert "Layer 'iteration-2'" in out

    def test_incomplete_truncation(self):
        issues = {
            "iteration-1": {
                "diff": {"added": [], "removed": [], "modified": []},
                "incomplete": {i: ["0"] for i in range(10)},
            }
        }
        out = DUSTrack._format_pre_flight_summary(
            issues, max_incomplete_examples=2
        )
        assert "Incomplete frames: 10" in out
        assert "... (8 more)" in out
