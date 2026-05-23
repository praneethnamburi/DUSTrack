"""Tests for stray-label detection + the opt-in "keep strays" remediation path.

"Stray label" = a label with at least one annotation that isn't a
project bodypart. Detection happens in
:func:`_preflight.scan_stray_labels`; the user decides whether to
keep strays in the saved file via a checkbox on the Train pre-flight
modal (default unchecked -> strip on save). All logic is pure
(data dicts in, data/None out); the modal is pure Qt scaffolding
on top.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dustrack import _preflight
from dustrack._preflight import (
    apply_pre_flight_remediations,
    format_pre_flight_summary,
    has_strays,
    scan_stray_labels,
)


class TestScanStrayLabels:
    def test_project_unaware_returns_empty(self):
        # target_labels=None -> no external truth to call anything
        # "stray" against. Always returns {}.
        data = {"0": {0: [1.0, 1.0]}, "1": {0: [2.0, 2.0]}, "9": {5: [9.0, 9.0]}}
        assert scan_stray_labels(data, target_labels=None) == {}

    def test_no_strays_returns_empty(self):
        # Every annotated label is in target_labels.
        data = {"0": {0: [1.0, 1.0]}, "1": {0: [2.0, 2.0]}}
        assert scan_stray_labels(data, target_labels=["0", "1"]) == {}

    def test_single_stray_returns_its_frames(self):
        data = {
            "0": {0: [1.0, 1.0], 1: [1.0, 1.0]},
            "1": {0: [2.0, 2.0], 1: [2.0, 2.0]},
            "9": {5: [9.0, 9.0], 3: [9.0, 9.0]},
        }
        result = scan_stray_labels(data, target_labels=["0", "1"])
        assert result == {"9": [3, 5]}

    def test_multiple_strays_each_with_sorted_frames(self):
        data = {
            "0": {0: [1.0, 1.0]},
            "2": {10: [2.0, 2.0], 5: [2.0, 2.0]},
            "3": {1: [3.0, 3.0]},
        }
        result = scan_stray_labels(data, target_labels=["0"])
        assert result == {"2": [5, 10], "3": [1]}

    def test_empty_stray_label_excluded(self):
        # Label exists but has no annotations -- "stray" means
        # "annotated, not in bodyparts". Empty placeholders are
        # invisible to this scan.
        data = {
            "0": {0: [1.0, 1.0]},
            "9": {},
        }
        result = scan_stray_labels(data, target_labels=["0"])
        assert result == {}

    def test_empty_target_labels_treats_everything_as_stray(self):
        # Defensive: empty bodyparts list -> every annotated label
        # is stray. Train preflight upstream guards against this
        # (empty bodyparts -> target_labels=None), but the pure
        # function should still behave consistently.
        data = {"0": {0: [1.0, 1.0]}, "9": {5: [9.0, 9.0]}}
        result = scan_stray_labels(data, target_labels=[])
        assert result == {"0": [0], "9": [5]}


class TestHasStrays:
    def test_no_issues_returns_false(self):
        assert has_strays({}) is False

    def test_issues_without_strays_returns_false(self):
        issues = {"layer_a": {"diff": {}, "incomplete": {}}}
        assert has_strays(issues) is False

    def test_issues_with_empty_strays_returns_false(self):
        # Defensive: explicit empty dict in the strays slot is still
        # "no strays".
        issues = {"layer_a": {"diff": {}, "incomplete": {}, "strays": {}}}
        assert has_strays(issues) is False

    def test_issues_with_strays_returns_true(self):
        issues = {
            "layer_a": {"diff": {}, "incomplete": {}, "strays": {"9": [5]}},
        }
        assert has_strays(issues) is True

    def test_mixed_layers_returns_true_when_any_have_strays(self):
        issues = {
            "layer_a": {"diff": {}, "incomplete": {}, "strays": {}},
            "layer_b": {"diff": {}, "incomplete": {}, "strays": {"9": [5]}},
        }
        assert has_strays(issues) is True


class TestFormatPreFlightSummaryWithStrays:
    def test_stray_block_appears_when_present(self):
        issues = {
            "manual_layer": {
                "diff": {"added": [], "removed": [], "modified": []},
                "incomplete": {},
                "strays": {"9": [5, 10, 15]},
            },
        }
        out = format_pre_flight_summary(issues)
        assert "Labels not in this DLC project" in out
        assert "label '9'" in out
        assert "3 annotations" in out

    def test_no_stray_block_when_absent(self):
        issues = {
            "manual_layer": {
                "diff": {"added": [], "removed": [], "modified": []},
                "incomplete": {},
                "strays": {},
            },
        }
        out = format_pre_flight_summary(issues)
        assert "not in this DLC project" not in out


class TestApplyRemediationsStripStrays:
    """``apply_pre_flight_remediations(strip_strays=...)`` controls
    whether annotations on non-bodypart labels survive the save."""

    def _setup(self, tmp_path):
        """Build a real VideoAnnotation on disk with:
        - labels "0" and "1" fully annotated at frames 10..12 (complete)
        - label "9" annotated at frame 11 only (stray, present at a
          complete-by-bodyparts frame)
        Returns (ann, fake_dustrack_args).
        """
        import matplotlib.pyplot as plt
        from dustrack.annotations import VideoAnnotation

        ann_path = tmp_path / "v_annotations_iteration-1.json"
        with open(ann_path, "w") as f:
            json.dump(
                {
                    "0": {"10": [1.0, 1.0], "11": [1.1, 1.1], "12": [1.2, 1.2]},
                    "1": {"10": [2.0, 2.0], "11": [2.1, 2.1], "12": [2.2, 2.2]},
                    "9": {"11": [9.0, 9.0]},
                },
                f,
            )

        fig, ax = plt.subplots()
        ann = VideoAnnotation(
            fname=str(ann_path),
            vname=None,
            name="iteration-1",
            ax_list_scatter=[ax],
            ax_list_trace_x=[ax],
            ax_list_trace_y=[ax],
        )
        # Stray label "9" wasn't declared at construction time
        # (n_labels defaults to 1); inject it directly so we can
        # exercise the stray-strip path. The data dict already has
        # the key from the JSON; just make sure ``.labels`` knows
        # about it via add_label.
        if "9" not in ann.labels:
            ann.add_label(label="9")
        # Re-load the disk values for label 9 (add_label seeded the
        # in-memory entry empty).
        ann.data["9"][11] = [9.0, 9.0]

        issues = {
            "iteration-1": {
                "diff": {"added": [], "removed": [], "modified": []},
                "incomplete": {},
                "strays": {"9": [11]},
            },
        }
        return ann, fig, issues, str(tmp_path / "v.mp4")

    def test_strip_strays_true_removes_stray_annotations(self, tmp_path):
        import matplotlib.pyplot as plt
        ann, fig, issues, fake_video = self._setup(tmp_path)
        try:
            apply_pre_flight_remediations(
                annotations={"iteration-1": ann},
                video_fname=fake_video,
                issues=issues,
                make_annotation_file_name=lambda *_a, **_k: ann.fname,
                strip_strays=True,
            )
            # Bodypart labels untouched.
            assert ann.get_frames("0") == [10, 11, 12]
            assert ann.get_frames("1") == [10, 11, 12]
            # Stray label cleared in memory.
            assert ann.get_frames("9") == []
            # Persisted to disk too.
            saved = json.loads(Path(ann.fname).read_text())
            assert saved.get("9", {}) == {}
        finally:
            plt.close(fig)

    def test_strip_strays_false_preserves_stray_annotations(self, tmp_path):
        import matplotlib.pyplot as plt
        ann, fig, issues, fake_video = self._setup(tmp_path)
        try:
            apply_pre_flight_remediations(
                annotations={"iteration-1": ann},
                video_fname=fake_video,
                issues=issues,
                make_annotation_file_name=lambda *_a, **_k: ann.fname,
                strip_strays=False,
            )
            assert ann.get_frames("0") == [10, 11, 12]
            assert ann.get_frames("1") == [10, 11, 12]
            # Stray preserved both in memory and on disk.
            assert ann.get_frames("9") == [11]
            saved = json.loads(Path(ann.fname).read_text())
            assert saved["9"] == {"11": [9.0, 9.0]}
        finally:
            plt.close(fig)

    def test_strip_strays_default_is_true(self, tmp_path):
        # The default is "strip" -- the user has to opt INTO keeping
        # extra labels by checking the modal checkbox.
        import matplotlib.pyplot as plt
        ann, fig, issues, fake_video = self._setup(tmp_path)
        try:
            apply_pre_flight_remediations(
                annotations={"iteration-1": ann},
                video_fname=fake_video,
                issues=issues,
                make_annotation_file_name=lambda *_a, **_k: ann.fname,
            )
            assert ann.get_frames("9") == []
        finally:
            plt.close(fig)
