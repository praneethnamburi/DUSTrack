"""Tests for :meth:`DUSTrack.decimate_annotations_in_interval`.

Starter form of the "general-model workflow" decimation feature:
prune incomplete frames in the selected interval, then halve the
remaining (complete) frames by even-stride sampling. Frame-level --
dropped frames remove every label in the layer's schema (matches the
Train preflight's completeness rule via
:func:`_preflight.scan_incomplete_frames`). The DINOv3-feature
farthest-point-sampling variant is deferred.

The method only touches ``self.get_selected_interval()``,
``self.ann``, and ``self.update()`` -- a SimpleNamespace fake is
enough to drive it against a real :class:`VideoAnnotation`, no GUI
session needed.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import dustrack
from dustrack import DUSTrack


def _make_fake(ann, interval, dlcproject=None):
    """Bind the decimation method to a fake exposing only the surface
    it reads. ``dlcproject=None`` exercises the project-unaware path;
    pass a SimpleNamespace with a ``config`` dict to exercise the
    project-aware path."""
    return SimpleNamespace(
        ann=ann,
        get_selected_interval=lambda: interval,
        update=lambda: None,
        _dlcproject=dlcproject,
    )


def _fake_dlcproject(bodyparts):
    """Minimal DLCProject stand-in -- decimation only reads
    ``config["bodyparts"]``."""
    return SimpleNamespace(config={"bodyparts": list(bodyparts)})


def _ann_with_frames(frames_per_label: dict[str, list[int]]):
    """Build a VideoAnnotation with the given per-label frame lists.

    Labels may be non-contiguous (e.g. ``{"0": ..., "1": ..., "9":
    ...}``) -- the helper bootstraps with ``n_labels=1`` and uses
    ``add_label`` to declare each missing slot before annotating.

    All annotated frames get the same dummy location -- decimation
    only cares about presence/absence per (label, frame), not the
    value.
    """
    ann = dustrack.VideoAnnotation(n_labels=1)
    for label in sorted(frames_per_label.keys(), key=int):
        if label not in ann.labels:
            ann.add_label(label=label)
    for label, frames in frames_per_label.items():
        for frame_number in frames:
            ann.add(location=[1.0, 1.0], label=label, frame_number=frame_number)
    return ann


class TestDecimateAnnotationsInInterval:
    def test_drops_every_other_when_all_complete(self):
        # Two labels, both annotated on the same frames 10..19 ->
        # every frame complete. Decimation keeps frames at sorted
        # indices 0,2,4,6,8 -> 10,12,14,16,18 (on every label).
        frames = list(range(10, 20))
        ann = _ann_with_frames({"0": frames, "1": frames})
        fake = _make_fake(ann, interval=(10, 19))
        DUSTrack.decimate_annotations_in_interval(fake)
        assert ann.get_frames("0") == [10, 12, 14, 16, 18]
        assert ann.get_frames("1") == [10, 12, 14, 16, 18]

    def test_prunes_incomplete_before_halving(self):
        # Label "0" annotated on every frame 10..19; label "1"
        # missing on the odd frames (11, 13, 15, 17, 19). So the
        # complete frames in the interval are 10, 12, 14, 16, 18.
        # Decimation: prune incomplete (11,13,15,17,19 fully gone),
        # then halve complete -> keep indices 0,2,4 -> 10, 14, 18.
        ann = _ann_with_frames({
            "0": list(range(10, 20)),
            "1": [10, 12, 14, 16, 18],
        })
        fake = _make_fake(ann, interval=(10, 19))
        DUSTrack.decimate_annotations_in_interval(fake)
        assert ann.get_frames("0") == [10, 14, 18]
        assert ann.get_frames("1") == [10, 14, 18]

    def test_frames_outside_interval_untouched(self):
        # Frames straddle the interval [20, 29]; out-of-interval
        # frames must survive on every label, regardless of whether
        # they would have been "incomplete" or "decimated" if they
        # were inside.
        in_interval = list(range(20, 30))
        outside = [5, 7, 9, 35, 40, 45]
        ann = _ann_with_frames({
            "0": sorted(in_interval + outside),
            "1": sorted(in_interval + outside),
        })
        fake = _make_fake(ann, interval=(20, 29))
        DUSTrack.decimate_annotations_in_interval(fake)
        for label in ("0", "1"):
            kept = ann.get_frames(label)
            assert sorted(f for f in kept if 20 <= f <= 29) == [20, 22, 24, 26, 28]
            assert sorted(f for f in kept if f < 20 or f > 29) == outside

    def test_all_labels_removed_at_dropped_frame(self):
        # Frame-level semantic: when a frame is dropped (either as
        # incomplete or via even-stride), every label in the layer's
        # schema loses its annotation at that frame.
        ann = _ann_with_frames({
            "0": list(range(10, 20)),
            "1": list(range(10, 20)),
            "2": list(range(10, 20)),
        })
        fake = _make_fake(ann, interval=(10, 19))
        DUSTrack.decimate_annotations_in_interval(fake)
        # Survivors are the same across every label.
        expected = [10, 12, 14, 16, 18]
        assert ann.get_frames("0") == expected
        assert ann.get_frames("1") == expected
        assert ann.get_frames("2") == expected

    def test_noop_when_fewer_than_two_complete_in_interval(self):
        # Only a single complete frame in the interval -> nothing to
        # halve, and there are no incomplete frames either. Frame
        # survives untouched.
        ann = _ann_with_frames({
            "0": [12, 50],
            "1": [12, 50],
        })
        fake = _make_fake(ann, interval=(10, 19))
        DUSTrack.decimate_annotations_in_interval(fake)
        assert ann.get_frames("0") == [12, 50]
        assert ann.get_frames("1") == [12, 50]

    def test_noop_when_zero_in_interval(self):
        # No annotated frames in the interval on any label.
        ann = _ann_with_frames({
            "0": [5, 50, 60],
            "1": [5, 50, 60],
        })
        fake = _make_fake(ann, interval=(10, 19))
        DUSTrack.decimate_annotations_in_interval(fake)
        assert ann.get_frames("0") == [5, 50, 60]
        assert ann.get_frames("1") == [5, 50, 60]

    def test_prunes_incomplete_even_when_too_few_to_halve(self):
        # Single complete frame in the interval after pruning -> no
        # halving (fewer than 2 left), but the incomplete frames in
        # the interval are still removed across all labels.
        ann = _ann_with_frames({
            "0": [10, 11, 12],
            "1": [12],  # frames 10, 11 are incomplete; only 12 complete
        })
        fake = _make_fake(ann, interval=(10, 19))
        DUSTrack.decimate_annotations_in_interval(fake)
        assert ann.get_frames("0") == [12]
        assert ann.get_frames("1") == [12]

    def test_single_label_schema_treats_every_frame_complete(self):
        # Layer with only label "0" declared -> every annotated frame
        # is "complete" (no other labels to be missing). Decimation
        # halves them.
        ann = _ann_with_frames({"0": list(range(10, 20))})
        fake = _make_fake(ann, interval=(10, 19))
        DUSTrack.decimate_annotations_in_interval(fake)
        assert ann.get_frames("0") == [10, 12, 14, 16, 18]


class TestProjectAwareDecimation:
    """Project-aware mode: required-label set comes from DLC
    ``bodyparts`` instead of "labels with any annotation". Closes the
    project-unaware blindspot where an empty label is treated as a
    UI placeholder (right call when there's no external truth; wrong
    call when there IS one)."""

    def test_required_set_from_bodyparts_flags_missing(self):
        # Project declares two bodyparts (-> labels "0", "1"), but
        # the user has only annotated label "0" in the interval.
        # Project-unaware mode would treat label "1" as a placeholder
        # and halve label "0"; project-aware mode flags every frame
        # as incomplete (missing "1") and prunes them all.
        ann = _ann_with_frames({
            "0": list(range(10, 20)),
            "1": [],
        })
        proj = _fake_dlcproject(["point0", "point1"])  # -> ["0", "1"]
        fake = _make_fake(ann, interval=(10, 19), dlcproject=proj)
        DUSTrack.decimate_annotations_in_interval(fake)
        assert ann.get_frames("0") == []
        assert ann.get_frames("1") == []

    def test_stray_non_bodypart_label_at_complete_frame_survives(self):
        # Project bodyparts -> ["0", "1"]. Every frame in [10, 19]
        # has both 0 and 1 -> complete. A stray label "9" annotation
        # at frame 12 is NOT required, but frame 12 is complete and
        # passes the prune. Halving keeps frames 10, 12, 14, 16, 18.
        # Label-9 annotation at frame 12 survives because frame 12
        # survives.
        frames = list(range(10, 20))
        ann = _ann_with_frames({
            "0": frames,
            "1": frames,
            "9": [12],
        })
        proj = _fake_dlcproject(["point0", "point1"])
        fake = _make_fake(ann, interval=(10, 19), dlcproject=proj)
        DUSTrack.decimate_annotations_in_interval(fake)
        assert ann.get_frames("0") == [10, 12, 14, 16, 18]
        assert ann.get_frames("1") == [10, 12, 14, 16, 18]
        assert ann.get_frames("9") == [12]

    def test_stray_non_bodypart_label_at_incomplete_frame_pruned(self):
        # Project bodyparts -> ["0", "1"]. Frame 15 has only label
        # "9" annotated -- a stray. Required check finds 0 and 1
        # missing -> frame 15 is incomplete -> pruned across every
        # label on the layer, taking the stray "9" with it.
        ann = _ann_with_frames({
            "0": [10, 12, 14, 16, 18],
            "1": [10, 12, 14, 16, 18],
            "9": [15],
        })
        proj = _fake_dlcproject(["point0", "point1"])
        fake = _make_fake(ann, interval=(10, 19), dlcproject=proj)
        DUSTrack.decimate_annotations_in_interval(fake)
        # Complete frames before halving: [10, 12, 14, 16, 18]; frame
        # 15 was incomplete and dropped. Halving keeps indices 0,2,4
        # -> 10, 14, 18.
        assert ann.get_frames("0") == [10, 14, 18]
        assert ann.get_frames("1") == [10, 14, 18]
        assert ann.get_frames("9") == []

    def test_empty_bodyparts_falls_back_to_unaware_mode(self):
        # Edge: project loaded but ``bodyparts`` empty -- treat as if
        # no project (the wired path tests truthiness on bodyparts).
        # Label "1" is empty -> placeholder -> label "0" halves
        # through.
        ann = _ann_with_frames({
            "0": list(range(10, 20)),
            "1": [],
        })
        proj = _fake_dlcproject([])
        fake = _make_fake(ann, interval=(10, 19), dlcproject=proj)
        DUSTrack.decimate_annotations_in_interval(fake)
        assert ann.get_frames("0") == [10, 12, 14, 16, 18]
