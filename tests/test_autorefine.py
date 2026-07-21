"""Tests for :mod:`dustrack.autorefine`.

Detection is pure numpy over a prediction table, so it is tested
directly. Repair needs LK, which is faked -- what matters is not LK's
arithmetic (tested in ``test_pointtracking``) but the decisions built on
top of it: which gaps are attempted, which repairs are trusted, and
which frames are allowed to become training data.

The bias throughout is toward *refusing*. Auto-generated labels that are
wrong are worse than no labels here: bistability accrues from cumulative
training over labels that disagree across non-adjacent repeats, so a
pass that invented labels in the ambiguous places would manufacture the
very pathology it exists to relieve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dustrack import autorefine as ar


def make_predictions(likelihood_per_label, xy=None):
    """A DLC-format table from ``{bodypart: [likelihood, ...]}``."""
    bodyparts = list(likelihood_per_label)
    n = len(next(iter(likelihood_per_label.values())))
    cols = pd.MultiIndex.from_product(
        [["scorerX"], bodyparts, ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    data = {}
    for bp in bodyparts:
        x, y = (xy or {}).get(bp, (np.arange(n) * 0.01, np.arange(n) * 0.01))
        data[(("scorerX"), bp, "x")] = x
        data[("scorerX", bp, "y")] = y
        data[("scorerX", bp, "likelihood")] = np.asarray(
            likelihood_per_label[bp], dtype=float
        )
    return pd.DataFrame(
        {c: data[c] for c in cols}, columns=cols, index=pd.Index(range(n))
    )


def lik_with_gap(n, start, end, value=0.1, base=1.0):
    lik = np.full(n, base)
    lik[start : end + 1] = value
    return lik


# --------------------------------------------------------------------- #
# Detection                                                             #
# --------------------------------------------------------------------- #
class TestFindGaps:
    def test_finds_a_bracketed_gap(self):
        df = make_predictions({"point0": lik_with_gap(200, 100, 109)})
        r = ar.find_gaps(df)
        assert len(r.gaps) == 1
        g = r.gaps[0]
        assert (g.start, g.end, g.label) == (100, 109, "0")
        assert g.length == 10

    def test_label_is_the_annotation_name_not_the_bodypart(self):
        """A repaired layer has to line up with the rest of the session,
        which names labels '0'/'1', not 'point0'/'point1'."""
        df = make_predictions({"point1": lik_with_gap(200, 50, 55)})
        assert ar.find_gaps(df).gaps[0].label == "1"

    def test_anchors_come_from_the_confident_neighbours(self):
        n = 200
        x = np.zeros(n)
        y = np.zeros(n)
        x[99], y[99] = 10.0, 20.0        # last confident before
        x[110], y[110] = 30.0, 40.0      # first confident after
        df = make_predictions(
            {"point0": lik_with_gap(n, 100, 109)}, xy={"point0": (x, y)}
        )
        g = ar.find_gaps(df).gaps[0]
        assert g.anchor_before == (10.0, 20.0)
        assert g.anchor_after == (30.0, 40.0)

    def test_long_run_is_declined_not_repaired(self):
        """The bistable case excludes itself: a sustained run has no
        confident bracket within reach, and neither lane is obviously
        right."""
        df = make_predictions({"point0": lik_with_gap(2000, 500, 1400)})
        r = ar.find_gaps(df)
        assert r.gaps == []
        assert len(r.rejected["too_long"]) == 1

    def test_mild_uncertainty_is_not_severe_enough(self):
        """A dip to 0.8 is not the model being lost."""
        df = make_predictions({"point0": lik_with_gap(200, 100, 109, value=0.8)})
        r = ar.find_gaps(df, low=0.6, high=0.9)
        assert r.gaps == []
        assert len(r.rejected["not_severe"]) == 1

    def test_gap_at_video_start_is_declined(self):
        """No bracket on one side -- nothing to track from."""
        df = make_predictions({"point0": lik_with_gap(200, 0, 5)})
        r = ar.find_gaps(df)
        assert r.gaps == []
        assert "at_video_edge" in r.rejected

    def test_gap_at_video_end_is_declined(self):
        df = make_predictions({"point0": lik_with_gap(200, 194, 199)})
        r = ar.find_gaps(df)
        assert r.gaps == []
        assert "at_video_edge" in r.rejected

    def test_short_confident_island_is_declined(self):
        """Two bad runs separated by 1 confident frame: that frame is
        too thin to anchor either side."""
        lik = np.full(300, 1.0)
        lik[100:120] = 0.1
        lik[121:140] = 0.1          # frame 120 alone is confident
        df = make_predictions({"point0": lik})
        r = ar.find_gaps(df, bracket=5)
        assert "bracket_too_short" in r.rejected

    def test_one_threshold_defines_gap_and_bracket(self):
        """Frames in the middle band must not orphan a gap.

        Defining gaps by <0.6 and brackets by >=0.9 left 0.6-0.9 frames
        satisfying neither, which split one real problem into runs that
        each failed to find an anchor -- 6 repairable gaps out of 1353
        on real data. A run of middling frames adjacent to a lost run
        must therefore be part of the gap, not a broken bracket.
        """
        lik = np.full(300, 1.0)
        lik[100:110] = 0.1          # lost
        lik[110:120] = 0.75         # middling -- neither low nor confident
        df = make_predictions({"point0": lik})
        r = ar.find_gaps(df, low=0.6, high=0.9)
        assert len(r.gaps) == 1
        assert r.gaps[0].end == 119   # the middling frames joined the gap

    def test_multiple_labels_reported_separately(self):
        df = make_predictions(
            {
                "point0": lik_with_gap(300, 100, 105),
                "point1": lik_with_gap(300, 200, 204),
            }
        )
        r = ar.find_gaps(df)
        assert {g.label for g in r.gaps} == {"0", "1"}
        assert r.per_label["0"]["n_gaps"] == 1

    def test_clean_video_yields_nothing(self):
        df = make_predictions({"point0": np.full(500, 1.0)})
        r = ar.find_gaps(df)
        assert r.gaps == []

    def test_summary_mentions_declined_work(self):
        """Declined runs are the human's queue -- hiding them would
        misrepresent how much is left."""
        df = make_predictions({"point0": lik_with_gap(2000, 500, 1400)})
        assert "declined" in ar.find_gaps(df).summary()


# --------------------------------------------------------------------- #
# Repair                                                                #
# --------------------------------------------------------------------- #
class FakeAnn:
    """Minimal stand-in: repair only needs ``video``, ``labels``, ``fname``."""

    class _Reader:
        """VideoAnnotation probes ``len`` on the reader it is handed."""

        def __len__(self):
            return 1000

    def __init__(self, labels=("0",)):
        self.labels = list(labels)
        self.video = FakeAnn._Reader()
        self.fname = "vid_annotations_pred.json"


def fake_lk(disagreement, n_pts=1):
    """Patch target returning (rstc, forward, reverse) with a chosen
    per-frame forward/reverse separation."""

    def _lk(video, start, end, start_pts, end_pts, return_paths=False, **kw):
        n = end - start + 1
        rstc = np.zeros((n, n_pts, 2), dtype=float)
        fwd = np.zeros((n, n_pts, 2), dtype=float)
        rev = np.zeros((n, n_pts, 2), dtype=float)
        d = np.asarray(disagreement, dtype=float)
        if d.ndim == 0:
            d = np.full(n - 2, float(d))
        rev[1 : 1 + len(d), 0, 0] = d      # separation along x
        if not return_paths:
            return rstc
        return rstc, fwd, rev

    return _lk


class TestRepair:
    def test_records_per_frame_disagreement(self, monkeypatch):
        monkeypatch.setattr(ar, "lucas_kanade_rstc", fake_lk([0.5, 1.5, 3.0]))
        df = make_predictions({"point0": lik_with_gap(200, 100, 102)})
        rep = ar.find_gaps(df)
        res = ar.repair(rep, FakeAnn())
        assert res.disagreement["0"] == {100: 0.5, 101: 1.5, 102: 3.0}

    def test_gap_rejected_outright_when_tracks_never_converge(self, monkeypatch):
        monkeypatch.setattr(ar, "lucas_kanade_rstc", fake_lk(500.0))
        df = make_predictions({"point0": lik_with_gap(200, 100, 105)})
        rep = ar.find_gaps(df)
        res = ar.repair(rep, FakeAnn(), max_disagreement=50.0)
        assert res.repaired == []
        assert len(res.untrusted) == 1
        assert res.corrections.data["0"] == {}

    def test_corrections_cover_every_gap_frame(self, monkeypatch):
        monkeypatch.setattr(ar, "lucas_kanade_rstc", fake_lk(0.1))
        df = make_predictions({"point0": lik_with_gap(200, 100, 109)})
        res = ar.repair(ar.find_gaps(df), FakeAnn())
        assert sorted(res.corrections.data["0"]) == list(range(100, 110))

    def test_needs_a_video_reader(self):
        ann = FakeAnn()
        ann.video = None
        with pytest.raises(ValueError, match="video reader"):
            ar.repair(ar.GapReport(), ann)


# --------------------------------------------------------------------- #
# Training-frame selection                                              #
# --------------------------------------------------------------------- #
def _result_with(frames_and_disagreement, gap_span=(100, 140)):
    res = ar.RepairResult()
    label = "0"
    data = {label: {f: [1.0, 2.0] for f in frames_and_disagreement}}
    res.disagreement = {label: dict(frames_and_disagreement)}
    res.repaired = [
        ar.Gap(
            label=label,
            start=gap_span[0],
            end=gap_span[1],
            anchor_before=(0.0, 0.0),
            anchor_after=(0.0, 0.0),
            min_likelihood=0.1,
        )
    ]

    class _Corr:
        def __init__(self, d):
            self.data = d
            self.labels = list(d)
            self.fname = "x_autorefine.json"
            self.video = None

    res.corrections = _Corr(data)
    return res


class TestSelectTrainingFrames:
    def test_untrusted_frames_are_excluded(self):
        res = _result_with({100: 0.5, 110: 9.0, 120: 0.8, 130: 20.0})
        layer = ar.select_training_frames(
            res, max_disagreement=2.0, per_gap=10, min_spacing=1
        )
        assert sorted(layer.data["0"]) == [100, 120]

    def test_trust_gate_is_tighter_than_the_repair_cap(self):
        """A label several px off teaches a position that is not there;
        the default keeps only tightly-determined frames."""
        res = _result_with({100: 3.0, 110: 1.0})
        layer = ar.select_training_frames(res, per_gap=10, min_spacing=1)
        assert sorted(layer.data["0"]) == [110]

    def test_decimates_within_a_gap(self):
        """Consecutive frames in one gap are near-duplicates."""
        res = _result_with({f: 0.1 for f in range(100, 141)})
        layer = ar.select_training_frames(res, per_gap=2, min_spacing=1)
        assert len(layer.data["0"]) == 2

    def test_min_spacing_drops_neighbours(self):
        res = _result_with({100: 0.1, 101: 0.1, 102: 0.1})
        layer = ar.select_training_frames(
            res, per_gap=3, min_spacing=10
        )
        assert sorted(layer.data["0"]) == [100]

    def test_max_frames_caps_the_total(self):
        res = ar.RepairResult()
        frames = {f: 0.1 for f in range(0, 2000, 10)}
        res.disagreement = {"0": frames}
        res.repaired = [
            ar.Gap("0", f, f, (0.0, 0.0), (0.0, 0.0), 0.1) for f in frames
        ]

        class _Corr:
            data = {"0": {f: [1.0, 2.0] for f in frames}}
            labels = ["0"]
            fname = "x.json"
            video = None

        res.corrections = _Corr()
        layer = ar.select_training_frames(
            res, per_gap=1, min_spacing=1, max_frames=20
        )
        assert len(layer.data["0"]) <= 22   # rounding slack

    def test_cap_spreads_across_the_video(self):
        """A budget spent on the first minute leaves the rest unrefined."""
        res = ar.RepairResult()
        frames = {f: 0.1 for f in range(0, 10000, 10)}
        res.disagreement = {"0": frames}
        res.repaired = [
            ar.Gap("0", f, f, (0.0, 0.0), (0.0, 0.0), 0.1) for f in frames
        ]

        class _Corr:
            data = {"0": {f: [1.0, 2.0] for f in frames}}
            labels = ["0"]
            fname = "x.json"
            video = None

        res.corrections = _Corr()
        layer = ar.select_training_frames(
            res, per_gap=1, min_spacing=1, max_frames=20
        )
        kept = sorted(layer.data["0"])
        assert kept[-1] > 8000     # reaches the end, not truncated early

    def test_nothing_to_select_from_raises(self):
        with pytest.raises(ValueError, match="run repair first"):
            ar.select_training_frames(ar.RepairResult())
