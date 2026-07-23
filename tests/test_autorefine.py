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


# --------------------------------------------------------------------- #
# Blip adapter -- the confidently-wrong half                            #
# --------------------------------------------------------------------- #
class _FakeBlip:
    def __init__(self, label, start, end):
        self.label = label
        self.start = start
        self.end = end
        self.anchor_before = (1.0, 2.0)
        self.anchor_after = (3.0, 4.0)


class _FakeBlipReport:
    def __init__(self, blips):
        self.blips = blips


class TestGapsFromBlips:
    def test_confident_blips_are_kept(self):
        """The whole point: likelihood cannot see these.

        On real data 2120 of 3308 blips sat at likelihood >= 0.80, many
        at exactly 1.00 -- the confidence pass is blind to every one.
        """
        df = make_predictions({"point0": np.full(200, 1.0)})
        rep = ar.gaps_from_blips(_FakeBlipReport([_FakeBlip("0", 100, 102)]), df)
        assert len(rep.gaps) == 1
        assert rep.gaps[0].min_likelihood == 1.0

    def test_low_likelihood_blips_deferred_to_the_confidence_pass(self):
        """Both detectors firing on one run must not relabel it twice."""
        df = make_predictions({"point0": lik_with_gap(200, 100, 102)})
        rep = ar.gaps_from_blips(_FakeBlipReport([_FakeBlip("0", 100, 102)]), df)
        assert rep.gaps == []
        assert "also_low_likelihood" in rep.rejected

    def test_confident_only_can_be_disabled(self):
        df = make_predictions({"point0": lik_with_gap(200, 100, 102)})
        rep = ar.gaps_from_blips(
            _FakeBlipReport([_FakeBlip("0", 100, 102)]), df, confident_only=False
        )
        assert len(rep.gaps) == 1

    def test_anchors_carry_over(self):
        df = make_predictions({"point0": np.full(200, 1.0)})
        g = ar.gaps_from_blips(_FakeBlipReport([_FakeBlip("0", 50, 51)]), df).gaps[0]
        assert g.anchor_before == (1.0, 2.0)
        assert g.anchor_after == (3.0, 4.0)

    def test_works_without_predictions(self):
        rep = ar.gaps_from_blips(_FakeBlipReport([_FakeBlip("0", 5, 6)]))
        assert len(rep.gaps) == 1
        assert np.isnan(rep.gaps[0].min_likelihood)


# --------------------------------------------------------------------- #
# Cross-video selection                                                 #
# --------------------------------------------------------------------- #
def _res(frames_and_d, label="0"):
    res = ar.RepairResult()
    res.disagreement = {label: dict(frames_and_d)}

    class _Corr:
        data = {label: {f: [float(f), 0.0] for f in frames_and_d}}
        labels = [label]
        fname = "x.json"
        video = None

    res.corrections = _Corr()
    return res


class TestSelectAcrossVideos:
    def test_ranks_by_trust(self):
        results = {"a": _res({10: 5.0, 20: 0.1, 30: 2.0})}
        out = ar.select_across_videos(results, n=2, max_disagreement=10.0)
        assert sorted(out["a"]["0"]) == [20, 30]

    def test_quality_floor_excludes_the_rest(self):
        results = {"a": _res({10: 9.0, 20: 0.1})}
        out = ar.select_across_videos(results, n=5, max_disagreement=4.0)
        assert sorted(out["a"]["0"]) == [20]

    def test_budget_spreads_across_videos(self):
        """One model serves every trial; a round spent inside one video
        teaches that trial's appearance and no other."""
        results = {
            "a": _res({f: 0.01 for f in range(0, 500, 50)}),   # all excellent
            "b": _res({f: 1.0 for f in range(0, 500, 50)}),    # all merely good
        }
        out = ar.select_across_videos(results, n=4, max_disagreement=4.0)
        assert set(out) == {"a", "b"}
        assert len(out["b"]["0"]) >= 1     # b gets a share despite worse trust

    def test_a_video_with_nothing_good_forfeits_its_share(self):
        """The quota is a ceiling, not a reservation."""
        results = {
            "a": _res({f: 0.01 for f in range(0, 500, 50)}),
            "b": _res({10: 99.0}),          # nothing under the floor
        }
        out = ar.select_across_videos(results, n=4, max_disagreement=4.0)
        assert "b" not in out
        assert len(out["a"]["0"]) == 4      # a takes the whole budget

    def test_respects_min_spacing(self):
        results = {"a": _res({100: 0.1, 101: 0.1, 200: 0.1})}
        out = ar.select_across_videos(
            results, n=5, max_disagreement=4.0, min_spacing=10
        )
        assert sorted(out["a"]["0"]) == [100, 200]

    def test_returns_positions_not_just_frames(self):
        results = {"a": _res({42: 0.1})}
        out = ar.select_across_videos(results, n=1, max_disagreement=4.0)
        assert out["a"]["0"][42] == [42.0, 0.0]

    def test_empty_input(self):
        assert ar.select_across_videos({}, n=10) == {}

    def test_nothing_under_the_floor_is_an_empty_round(self):
        """Which is also the curriculum's natural stopping condition."""
        results = {"a": _res({10: 50.0})}
        assert ar.select_across_videos(results, n=10, max_disagreement=4.0) == {}

    def test_redistribution_still_respects_the_floor(self):
        """Forfeited quota must not drag in labels below the floor."""
        results = {
            "a": _res({f: 0.01 for f in range(0, 200, 50)}),   # 4 good
            "b": _res({10: 99.0}),                             # nothing usable
        }
        out = ar.select_across_videos(results, n=20, max_disagreement=4.0)
        assert "b" not in out
        assert len(out["a"]["0"]) == 4        # all of a's, and no more


# --------------------------------------------------------------------- #
# Confident-neighbour lookup                                            #
# --------------------------------------------------------------------- #
class TestConfidentRuns:
    def test_before_and_after(self):
        conf = np.array([False, True, False, False, True, False])
        before, after = ar._confident_runs(conf)
        assert before.tolist() == [-1, -1, 1, 1, 1, 4]
        assert after.tolist() == [1, 4, 4, 4, 6, 6]

    def test_none_confident(self):
        before, after = ar._confident_runs(np.zeros(4, dtype=bool))
        assert before.tolist() == [-1, -1, -1, -1]
        assert after.tolist() == [4, 4, 4, 4]


# --------------------------------------------------------------------- #
# complete_frames -- no half-labelled training frames                  #
# --------------------------------------------------------------------- #
class TestCompleteFrames:
    """A frame chosen for one point's gap reaches DLC with the other point
    NaN, which DLC trains toward low confidence -- the mechanism that
    collapsed s061 point0's likelihood to ~0.60 while its position held.
    Every selected frame must leave here carrying both points, filled from
    a confident prediction or a trusted LK estimate, or dropped."""

    def _df(self, n=200, *, p1_lik=1.0, p1_xy=(7.0, 8.0)):
        x1 = np.full(n, p1_xy[0])
        y1 = np.full(n, p1_xy[1])
        return make_predictions(
            {"point0": np.full(n, 1.0),
             "point1": np.asarray(p1_lik) if np.ndim(p1_lik) else np.full(n, p1_lik)},
            xy={"point1": (x1, y1)},
        )

    def test_already_complete_frame_is_kept_verbatim(self):
        sel = {"0": {50: [1.0, 2.0]}, "1": {50: [3.0, 4.0]}}
        out, st = ar.complete_frames(sel, self._df(), FakeAnn(("0", "1")))
        assert out["0"][50] == [1.0, 2.0] and out["1"][50] == [3.0, 4.0]
        assert st["kept"] == 1 and st["by_prediction"] == 0

    def test_missing_co_label_filled_from_confident_prediction(self):
        sel = {"0": {50: [1.0, 2.0]}}            # only point0 repaired
        out, st = ar.complete_frames(sel, self._df(), FakeAnn(("0", "1")))
        assert out["0"][50] == [1.0, 2.0]
        assert out["1"][50] == [7.0, 8.0]        # from the prediction
        assert st["kept"] == 1 and st["by_prediction"] == 1 and st["dropped"] == 0

    def test_frame_dropped_when_co_label_has_no_trustworthy_value(self):
        # point1 never confident -> no bracket -> LK refuses -> drop.
        df = self._df(p1_lik=0.1)
        sel = {"0": {50: [1.0, 2.0]}}
        out, st = ar.complete_frames(sel, df, FakeAnn(("0", "1")))
        assert out["0"] == {} and out["1"] == {}
        assert st["dropped"] == 1 and st["kept"] == 0

    def test_co_label_filled_by_trusted_lk_when_prediction_unsure(self, monkeypatch):
        monkeypatch.setattr(ar, "lucas_kanade_rstc", fake_lk(0.5))
        lik = np.full(200, 1.0)
        lik[50] = 0.1                            # point1 unsure only at 50
        df = self._df(p1_lik=lik)
        sel = {"0": {50: [1.0, 2.0]}}
        out, st = ar.complete_frames(sel, df, FakeAnn(("0", "1")))
        assert 50 in out["1"] and st["by_lk"] == 1 and st["kept"] == 1

    def test_frame_dropped_when_lk_tracks_disagree(self, monkeypatch):
        monkeypatch.setattr(ar, "lucas_kanade_rstc", fake_lk(10.0))
        lik = np.full(200, 1.0)
        lik[50] = 0.1
        df = self._df(p1_lik=lik)
        sel = {"0": {50: [1.0, 2.0]}}
        out, st = ar.complete_frames(sel, df, FakeAnn(("0", "1")))
        assert out["1"] == {} and st["dropped"] == 1


# --------------------------------------------------------------------- #
# nudge_labels -- refine toward a converged model, never chase a lane   #
# --------------------------------------------------------------------- #
class TestNudgeLabels:
    def _pred(self, x, y, lik, n=100, f=50):
        la = np.ones(n)
        la[f] = lik
        return make_predictions(
            {"point0": la}, xy={"point0": (np.full(n, x), np.full(n, y))}
        )

    def test_outlier_label_moves_onto_converged_prediction(self):
        lat = self._pred(100.0, 100.0, 1.0)
        prev = self._pred(100.2, 100.0, 1.0)          # within converge_tol
        out, st = ar.nudge_labels({"0": {50: [103.0, 100.0]}}, lat, prev)
        assert st["nudged"] == 1
        assert out["0"][50] == [100.0, 100.0]          # snapped (alpha=1)

    def test_partial_nudge_respects_alpha(self):
        lat = self._pred(100.0, 100.0, 1.0)
        prev = self._pred(100.0, 100.0, 1.0)
        out, _ = ar.nudge_labels(
            {"0": {50: [104.0, 100.0]}}, lat, prev, alpha=0.5
        )
        assert out["0"][50] == [102.0, 100.0]

    def test_label_already_on_target_is_left_alone(self):
        lat = self._pred(100.0, 100.0, 1.0)
        prev = self._pred(100.0, 100.0, 1.0)
        out, st = ar.nudge_labels({"0": {50: [100.3, 100.0]}}, lat, prev)
        assert st["on_target"] == 1 and out["0"][50] == [100.3, 100.0]

    def test_far_outlier_is_flagged_not_moved(self):
        """Confidence is not licence to drag a label a long way -- a
        confidently-wrong bistable lane scores exactly here."""
        lat = self._pred(100.0, 100.0, 1.0)
        prev = self._pred(100.0, 100.0, 1.0)
        out, st = ar.nudge_labels({"0": {50: [120.0, 100.0]}}, lat, prev)
        assert st["flagged"] == 1 and out["0"][50] == [120.0, 100.0]
        assert st["flagged_frames"] == {"0": [50]}

    def test_disagreeing_snapshots_are_not_converged(self):
        lat = self._pred(100.0, 100.0, 1.0)
        prev = self._pred(106.0, 100.0, 1.0)          # 6 px apart -> not converged
        out, st = ar.nudge_labels({"0": {50: [103.0, 100.0]}}, lat, prev)
        assert st["unconverged"] == 1 and out["0"][50] == [103.0, 100.0]

    def test_low_confidence_is_not_converged(self):
        lat = self._pred(100.0, 100.0, 0.5)           # unsure
        prev = self._pred(100.0, 100.0, 1.0)
        out, st = ar.nudge_labels({"0": {50: [103.0, 100.0]}}, lat, prev)
        assert st["unconverged"] == 1 and out["0"][50] == [103.0, 100.0]


# --------------------------------------------------------------------- #
# select_flow_blips -- rank by SIZE of error, not by trust               #
# --------------------------------------------------------------------- #
class TestSelectFlowBlips:
    """The mirror image of select_across_videos: here the ranking is
    inverted -- every candidate is already trusted, so the biggest error is
    the most valuable label, not the safest one."""

    def _res(self, corr_res):
        from dustrack.blip import FlowBlipResult
        corrections = {lab: {f: xy for f, (xy, r) in d.items()}
                       for lab, d in corr_res.items()}
        residual = {lab: {f: r for f, (xy, r) in d.items()}
                    for lab, d in corr_res.items()}
        return FlowBlipResult(corrections=corrections, residual=residual)

    def test_biggest_error_wins(self):
        results = {"a": self._res({"0": {10: ([1, 1], 3.0), 50: ([2, 2], 30.0)}})}
        out = ar.select_flow_blips(results, n=1)
        assert out == {"a": {"0": {50: [2.0, 2.0]}}}

    def test_min_residual_floor(self):
        results = {"a": self._res({"0": {10: ([1, 1], 2.0)}})}
        assert ar.select_flow_blips(results, n=5, min_residual=5.0) == {}

    def test_spacing_within_video(self):
        results = {"a": self._res({"0": {10: ([0, 0], 20.0), 12: ([0, 0], 19.0),
                                         40: ([0, 0], 18.0)}})}
        out = ar.select_flow_blips(results, n=3, min_spacing=10)
        assert sorted(out["a"]["0"]) == [10, 40]        # 12 too close to 10

    def test_per_video_cap_spreads_across_videos(self):
        results = {
            "a": self._res({"0": {i: ([0, 0], 100.0 - i) for i in range(0, 60, 10)}}),
            "b": self._res({"0": {5: ([0, 0], 1.0)}}),
        }
        out = ar.select_flow_blips(results, n=2)
        assert "a" in out and "b" in out               # cap=1 each, b not starved
