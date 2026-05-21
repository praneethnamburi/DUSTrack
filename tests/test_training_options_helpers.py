"""Tests for the Training options modal pure helpers.

``_default_training_options(dlcproject)`` builds the initial state
dict the modal seeds itself from. ``_training_options_to_train_iteration_kwargs(options)``
translates the user-modified state back to
:meth:`DLCProject.train_iteration` kwargs.

Both helpers are pure-Python (no Qt deps), so they can be tested
without instantiating the modal class. The Qt widget itself
(:class:`TrainingOptionsDialog`) is covered by manual smoke per the
EnhanceWidget / ConfirmOverlay precedent (synchronous-modal /
palette-aware Qt is painful to exercise headlessly).
"""
from types import SimpleNamespace

import pytest

from dustrack import dlcinterface
from dustrack.dlcinterface import (
    _default_training_options,
    _training_options_to_train_iteration_kwargs,
)


def _fake_project(*, snapshots):
    """Minimal DLCProject-shape stub.

    The helpers only read ``all_snapshots``; pass a dict
    ``{iteration_num: list[snapshot_num]}``. Empty list means
    "iteration exists but is untrained".
    """
    return SimpleNamespace(all_snapshots=snapshots)


@pytest.fixture
def dlc3(monkeypatch):
    monkeypatch.setattr(dlcinterface, "DLC3", True)
    return True


@pytest.fixture
def dlc2(monkeypatch):
    monkeypatch.setattr(dlcinterface, "DLC3", False)
    return False


# ---------------------------------------------------------------------------
# _default_training_options
# ---------------------------------------------------------------------------


class TestDefaultTrainingOptions:
    def test_no_trained_iterations_defaults_to_scratch(self, dlc3):
        # Iteration 0 exists but isn't trained.
        opts = _default_training_options(_fake_project(snapshots={0: []}))
        assert opts["refine_mode"] == "scratch"
        assert opts["source_iteration"] is None
        assert opts["trained_iterations"] == []
        assert opts["snapshots_by_iteration"] == {}

    def test_some_trained_iterations_defaults_to_in_project(self, dlc3):
        # Iterations 0 and 1 are trained, 2 is the current (untrained).
        opts = _default_training_options(
            _fake_project(snapshots={0: [50, 100], 1: [50, 100], 2: []})
        )
        assert opts["refine_mode"] == "in_project"
        # Latest TRAINED, not just latest -- 2 is untrained.
        assert opts["source_iteration"] == 1
        # Best snapshot is the default ("None" lets initialize_weights pick).
        assert opts["source_snapshot"] is None
        assert opts["trained_iterations"] == [0, 1]
        assert opts["snapshots_by_iteration"] == {0: [50, 100], 1: [50, 100]}

    def test_dlc3_maxiters_default_is_50(self, dlc3):
        opts = _default_training_options(_fake_project(snapshots={}))
        assert opts["maxiters"] == 50
        assert opts["is_dlc3"] is True

    def test_dlc2_maxiters_default_is_500000(self, dlc2):
        opts = _default_training_options(_fake_project(snapshots={}))
        assert opts["maxiters"] == 500000
        assert opts["is_dlc3"] is False

    def test_create_video_default_is_false(self, dlc3):
        opts = _default_training_options(_fake_project(snapshots={}))
        assert opts["create_video"] is False

    def test_external_snapshot_path_default_is_empty(self, dlc3):
        opts = _default_training_options(_fake_project(snapshots={}))
        assert opts["external_snapshot_path"] == ""

    def test_trained_iterations_sorted_ascending(self, dlc3):
        # Pass them out-of-order to make sure we sort.
        opts = _default_training_options(
            _fake_project(snapshots={2: [50], 0: [50], 1: [50]})
        )
        assert opts["trained_iterations"] == [0, 1, 2]
        # source_iteration is the LAST one (latest trained).
        assert opts["source_iteration"] == 2

    def test_snapshots_by_iteration_only_includes_trained(self, dlc3):
        opts = _default_training_options(
            _fake_project(snapshots={0: [50], 1: [], 2: [100, 200]})
        )
        # Iteration 1 has no snapshots -- excluded.
        assert set(opts["snapshots_by_iteration"].keys()) == {0, 2}
        assert opts["snapshots_by_iteration"][0] == [50]
        assert opts["snapshots_by_iteration"][2] == [100, 200]


# ---------------------------------------------------------------------------
# _training_options_to_train_iteration_kwargs
# ---------------------------------------------------------------------------


def _user_choices(mode, **overrides):
    """Build a fully-populated options dict (modal state after user clicks Train).

    Mirrors the shape :func:`_default_training_options` returns plus
    the radio/combo/spinbox/checkbox values the dialog mutates before
    returning.
    """
    base = {
        "refine_mode": mode,
        "source_iteration": None,
        "source_snapshot": None,
        "external_snapshot_path": "",
        "maxiters": 50,
        "create_video": False,
        "trained_iterations": [],
        "snapshots_by_iteration": {},
        "is_dlc3": True,
    }
    base.update(overrides)
    return base


class TestPayloadShape:
    def test_scratch_payload_drops_source_keys(self):
        choices = _user_choices("scratch", maxiters=200, create_video=True)
        kwargs = _training_options_to_train_iteration_kwargs(choices)
        assert kwargs == {
            "refine_mode": "scratch",
            "maxiters": 200,
            "create_video": True,
        }
        # No source / external keys should leak through.
        assert "source_iteration" not in kwargs
        assert "source_snapshot" not in kwargs
        assert "external_snapshot_path" not in kwargs

    def test_scratch_drops_source_keys_even_when_dict_carries_them(self):
        # Defensive: the modal might still have source_iteration filled
        # in from a previous radio selection. Translation must drop it
        # because train_iteration's validator rejects non-None values
        # in scratch mode.
        choices = _user_choices(
            "scratch",
            source_iteration=1,  # leftover from previous radio state
            source_snapshot=100,
            external_snapshot_path="/old/path.pt",
        )
        kwargs = _training_options_to_train_iteration_kwargs(choices)
        assert "source_iteration" not in kwargs
        assert "source_snapshot" not in kwargs
        assert "external_snapshot_path" not in kwargs

    def test_in_project_payload_forwards_source_args(self):
        choices = _user_choices(
            "in_project",
            source_iteration=2,
            source_snapshot=200,
            maxiters=100,
        )
        kwargs = _training_options_to_train_iteration_kwargs(choices)
        assert kwargs == {
            "refine_mode": "in_project",
            "source_iteration": 2,
            "source_snapshot": 200,
            "maxiters": 100,
            "create_video": False,
        }
        # external_snapshot_path must NOT be in the payload -- the
        # validator would reject it.
        assert "external_snapshot_path" not in kwargs

    def test_in_project_snapshot_none_forwarded_as_none(self):
        # "best (auto)" combo entry maps to None userData. The
        # translation must preserve None (not drop the key) so
        # train_iteration's initialize_weights uses its default
        # (best snapshot).
        choices = _user_choices(
            "in_project",
            source_iteration=1,
            source_snapshot=None,
        )
        kwargs = _training_options_to_train_iteration_kwargs(choices)
        assert kwargs["source_snapshot"] is None
        assert "source_snapshot" in kwargs

    def test_external_payload_forwards_path_only(self):
        choices = _user_choices(
            "external",
            external_snapshot_path="/path/to/external.pt",
            maxiters=75,
            create_video=True,
        )
        kwargs = _training_options_to_train_iteration_kwargs(choices)
        assert kwargs == {
            "refine_mode": "external",
            "external_snapshot_path": "/path/to/external.pt",
            "maxiters": 75,
            "create_video": True,
        }
        # source_* keys must NOT be in the payload.
        assert "source_iteration" not in kwargs
        assert "source_snapshot" not in kwargs

    def test_unknown_refine_mode_raises(self):
        choices = _user_choices("scratch")
        choices["refine_mode"] = "bogus"
        with pytest.raises(ValueError, match="unknown refine_mode"):
            _training_options_to_train_iteration_kwargs(choices)


# ---------------------------------------------------------------------------
# End-to-end shape: defaults -> translation -> train_iteration-compatible kwargs
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_no_trained_iterations_defaults_translate_to_scratch_kwargs(self, dlc3):
        # Simulates the first-time training scenario: no trained
        # iterations, user accepts defaults and clicks Train.
        opts = _default_training_options(_fake_project(snapshots={0: []}))
        kwargs = _training_options_to_train_iteration_kwargs(opts)
        assert kwargs["refine_mode"] == "scratch"
        assert kwargs["maxiters"] == 50
        assert kwargs["create_video"] is False
        assert "source_iteration" not in kwargs

    def test_trained_iterations_defaults_translate_to_in_project_kwargs(self, dlc3):
        # Simulates the typical refine scenario: user accepts default
        # source iteration (latest trained) + best snapshot, clicks Train.
        opts = _default_training_options(
            _fake_project(snapshots={0: [50, 100], 1: [50, 100]})
        )
        kwargs = _training_options_to_train_iteration_kwargs(opts)
        assert kwargs["refine_mode"] == "in_project"
        assert kwargs["source_iteration"] == 1
        assert kwargs["source_snapshot"] is None  # "best (auto)"
        assert kwargs["maxiters"] == 50
