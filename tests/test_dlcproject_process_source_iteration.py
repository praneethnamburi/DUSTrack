"""Tests for ``DLCProject.process`` source_iteration kwarg routing.

These are kwarg-routing tests, not end-to-end DLC integration tests --
the heavy operations (``extract_frames`` / ``create_training_dataset``
/ ``train`` / ``evaluate`` / ``analyze_videos`` / ``increment_iteration``)
are mocked. The single behaviour under test is: given a particular
combination of ``refine`` / ``source_iteration`` / ``source_snapshot``
kwargs, does ``DLCProject.process`` (a) raise on a bad source_iteration
early, and (b) pass the right kwargs to ``initialize_weights``.

A ``_StubDLCProject`` subclass bypasses the real ``__init__`` (which
needs a live DeepLabCut install + a real project on disk) and provides
the minimum surface ``process`` reads -- just enough to drive the
refine branch deterministically.
"""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dustrack.dlcinterface import DLCProject


class _StubDLCProject(DLCProject):
    """Lightweight DLCProject for kwarg-routing tests.

    Bypasses the real ``__init__`` (which requires DLC + a real project
    on disk). Overrides the properties / methods that ``process`` reads
    so the refine branch can run end-to-end without touching the filesystem
    or DLC. ``initialize_weights`` and the heavy training/eval ops are
    captured as ``MagicMock`` so tests can assert on their call args.
    """

    def __init__(self, *, latest=3, latest_trained=False, trained_iters=(0, 1, 2)):
        # Deliberately do NOT call super().__init__ -- it needs DLC and
        # a real config.yaml.
        self._latest = latest
        self._latest_trained = latest_trained
        self._trained_iters = set(trained_iters)
        self._current = latest

        # Capture initialize_weights calls -- this is the routing under test.
        self.initialize_weights = MagicMock(name="initialize_weights")
        # Heavy ops we don't care about -- just no-op them.
        self.extract_frames = MagicMock(name="extract_frames")
        self.increment_iteration = MagicMock(name="increment_iteration")
        self.create_training_dataset = MagicMock(name="create_training_dataset")
        self.train = MagicMock(name="train")
        self.evaluate = MagicMock(name="evaluate", return_value=self)
        self.analyze_videos = MagicMock(name="analyze_videos", return_value=self)

    @property
    def paths(self):
        # Real Path so the ``paths['training_data'] / f'iteration-{N}'``
        # division operator works. The path doesn't have to exist --
        # os.path.exists returning False is fine and triggers
        # create_training_dataset (which is mocked).
        return {"training_data": Path("/_nonexistent_training_data_for_tests_")}

    @property
    def current_iteration(self):
        return self._current

    @current_iteration.setter
    def current_iteration(self, value):
        if value == "latest":
            self._current = self._latest
        elif value == "next":
            self._current = self._latest + 1 if self._latest_trained else self._latest
        else:
            assert isinstance(value, int)
            self._current = value

    @property
    def latest_iteration(self):
        return self._latest

    def latest_iteration_is_trained(self):
        return self._latest_trained

    def current_iteration_is_trained(self):
        # Force the training branch in process(); the train() mock no-ops.
        return False

    def iteration_is_trained(self, iteration_num):
        return iteration_num in self._trained_iters

    @property
    def all_snapshots(self):
        return {
            i: ([100] if i in self._trained_iters else [])
            for i in range(self._latest + 1)
        }


class TestSourceIterationValidation:
    def test_untrained_source_iteration_raises(self):
        p = _StubDLCProject(trained_iters=(0, 1, 2))
        with pytest.raises(ValueError, match="not a trained iteration"):
            p.process(source_iteration=5)
        # initialize_weights must never be called when validation fails.
        p.initialize_weights.assert_not_called()

    def test_non_int_source_iteration_raises(self):
        p = _StubDLCProject(trained_iters=(0, 1, 2))
        with pytest.raises(AssertionError):
            p.process(source_iteration="2")

    def test_trained_source_iteration_passes_validation(self):
        p = _StubDLCProject(trained_iters=(0, 1, 2))
        # Must not raise.
        p.process(source_iteration=1)


class TestRefineKwargRouting:
    def test_refine_true_no_kwargs_passes_none_none(self):
        """Existing default: refine=True, no source args → initialize_weights
        is called with both kwargs None (let it pick its own defaults =
        second-to-last + best snapshot).
        """
        p = _StubDLCProject(trained_iters=(0, 1, 2), latest=3, latest_trained=False)
        p.process(refine=True)
        p.initialize_weights.assert_called_once_with(
            source_iteration=None, source_snapshot=None
        )

    def test_refine_true_source_snapshot_only_picks_second_to_last(self):
        """Backward-compat: when only source_snapshot is given, process
        forces source_iteration to second-to-last (preserves pre-extension
        behavior).
        """
        p = _StubDLCProject(trained_iters=(0, 1, 2), latest=3, latest_trained=False)
        p.process(refine=True, source_snapshot=200)
        # latest=3, latest_trained=False → source_iteration = 3 - 1 = 2.
        p.initialize_weights.assert_called_once_with(
            source_iteration=2, source_snapshot=200
        )

    def test_refine_true_source_iteration_passes_through(self):
        """New: explicit source_iteration is passed through to
        initialize_weights verbatim, overriding the second-to-last default.
        """
        p = _StubDLCProject(trained_iters=(0, 1, 2), latest=3, latest_trained=False)
        p.process(refine=True, source_iteration=0)
        p.initialize_weights.assert_called_once_with(
            source_iteration=0, source_snapshot=None
        )

    def test_refine_true_both_source_args_pass_through(self):
        """source_iteration + source_snapshot together -- both pass through
        verbatim. The backward-compat second-to-last computation does NOT
        override the explicit source_iteration.
        """
        p = _StubDLCProject(trained_iters=(0, 1, 2), latest=3, latest_trained=False)
        p.process(refine=True, source_iteration=1, source_snapshot=150)
        p.initialize_weights.assert_called_once_with(
            source_iteration=1, source_snapshot=150
        )

    def test_refine_false_skips_initialize_weights(self):
        """Start-from-scratch path: no initialize_weights call."""
        p = _StubDLCProject(trained_iters=(0, 1, 2), latest=3, latest_trained=False)
        p.process(refine=False)
        p.initialize_weights.assert_not_called()

    def test_refine_string_path_skips_initialize_weights(self):
        """External .pt path: refine=<str> routes through train(snapshot_path=)
        and bypasses initialize_weights entirely.
        """
        p = _StubDLCProject(trained_iters=(0, 1, 2), latest=3, latest_trained=False)
        p.process(refine="/path/to/external/snapshot.pt")
        p.initialize_weights.assert_not_called()
