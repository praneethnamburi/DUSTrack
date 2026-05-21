"""Tests for ``DLCProject.train_iteration`` and its helpers.

``train_iteration`` is the explicit-args sibling of ``process()`` --
the UI Training options modal calls it once the user has picked
refine mode + source + epochs + create_video. Unlike ``process()``
(state-inference + sane defaults for CLI ergonomics), this method
performs strict validation per ``refine_mode`` and dispatches DLC2
vs DLC3 explicitly:

* In-project refine → :meth:`initialize_weights` (both DLC versions).
* External refine on DLC3 → ``train_network(snapshot_path=...)``.
* External refine on DLC2 →
  :meth:`_initialize_weights_from_external_path` edits pose_cfg's
  ``init_weights`` to the external path.

These are kwarg-routing tests, not end-to-end DLC integration tests.
The heavy operations are mocked; the single behaviour under test is
which method ``train_iteration`` calls (and with which args) given
a particular argument combination.

A ``_StubDLCProject`` subclass bypasses the real ``__init__`` (which
needs DLC + a real project on disk) and provides the minimum
surface ``train_iteration`` reads.
"""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dustrack import dlcinterface
from dustrack.dlcinterface import DLCProject


class _StubDLCProject(DLCProject):
    """Lightweight DLCProject for kwarg-routing tests.

    Bypasses the real ``__init__`` (which requires DLC + a real project
    on disk). Overrides the properties / methods that ``train_iteration``
    reads so the refine branch can run end-to-end without touching the
    filesystem or DLC.
    """

    def __init__(self, *, latest=3, latest_trained=False, trained_iters=(0, 1, 2)):
        # Deliberately do NOT call super().__init__ -- it needs DLC and
        # a real config.yaml.
        self._latest = latest
        self._latest_trained = latest_trained
        self._trained_iters = set(trained_iters)
        self._current = latest

        # Capture initialize_weights / _initialize_weights_from_external_path
        # / train / edit_config calls -- the routing under test.
        self.initialize_weights = MagicMock(name="initialize_weights")
        self._initialize_weights_from_external_path = MagicMock(
            name="_initialize_weights_from_external_path",
        )
        self.train = MagicMock(name="train")
        self.edit_config = MagicMock(name="edit_config")
        # Heavy ops we don't care about -- no-op them.
        self.extract_frames = MagicMock(name="extract_frames")
        self.increment_iteration = MagicMock(name="increment_iteration")
        self.create_training_dataset = MagicMock(name="create_training_dataset")
        self.evaluate = MagicMock(name="evaluate", return_value=self)
        self.analyze_videos = MagicMock(name="analyze_videos", return_value=self)
        # get_pose_cfg_file used by _initialize_weights_from_external_path
        # (when we test that helper directly).
        self.get_pose_cfg_file = MagicMock(
            name="get_pose_cfg_file",
            return_value=Path("/_fake_pose_cfg.yaml"),
        )

    @property
    def paths(self):
        # Real Path so the ``paths['training_data'] / f'iteration-{N}'``
        # division operator works. The path doesn't have to exist --
        # os.path.exists returning False triggers create_training_dataset
        # (which is mocked).
        return {
            "training_data": Path("/_nonexistent_training_data_for_tests_"),
            "models": Path("/_nonexistent_models_for_tests_"),
        }

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
        # Force the training branch in train_iteration; train() is mocked.
        return False

    def iteration_is_trained(self, iteration_num):
        return iteration_num in self._trained_iters

    @property
    def all_iterations(self):
        return sorted(list(range(self._latest + 1)))

    @property
    def all_snapshots(self):
        return {
            i: ([100] if i in self._trained_iters else [])
            for i in range(self._latest + 1)
        }


@pytest.fixture
def dlc3(monkeypatch):
    """Pin the module-level ``DLC3`` flag to True for the duration of a test."""
    monkeypatch.setattr(dlcinterface, "DLC3", True)
    return True


@pytest.fixture
def dlc2(monkeypatch):
    """Pin the module-level ``DLC3`` flag to False for the duration of a test."""
    monkeypatch.setattr(dlcinterface, "DLC3", False)
    return False


# ---------------------------------------------------------------------------
# Validation: refine_mode discriminator + per-mode required / forbidden args
# ---------------------------------------------------------------------------


class TestValidationInvalidMode:
    def test_invalid_refine_mode_raises(self):
        p = _StubDLCProject()
        with pytest.raises(ValueError, match="refine_mode must be one of"):
            p.train_iteration(refine_mode="bogus")


class TestValidationScratch:
    @pytest.mark.parametrize(
        "kwarg,value",
        [
            ("source_iteration", 1),
            ("source_snapshot", 100),
            ("external_snapshot_path", "/tmp/x.pt"),
        ],
    )
    def test_scratch_rejects_source_args(self, kwarg, value):
        p = _StubDLCProject()
        with pytest.raises(ValueError, match="refine_mode='scratch'"):
            p.train_iteration(refine_mode="scratch", **{kwarg: value})


class TestValidationInProject:
    def test_in_project_missing_source_iteration_raises(self):
        p = _StubDLCProject()
        with pytest.raises(ValueError, match="requires source_iteration"):
            p.train_iteration(refine_mode="in_project")

    def test_in_project_non_int_source_iteration_raises(self):
        p = _StubDLCProject()
        with pytest.raises(TypeError, match="source_iteration must be int"):
            p.train_iteration(refine_mode="in_project", source_iteration="2")

    def test_in_project_untrained_source_iteration_raises(self):
        p = _StubDLCProject(trained_iters=(0, 1, 2))
        with pytest.raises(ValueError, match="not a trained iteration"):
            p.train_iteration(refine_mode="in_project", source_iteration=5)

    def test_in_project_with_external_path_raises(self):
        p = _StubDLCProject()
        with pytest.raises(ValueError, match="incompatible with external_snapshot_path"):
            p.train_iteration(
                refine_mode="in_project",
                source_iteration=1,
                external_snapshot_path="/tmp/x.pt",
            )

    def test_in_project_non_int_source_snapshot_raises(self):
        p = _StubDLCProject()
        with pytest.raises(TypeError, match="source_snapshot must be int"):
            p.train_iteration(
                refine_mode="in_project",
                source_iteration=1,
                source_snapshot="100",
            )


class TestValidationExternal:
    def test_external_missing_path_raises(self):
        p = _StubDLCProject()
        with pytest.raises(ValueError, match="requires external_snapshot_path"):
            p.train_iteration(refine_mode="external")

    def test_external_nonexistent_path_raises(self):
        p = _StubDLCProject()
        with pytest.raises(FileNotFoundError, match="External snapshot not found"):
            p.train_iteration(
                refine_mode="external",
                external_snapshot_path="/_definitely_does_not_exist_abc123.pt",
            )

    @pytest.mark.parametrize(
        "kwarg,value",
        [
            ("source_iteration", 1),
            ("source_snapshot", 100),
        ],
    )
    def test_external_rejects_in_project_args(self, kwarg, value, tmp_path):
        p = _StubDLCProject()
        snapshot = tmp_path / "snapshot.pt"
        snapshot.write_bytes(b"x")
        with pytest.raises(ValueError, match="refine_mode='external'"):
            p.train_iteration(
                refine_mode="external",
                external_snapshot_path=str(snapshot),
                **{kwarg: value},
            )


# ---------------------------------------------------------------------------
# Routing: which init / train call fires under each mode + DLC version
# ---------------------------------------------------------------------------


class TestRoutingScratch:
    def test_dlc3_scratch_no_init_weights_train_with_epochs(self, dlc3):
        p = _StubDLCProject()
        p.train_iteration(refine_mode="scratch")
        p.initialize_weights.assert_not_called()
        p._initialize_weights_from_external_path.assert_not_called()
        p.train.assert_called_once_with(epochs=50)

    def test_dlc2_scratch_no_init_weights_train_with_maxiters(self, dlc2):
        p = _StubDLCProject()
        p.train_iteration(refine_mode="scratch")
        p.initialize_weights.assert_not_called()
        p._initialize_weights_from_external_path.assert_not_called()
        p.train.assert_called_once_with(maxiters=500000)


class TestRoutingInProject:
    def test_dlc3_in_project_calls_initialize_weights(self, dlc3):
        p = _StubDLCProject(trained_iters=(0, 1, 2))
        p.train_iteration(
            refine_mode="in_project",
            source_iteration=2,
            source_snapshot=200,
        )
        p.initialize_weights.assert_called_once_with(
            source_iteration=2, source_snapshot=200
        )
        p.train.assert_called_once_with(epochs=50)
        # snapshot_path must NOT be passed -- the pose_cfg edit handles it.
        assert "snapshot_path" not in p.train.call_args.kwargs

    def test_dlc2_in_project_calls_initialize_weights(self, dlc2):
        p = _StubDLCProject(trained_iters=(0, 1, 2))
        p.train_iteration(
            refine_mode="in_project",
            source_iteration=1,
            source_snapshot=150,
        )
        p.initialize_weights.assert_called_once_with(
            source_iteration=1, source_snapshot=150
        )
        p.train.assert_called_once_with(maxiters=500000)

    def test_in_project_snapshot_none_passes_none(self, dlc3):
        p = _StubDLCProject(trained_iters=(0, 1, 2))
        p.train_iteration(refine_mode="in_project", source_iteration=2)
        p.initialize_weights.assert_called_once_with(
            source_iteration=2, source_snapshot=None
        )


class TestRoutingExternal:
    def test_dlc3_external_passes_snapshot_path_to_train(self, dlc3, tmp_path):
        snapshot = tmp_path / "external.pt"
        snapshot.write_bytes(b"x")
        p = _StubDLCProject()
        p.train_iteration(
            refine_mode="external",
            external_snapshot_path=str(snapshot),
        )
        # No pose_cfg edits on DLC3 external -- snapshot_path is the
        # runtime override.
        p.initialize_weights.assert_not_called()
        p._initialize_weights_from_external_path.assert_not_called()
        p.train.assert_called_once_with(epochs=50, snapshot_path=str(snapshot))

    def test_dlc2_external_edits_pose_cfg_then_trains(self, dlc2, tmp_path):
        snapshot = tmp_path / "external.index"
        snapshot.write_bytes(b"x")
        p = _StubDLCProject()
        p.train_iteration(
            refine_mode="external",
            external_snapshot_path=str(snapshot),
        )
        # DLC2: pose_cfg edit happens before train; train() takes no
        # snapshot_path on DLC2.
        p.initialize_weights.assert_not_called()
        p._initialize_weights_from_external_path.assert_called_once_with(
            str(snapshot)
        )
        p.train.assert_called_once_with(maxiters=500000)
        assert "snapshot_path" not in p.train.call_args.kwargs


# ---------------------------------------------------------------------------
# maxiters defaults + override pass-through
# ---------------------------------------------------------------------------


class TestMaxitersDefaults:
    def test_dlc3_default_is_50_epochs(self, dlc3):
        p = _StubDLCProject()
        p.train_iteration(refine_mode="scratch")
        p.train.assert_called_once_with(epochs=50)

    def test_dlc2_default_is_500000_iters(self, dlc2):
        p = _StubDLCProject()
        p.train_iteration(refine_mode="scratch")
        p.train.assert_called_once_with(maxiters=500000)

    def test_dlc3_explicit_maxiters_passes_through(self, dlc3):
        p = _StubDLCProject()
        p.train_iteration(refine_mode="scratch", maxiters=200)
        p.train.assert_called_once_with(epochs=200)

    def test_dlc2_explicit_maxiters_passes_through(self, dlc2):
        p = _StubDLCProject()
        p.train_iteration(refine_mode="scratch", maxiters=100000)
        p.train.assert_called_once_with(maxiters=100000)


# ---------------------------------------------------------------------------
# analyze_videos forwarding (videos, batchsize, create_video)
# ---------------------------------------------------------------------------


class TestAnalyzeForwarding:
    def test_default_create_video_is_false(self, dlc3):
        p = _StubDLCProject()
        p.train_iteration(refine_mode="scratch")
        p.analyze_videos.assert_called_once_with(create_video=False)

    def test_create_video_true_passes_through(self, dlc3):
        p = _StubDLCProject()
        p.train_iteration(refine_mode="scratch", create_video=True)
        p.analyze_videos.assert_called_once_with(create_video=True)

    def test_videos_none_not_forwarded(self, dlc3):
        p = _StubDLCProject()
        p.train_iteration(refine_mode="scratch")
        # videos kwarg must be absent from analyze_videos call.
        assert "videos" not in p.analyze_videos.call_args.kwargs

    def test_videos_list_forwarded(self, dlc3):
        p = _StubDLCProject()
        p.train_iteration(refine_mode="scratch", videos=[0, 1])
        p.analyze_videos.assert_called_once_with(create_video=False, videos=[0, 1])

    def test_analyze_batchsize_none_not_forwarded(self, dlc3):
        p = _StubDLCProject()
        p.train_iteration(refine_mode="scratch")
        assert "batchsize" not in p.analyze_videos.call_args.kwargs

    def test_analyze_batchsize_forwarded_as_batchsize(self, dlc3):
        p = _StubDLCProject()
        p.train_iteration(refine_mode="scratch", analyze_batchsize=4)
        p.analyze_videos.assert_called_once_with(create_video=False, batchsize=4)


# ---------------------------------------------------------------------------
# _initialize_weights_from_external_path direct tests
# ---------------------------------------------------------------------------


class TestInitializeWeightsFromExternalPath:
    def test_dlc3_edits_resume_training_from(self, dlc3):
        # Don't mock the helper itself this time -- call it for real.
        p = _StubDLCProject()
        # Restore the real method (the fixture stubs it as a MagicMock).
        p._initialize_weights_from_external_path = (
            DLCProject._initialize_weights_from_external_path.__get__(p)
        )
        p._initialize_weights_from_external_path("/path/to/snapshot.pt")
        p.edit_config.assert_called_once()
        args, kwargs = p.edit_config.call_args
        assert kwargs.get("resume_training_from") == "/path/to/snapshot"  # .pt stripped
        assert "init_weights" not in kwargs

    def test_dlc2_edits_init_weights(self, dlc2):
        p = _StubDLCProject()
        p._initialize_weights_from_external_path = (
            DLCProject._initialize_weights_from_external_path.__get__(p)
        )
        p._initialize_weights_from_external_path("/path/to/snapshot.index")
        p.edit_config.assert_called_once()
        args, kwargs = p.edit_config.call_args
        assert kwargs.get("init_weights") == "/path/to/snapshot"  # .index stripped
        assert "resume_training_from" not in kwargs

    def test_no_extension_passed_through(self, dlc3):
        p = _StubDLCProject()
        p._initialize_weights_from_external_path = (
            DLCProject._initialize_weights_from_external_path.__get__(p)
        )
        p._initialize_weights_from_external_path("/path/to/snapshot")
        args, kwargs = p.edit_config.call_args
        assert kwargs.get("resume_training_from") == "/path/to/snapshot"

    def test_path_object_accepted(self, dlc3):
        p = _StubDLCProject()
        p._initialize_weights_from_external_path = (
            DLCProject._initialize_weights_from_external_path.__get__(p)
        )
        p._initialize_weights_from_external_path(Path("/path/snap.pt"))
        args, kwargs = p.edit_config.call_args
        # Path conversion + .pt stripping. Use Path-as-str for cross-platform.
        assert kwargs.get("resume_training_from").endswith("snap")
        assert "snap.pt" not in kwargs.get("resume_training_from")
