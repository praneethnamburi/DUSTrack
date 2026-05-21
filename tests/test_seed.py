"""Tests for ``dustrack.seed.extract_snapshot_for_seeding``.

Uses synthetic byte blobs in place of real .pt / YAML files so the
test runs without DLC installed and without touching a real project.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from dustrack import (
    extract_snapshot_for_seeding,
    import_seed_bundle_into_project,
    inspect_seed_bundle,
)
from dustrack.dlcinterface import _dlc_bodyparts_to_layer_labels
from dustrack.seed import extract_snapshot_for_seeding as _extract_direct


def _make_fake_train_dir(tmp_path: Path) -> Path:
    """Construct a fake ``dlc-models-pytorch/iteration-0/.../train/``
    folder (and sibling ``test/`` with a pose_cfg.yaml)."""
    modelfolder = (
        tmp_path
        / "fake_project-x-2026-01-01"
        / "dlc-models-pytorch"
        / "iteration-0"
        / "fake_projectJan01-trainset95shuffle1"
    )
    train_dir = modelfolder / "train"
    test_dir = modelfolder / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    (train_dir / "snapshot-best-270.pt").write_bytes(b"\x80\x02fake-torch-pickle")
    (train_dir / "snapshot-250.pt").write_bytes(b"\x80\x02fake-older-snapshot")
    (train_dir / "pytorch_config.yaml").write_text(
        textwrap.dedent(
            """
            metadata:
              bodyparts:
              - point0
              - point1
            net_type: resnet_50
            """
        ).strip()
        + "\n"
    )
    (train_dir / "learning_stats.csv").write_text("epoch,loss\n1,0.5\n")
    (test_dir / "pose_cfg.yaml").write_text(
        textwrap.dedent(
            """
            dataset_type: multi-animal-imgaug
            num_joints: 2
            all_joints_names:
            - point0
            - point1
            net_type: resnet_50
            """
        ).strip()
        + "\n"
    )
    return train_dir


def test_extract_copies_snapshot_and_config(tmp_path):
    train_dir = _make_fake_train_dir(tmp_path)
    snapshot = train_dir / "snapshot-best-270.pt"
    dest = tmp_path / "bundle"

    returned = extract_snapshot_for_seeding(snapshot, dest)

    assert returned == dest
    assert (dest / "snapshot-best-270.pt").read_bytes() == snapshot.read_bytes()
    assert (dest / "pytorch_config.yaml").read_text() == (
        train_dir / "pytorch_config.yaml"
    ).read_text()
    assert (dest / "pose_cfg.yaml").read_text() == (
        train_dir.parent / "test" / "pose_cfg.yaml"
    ).read_text()


def test_extract_omits_training_provenance(tmp_path):
    """Only the .pt, pytorch_config.yaml, and pose_cfg.yaml are
    copied; other train/ files (learning_stats.csv, train.txt,
    sibling snapshots) are not."""
    train_dir = _make_fake_train_dir(tmp_path)
    snapshot = train_dir / "snapshot-best-270.pt"
    dest = tmp_path / "bundle"

    extract_snapshot_for_seeding(snapshot, dest)

    assert {p.name for p in dest.iterdir()} == {
        "snapshot-best-270.pt",
        "pytorch_config.yaml",
        "pose_cfg.yaml",
    }


def test_extract_creates_destination_dir(tmp_path):
    train_dir = _make_fake_train_dir(tmp_path)
    snapshot = train_dir / "snapshot-best-270.pt"
    dest = tmp_path / "nested" / "does" / "not" / "exist" / "yet"
    assert not dest.exists()

    extract_snapshot_for_seeding(snapshot, dest)

    assert dest.is_dir()
    assert (dest / "snapshot-best-270.pt").is_file()


def test_extract_overwrites_matching_files(tmp_path):
    train_dir = _make_fake_train_dir(tmp_path)
    snapshot = train_dir / "snapshot-best-270.pt"
    dest = tmp_path / "bundle"
    dest.mkdir()
    (dest / "snapshot-best-270.pt").write_bytes(b"stale")
    (dest / "unrelated.txt").write_text("keep me")

    extract_snapshot_for_seeding(snapshot, dest)

    assert (dest / "snapshot-best-270.pt").read_bytes() == snapshot.read_bytes()
    assert (dest / "unrelated.txt").read_text() == "keep me"


def test_extract_rejects_missing_snapshot(tmp_path):
    with pytest.raises(FileNotFoundError, match="must point to an existing .pt"):
        extract_snapshot_for_seeding(tmp_path / "nope.pt", tmp_path / "bundle")


def test_extract_rejects_non_pt_file(tmp_path):
    train_dir = _make_fake_train_dir(tmp_path)
    (train_dir / "snapshot-best-270.txt").write_text("not a checkpoint")

    with pytest.raises(FileNotFoundError, match="must point to an existing .pt"):
        extract_snapshot_for_seeding(
            train_dir / "snapshot-best-270.txt", tmp_path / "bundle"
        )


def test_extract_requires_pytorch_config(tmp_path):
    """A DLC2 project (no pytorch_config.yaml) is rejected with a
    clear message."""
    train_dir = _make_fake_train_dir(tmp_path)
    (train_dir / "pytorch_config.yaml").unlink()
    snapshot = train_dir / "snapshot-best-270.pt"

    with pytest.raises(
        FileNotFoundError, match="No pytorch_config.yaml alongside the snapshot"
    ):
        extract_snapshot_for_seeding(snapshot, tmp_path / "bundle")


def test_extract_requires_pose_cfg(tmp_path):
    """Missing test/pose_cfg.yaml is rejected with a clear message
    (DLC's analyze_videos hard-errors without it)."""
    train_dir = _make_fake_train_dir(tmp_path)
    (train_dir.parent / "test" / "pose_cfg.yaml").unlink()
    snapshot = train_dir / "snapshot-best-270.pt"

    with pytest.raises(
        FileNotFoundError, match="No test/pose_cfg.yaml"
    ):
        extract_snapshot_for_seeding(snapshot, tmp_path / "bundle")


def test_top_level_export_matches_module_export():
    assert extract_snapshot_for_seeding is _extract_direct


# ---------------------------------------------------------------------------
# _dlc_bodyparts_to_layer_labels
# ---------------------------------------------------------------------------


def test_bodyparts_to_labels_sequential_point_prefix():
    assert _dlc_bodyparts_to_layer_labels(["point0", "point1"]) == ["0", "1"]


def test_bodyparts_to_labels_sparse_point_prefix():
    """User-cited case: a project trained with non-sequential
    bodyparts should preserve the digit identity, not renumber from 0."""
    assert _dlc_bodyparts_to_layer_labels(["point1", "point3"]) == ["1", "3"]


def test_bodyparts_to_labels_non_digit_falls_back_to_indices():
    """Bodyparts from a foreign-project DLC config (e.g. SuperAnimal
    body-part names) should renumber to consecutive indices."""
    assert _dlc_bodyparts_to_layer_labels(["nose", "ear"]) == ["0", "1"]


def test_bodyparts_to_labels_mixed_falls_back_to_indices():
    """Any non-digit bodypart in the set forces the index fallback."""
    assert _dlc_bodyparts_to_layer_labels(["point0", "ear"]) == ["0", "1"]


def test_bodyparts_to_labels_empty():
    assert _dlc_bodyparts_to_layer_labels([]) == []


# ---------------------------------------------------------------------------
# inspect_seed_bundle
# ---------------------------------------------------------------------------


def _make_real_bundle(tmp_path: Path) -> Path:
    """Build a bundle by going through extract_snapshot_for_seeding,
    so the layout exactly matches what the extractor produces."""
    train_dir = _make_fake_train_dir(tmp_path)
    bundle = tmp_path / "bundle"
    extract_snapshot_for_seeding(train_dir / "snapshot-best-270.pt", bundle)
    return bundle


def test_inspect_returns_bodyparts_and_paths(tmp_path):
    bundle = _make_real_bundle(tmp_path)
    info = inspect_seed_bundle(bundle)
    assert info["snapshot"].name == "snapshot-best-270.pt"
    assert info["pytorch_config"].name == "pytorch_config.yaml"
    assert info["pose_cfg"].name == "pose_cfg.yaml"
    assert info["bodyparts"] == ["point0", "point1"]
    assert info["net_type"] == "resnet_50"


def test_inspect_rejects_non_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a directory"):
        inspect_seed_bundle(tmp_path / "nope")


def test_inspect_rejects_missing_snapshot(tmp_path):
    bundle = _make_real_bundle(tmp_path)
    (bundle / "snapshot-best-270.pt").unlink()
    with pytest.raises(FileNotFoundError, match="No snapshot-\\*\\.pt"):
        inspect_seed_bundle(bundle)


def test_inspect_rejects_multiple_snapshots(tmp_path):
    bundle = _make_real_bundle(tmp_path)
    (bundle / "snapshot-100.pt").write_bytes(b"second")
    with pytest.raises(ValueError, match="2 snapshot-\\*\\.pt"):
        inspect_seed_bundle(bundle)


def test_inspect_rejects_missing_pytorch_config(tmp_path):
    bundle = _make_real_bundle(tmp_path)
    (bundle / "pytorch_config.yaml").unlink()
    with pytest.raises(FileNotFoundError, match="Missing pytorch_config.yaml"):
        inspect_seed_bundle(bundle)


def test_inspect_rejects_missing_pose_cfg(tmp_path):
    bundle = _make_real_bundle(tmp_path)
    (bundle / "pose_cfg.yaml").unlink()
    with pytest.raises(FileNotFoundError, match="Missing pose_cfg.yaml"):
        inspect_seed_bundle(bundle)


def test_inspect_rejects_pytorch_config_without_bodyparts(tmp_path):
    bundle = _make_real_bundle(tmp_path)
    (bundle / "pytorch_config.yaml").write_text("net_type: resnet_50\n")
    with pytest.raises(ValueError, match="no metadata.bodyparts"):
        inspect_seed_bundle(bundle)


# ---------------------------------------------------------------------------
# import_seed_bundle_into_project
# ---------------------------------------------------------------------------


class _FakeDLCProject:
    """Minimal stand-in for ``DLCProject`` -- exposes only the surface
    ``import_seed_bundle_into_project`` reads / writes.

    Mirrors the real class's contract:
      - ``config_path`` is a path to ``<project>/config.yaml``
      - ``config`` reads + parses ``config.yaml``
      - ``edit_config(**kwargs)`` merges kwargs into ``config.yaml``

    Avoids importing ``deeplabcut`` so the test runs in the light
    environment.
    """

    def __init__(self, project_root: Path, task: str, date: str,
                 training_fraction: float = 0.95):
        project_root.mkdir(parents=True, exist_ok=True)
        self.config_path = project_root / "config.yaml"
        self.config_path.write_text(yaml.safe_dump({
            "Task": task,
            "date": date,
            "TrainingFraction": [training_fraction],
            "bodyparts": [],
            "iteration": 0,
        }, sort_keys=False))

    @property
    def config(self):
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def edit_config(self, **kwargs):
        data = self.config
        data.update(kwargs)
        with open(self.config_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)


def test_import_creates_modelfolder_layout(tmp_path):
    bundle = _make_real_bundle(tmp_path)
    project = _FakeDLCProject(tmp_path / "new_proj-x-2026-05-21",
                              task="new_proj", date="May21")

    modelfolder = import_seed_bundle_into_project(project, bundle)

    expected = (
        project.config_path.parent
        / "dlc-models-pytorch"
        / "iteration-0"
        / "new_projMay21-trainset95shuffle1"
    )
    assert modelfolder == expected
    assert (modelfolder / "train" / "snapshot-best-270.pt").is_file()
    assert (modelfolder / "train" / "pytorch_config.yaml").is_file()
    assert (modelfolder / "test" / "pose_cfg.yaml").is_file()


def test_import_overwrites_project_bodyparts(tmp_path):
    bundle = _make_real_bundle(tmp_path)
    project = _FakeDLCProject(tmp_path / "new_proj-x-2026-05-21",
                              task="new_proj", date="May21")
    assert project.config["bodyparts"] == []

    import_seed_bundle_into_project(project, bundle)

    assert project.config["bodyparts"] == ["point0", "point1"]


def test_import_rewrites_pytorch_config_paths(tmp_path):
    bundle = _make_real_bundle(tmp_path)
    project_root = tmp_path / "new_proj-x-2026-05-21"
    project = _FakeDLCProject(project_root, task="new_proj", date="May21")

    modelfolder = import_seed_bundle_into_project(project, bundle)

    with open(modelfolder / "train" / "pytorch_config.yaml") as f:
        rewritten = yaml.safe_load(f)
    assert rewritten["metadata"]["project_path"] == str(project_root)
    assert rewritten["metadata"]["pose_config_path"] == str(
        modelfolder / "train" / "pytorch_config.yaml"
    )
    # Non-path fields preserved.
    assert rewritten["metadata"]["bodyparts"] == ["point0", "point1"]
    assert rewritten["net_type"] == "resnet_50"


def test_import_rewrites_pose_cfg_dataset(tmp_path):
    bundle = _make_real_bundle(tmp_path)
    project_root = tmp_path / "new_proj-x-2026-05-21"
    project = _FakeDLCProject(project_root, task="new_proj", date="May21")

    modelfolder = import_seed_bundle_into_project(project, bundle)

    with open(modelfolder / "test" / "pose_cfg.yaml") as f:
        rewritten = yaml.safe_load(f)
    assert rewritten["dataset"] == str(project_root)
    # Other fields preserved verbatim.
    assert rewritten["num_joints"] == 2
    assert rewritten["net_type"] == "resnet_50"


def test_import_modelfolder_name_uses_training_fraction(tmp_path):
    bundle = _make_real_bundle(tmp_path)
    project = _FakeDLCProject(
        tmp_path / "low_frac_proj-x-2026-05-21",
        task="low_frac_proj", date="May21",
        training_fraction=0.7,
    )

    modelfolder = import_seed_bundle_into_project(project, bundle)

    assert modelfolder.name == "low_frac_projMay21-trainset70shuffle1"


def test_import_respects_iteration_and_shuffle_overrides(tmp_path):
    bundle = _make_real_bundle(tmp_path)
    project = _FakeDLCProject(tmp_path / "p-x-2026-05-21",
                              task="p", date="May21")

    modelfolder = import_seed_bundle_into_project(
        project, bundle, iteration=2, shuffle=3,
    )

    assert "iteration-2" in modelfolder.parts
    assert modelfolder.name == "pMay21-trainset95shuffle3"


def test_import_writes_training_dataset_metadata(tmp_path):
    """metadata.yaml in training-datasets/iteration-N/.../ is the
    DLC shuffle registry that ``analyze_videos`` consults. The
    smoke test pins the end-to-end behavior; this unit pins the
    file format so a future format drift is caught without needing
    the slow path."""
    bundle = _make_real_bundle(tmp_path)
    project = _FakeDLCProject(tmp_path / "p-x-2026-05-21",
                              task="p", date="May21")

    import_seed_bundle_into_project(project, bundle)

    metadata_path = (
        Path(project.config_path).parent
        / "training-datasets"
        / "iteration-0"
        / "UnaugmentedDataSet_pMay21"
        / "metadata.yaml"
    )
    assert metadata_path.is_file()
    metadata = yaml.safe_load(metadata_path.read_text())
    shuffles = metadata["shuffles"]
    assert "pMay21-trainset95shuffle1" in shuffles
    entry = shuffles["pMay21-trainset95shuffle1"]
    assert entry["train_fraction"] == 0.95
    assert entry["index"] == 1
    assert entry["engine"] == "pytorch"


def test_import_propagates_invalid_bundle_errors(tmp_path):
    bundle = _make_real_bundle(tmp_path)
    (bundle / "snapshot-best-270.pt").unlink()
    project = _FakeDLCProject(tmp_path / "p-x-2026-05-21",
                              task="p", date="May21")

    with pytest.raises(FileNotFoundError, match="No snapshot"):
        import_seed_bundle_into_project(project, bundle)


# ---------------------------------------------------------------------------
# End-to-end smoke (slow): real DLC project + real seed bundle + real video.
# ---------------------------------------------------------------------------


_REAL_SOURCE_VIDEO = Path("S:/_corpus/dustrack/pia02_s001_011_RFA2_min1_15s_mono.mp4")
_REAL_SEED_BUNDLE = Path(
    "M:/DLC_MODELS/seed_bundles/interosseous_pn24Oct24_iter0_snap-best-270"
)


@pytest.mark.slow
def test_seed_bundle_drives_iteration_0_inference(tmp_path):
    """Empirically verify the seeding contract: building a DLC
    project + ``import_seed_bundle_into_project`` + ``analyze_videos
    (iteration_num=0)`` produces predictions without any training.

    Bypasses ``DUSTrack`` (which needs a Qt-backed figure to
    construct -- datanavigator's QtScatterArtist crashes under
    Agg) and exercises the import + inference path directly, which
    is the part the modal wire-up exists to invoke.

    Settles the shuffle/folder-name question empirically: if DLC's
    default-shuffle inference doesn't match the ``shuffle=1``
    folder we manufacture, the predictions HDF5 won't appear and
    the test fails clearly.

    Skipped automatically when the test corpus isn't reachable.
    """
    if not _REAL_SOURCE_VIDEO.is_file():
        pytest.skip(f"Source video not available at {_REAL_SOURCE_VIDEO}")
    if not _REAL_SEED_BUNDLE.is_dir():
        pytest.skip(f"Seed bundle not available at {_REAL_SEED_BUNDLE}")

    import json
    import shutil as _shutil

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    video_path = workdir / _REAL_SOURCE_VIDEO.name
    _shutil.copy2(_REAL_SOURCE_VIDEO, video_path)

    # ``DLCProject.__init__`` calls ``copy_annotations`` which looks
    # for ``<video_stem>_annotations_<suffix>.json`` next to the
    # video. In the real seeding flow, ``DUSTrack.ann.save()`` writes
    # an empty ``{}`` JSON before constructing the project. Mirror
    # that here so the constructor's labels-set assert doesn't fail
    # on an empty zip.
    ann_path = (
        video_path.parent
        / f"{video_path.stem}_annotations_manual.json"
    )
    ann_path.write_text(json.dumps({}))

    from dustrack.dlcinterface import DLCProject

    project = DLCProject(
        path=str(workdir),
        videos=[str(video_path)],
        name="seedtest_iter0",
        experimenter="x",
        annotation_suffix="manual",
    )

    import_seed_bundle_into_project(project, _REAL_SEED_BUNDLE)

    assert project.config["bodyparts"] == ["point0", "point1"], (
        "Bundle's bodyparts should overwrite the empty-derived default"
    )
    assert project.latest_iteration_is_trained(), (
        "Iteration-0 should look trained after bundle import "
        "(snapshot is now present in dlc-models-pytorch/iteration-0/)"
    )

    project.analyze_videos(iteration_num=0, create_video=False)

    project_root = Path(project.config_path).parent
    predictions_dir = project_root / "videos" / "iteration-0"
    assert predictions_dir.is_dir(), (
        f"analyze_videos should have created {predictions_dir}"
    )
    h5_files = list(predictions_dir.glob("*.h5"))
    assert h5_files, (
        f"No prediction HDF5 in {predictions_dir}; analyze_videos "
        "appears to have failed or written elsewhere."
    )
