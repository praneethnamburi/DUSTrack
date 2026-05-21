"""Tests for ``dustrack.seed.extract_snapshot_for_seeding``.

Uses synthetic byte blobs in place of real .pt / YAML files so the
test runs without DLC installed and without touching a real project.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from dustrack import extract_snapshot_for_seeding
from dustrack.seed import extract_snapshot_for_seeding as _extract_direct


def _make_fake_train_dir(tmp_path: Path) -> Path:
    """Construct a fake ``dlc-models-pytorch/iteration-0/.../train/`` folder."""
    train_dir = (
        tmp_path
        / "fake_project-x-2026-01-01"
        / "dlc-models-pytorch"
        / "iteration-0"
        / "fake_projectJan01-trainset95shuffle1"
        / "train"
    )
    train_dir.mkdir(parents=True)
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


def test_extract_omits_training_provenance(tmp_path):
    """Only the .pt and pytorch_config.yaml are copied; other train/
    files (learning_stats.csv, train.txt, sibling snapshots) are not."""
    train_dir = _make_fake_train_dir(tmp_path)
    snapshot = train_dir / "snapshot-best-270.pt"
    dest = tmp_path / "bundle"

    extract_snapshot_for_seeding(snapshot, dest)

    assert {p.name for p in dest.iterdir()} == {
        "snapshot-best-270.pt",
        "pytorch_config.yaml",
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


def test_top_level_export_matches_module_export():
    assert extract_snapshot_for_seeding is _extract_direct
