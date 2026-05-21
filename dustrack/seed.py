"""Seed-bundle utilities for bootstrapping DLC projects from external snapshots.

A seed bundle is a minimal copy of a trained DLC3 iteration's
``<modelfolder>`` -- just enough for another project to use the
snapshot as if it were the result of its own iteration-0 training run:

    <bundle_dir>/
        snapshot-<NNN>.pt        # or snapshot-best-<NNN>.pt
        pytorch_config.yaml      # network architecture + bodyparts
        pose_cfg.yaml            # sourced from <modelfolder>/test/

All three are required at inference time: DLC's pytorch
``analyze_videos`` instantiates a ``DLCLoader`` (which reads
``pytorch_config.yaml``) and then ``read_plainconfig(loader.model_folder.parent
/ 'test' / 'pose_cfg.yaml')`` (``deeplabcut/pose_estimation_pytorch/apis/videos.py:425``).
Without ``pose_cfg.yaml`` the call hard-errors; without
``pytorch_config.yaml`` the model can't be built.

The companion import flow (``Create DLC project`` modal when the
active manual layer is empty) drops these three files into the new
project's ``dlc-models-pytorch/iteration-0/<modelfolder>/train/``
(snapshot + pytorch_config) and ``.../test/`` (pose_cfg), so DLC
sees iteration-0 as already trained, dense predictions become the
reference overlay, and the user's manual refinements become
iteration-1.

Bundled assets are copied verbatim; absolute paths inside
``pytorch_config.yaml`` (``metadata.project_path`` /
``metadata.pose_config_path``) and ``pose_cfg.yaml`` (``dataset``)
are not scrubbed at export time -- the importer is responsible for
rewriting them when wiring the bundle into a destination project.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Union


def extract_snapshot_for_seeding(
    snapshot_path: Union[str, Path],
    destination_path: Union[str, Path],
) -> Path:
    """Copy a DLC3 snapshot + its ``pytorch_config.yaml`` + sibling
    ``test/pose_cfg.yaml`` into a seed bundle.

    Args:
        snapshot_path: Path to a ``snapshot-*.pt`` file inside an
            existing DLC3 project's
            ``dlc-models-pytorch/iteration-N/<modelfolder>/train/`` folder.
        destination_path: Folder to write the bundle into. Created if
            missing; pre-existing unrelated files are preserved, but
            matching ``snapshot-*.pt`` / ``pytorch_config.yaml`` /
            ``pose_cfg.yaml`` are overwritten.

    Returns:
        Path: The destination folder (as a :class:`pathlib.Path`).

    Raises:
        FileNotFoundError: If ``snapshot_path`` does not point at an
            existing ``.pt`` file, or if ``pytorch_config.yaml`` /
            sibling ``test/pose_cfg.yaml`` are missing.
    """
    snapshot_path = Path(snapshot_path)
    destination_path = Path(destination_path)

    if not snapshot_path.is_file() or snapshot_path.suffix != ".pt":
        raise FileNotFoundError(
            f"snapshot_path must point to an existing .pt file: {snapshot_path}"
        )

    train_dir = snapshot_path.parent
    pytorch_config = train_dir / "pytorch_config.yaml"
    if not pytorch_config.is_file():
        raise FileNotFoundError(
            f"No pytorch_config.yaml alongside the snapshot at {train_dir}. "
            "Seed extraction only supports DLC3 (pytorch) projects."
        )

    # ``test/pose_cfg.yaml`` lives one level up from ``train/`` -- DLC
    # reads it at inference time via ``read_plainconfig``
    # (deeplabcut/pose_estimation_pytorch/apis/videos.py:425) so the
    # bundle is unusable without it.
    pose_cfg = train_dir.parent / "test" / "pose_cfg.yaml"
    if not pose_cfg.is_file():
        raise FileNotFoundError(
            f"No test/pose_cfg.yaml at {pose_cfg}. DLC's analyze_videos "
            "reads this file at inference time; the bundle is unusable "
            "without it."
        )

    destination_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(snapshot_path, destination_path / snapshot_path.name)
    shutil.copy2(pytorch_config, destination_path / pytorch_config.name)
    shutil.copy2(pose_cfg, destination_path / pose_cfg.name)

    return destination_path
