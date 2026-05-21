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

import json
import shutil
from pathlib import Path
from typing import Optional, Union

import yaml


# Per-user persistence for the "seed bundles root" the picker modal
# remembers between sessions. Path stays out of the project tree so
# every dustrack session on this machine shares it.
_USER_CONFIG_DIR = Path.home() / ".dustrack"
_USER_CONFIG_PATH = _USER_CONFIG_DIR / "config.json"


def _read_user_config() -> dict:
    if not _USER_CONFIG_PATH.is_file():
        return {}
    try:
        return json.loads(_USER_CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _write_user_config(cfg: dict) -> None:
    _USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _USER_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def get_seed_bundles_root() -> Optional[Path]:
    """Read the remembered seed-bundles root from
    ``~/.dustrack/config.json``. Returns ``None`` if unset or the
    path no longer exists on disk."""
    root = _read_user_config().get("seed_bundles_root")
    if not root:
        return None
    p = Path(root)
    return p if p.is_dir() else None


def set_seed_bundles_root(path: Union[str, Path, None]) -> None:
    """Persist the seed-bundles root path so the Create-DLC-Project
    seeding modal can populate its bundle list without forcing the
    user to navigate from scratch each time. Pass ``None`` to forget
    the current value."""
    cfg = _read_user_config()
    if path is None:
        cfg.pop("seed_bundles_root", None)
    else:
        cfg["seed_bundles_root"] = str(Path(path).resolve())
    _write_user_config(cfg)


def list_seed_bundles(root: Union[str, Path]) -> list[dict]:
    """Scan ``root`` for valid seed bundles and return their metadata.

    Each subdirectory that passes :func:`inspect_seed_bundle` is
    included; invalid folders are silently skipped (the picker modal
    is allowed to show "no bundles found" rather than surfacing a
    long list of validation errors).

    Returns:
        list[dict]: One entry per valid bundle, sorted by folder name.
        Each entry carries every field :func:`inspect_seed_bundle`
        returns plus ``name`` (the bundle folder's name relative to
        ``root``) and ``path`` (the absolute bundle folder path).
    """
    root = Path(root)
    if not root.is_dir():
        return []
    bundles = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        try:
            info = inspect_seed_bundle(sub)
        except (FileNotFoundError, ValueError):
            continue
        bundles.append({"name": sub.name, "path": sub, **info})
    return bundles


def extract_snapshot_for_seeding(
    snapshot_path: Union[str, Path],
    destination_path: Union[str, Path],
    description: str = "",
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
            ``pose_cfg.yaml`` / ``description.txt`` are overwritten.
        description: Optional human-readable description of the
            bundle (one paragraph; what the model tracks, training
            corpus, intended use). Written to ``description.txt`` in
            the bundle and surfaced by the seeding-modal picker.
            Empty string skips the file.

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
    if description:
        (destination_path / "description.txt").write_text(
            description.strip() + "\n"
        )

    return destination_path


def inspect_seed_bundle(bundle_path: Union[str, Path]) -> dict:
    """Validate a seed bundle and return its metadata.

    Args:
        bundle_path: Folder containing ``snapshot-*.pt``,
            ``pytorch_config.yaml``, and ``pose_cfg.yaml``.

    Returns:
        dict: ``{"snapshot": Path, "pytorch_config": Path,
        "pose_cfg": Path, "bodyparts": list[str], "net_type": str}``.

    Raises:
        FileNotFoundError: If any of the three required files is
            missing.
        ValueError: If multiple ``snapshot-*.pt`` files are present
            (bundle is ambiguous), or if ``pytorch_config.yaml``
            lacks ``metadata.bodyparts``.
    """
    bundle_path = Path(bundle_path)
    if not bundle_path.is_dir():
        raise FileNotFoundError(f"Bundle path is not a directory: {bundle_path}")

    pt_files = sorted(bundle_path.glob("snapshot-*.pt"))
    if not pt_files:
        raise FileNotFoundError(
            f"No snapshot-*.pt in bundle: {bundle_path}"
        )
    if len(pt_files) > 1:
        raise ValueError(
            f"Bundle has {len(pt_files)} snapshot-*.pt files; expected exactly one: "
            f"{[p.name for p in pt_files]}"
        )

    pytorch_config = bundle_path / "pytorch_config.yaml"
    pose_cfg = bundle_path / "pose_cfg.yaml"
    if not pytorch_config.is_file():
        raise FileNotFoundError(f"Missing pytorch_config.yaml in bundle: {bundle_path}")
    if not pose_cfg.is_file():
        raise FileNotFoundError(f"Missing pose_cfg.yaml in bundle: {bundle_path}")

    with open(pytorch_config) as f:
        pytorch_cfg_data = yaml.safe_load(f)
    try:
        bodyparts = pytorch_cfg_data["metadata"]["bodyparts"]
    except (KeyError, TypeError):
        raise ValueError(
            f"pytorch_config.yaml in {bundle_path} has no metadata.bodyparts"
        )
    net_type = pytorch_cfg_data.get("net_type", "")

    description_path = bundle_path / "description.txt"
    description = (
        description_path.read_text().strip()
        if description_path.is_file()
        else ""
    )

    return {
        "snapshot": pt_files[0],
        "pytorch_config": pytorch_config,
        "pose_cfg": pose_cfg,
        "bodyparts": list(bodyparts),
        "net_type": net_type,
        "description": description,
    }


def import_seed_bundle_into_project(
    dlc_project,
    bundle_path: Union[str, Path],
    iteration: int = 0,
    shuffle: int = 1,
) -> Path:
    """Wire a seed bundle into a DLC project so DLC sees iteration-N
    as already trained with the bundled snapshot.

    Side effects:
      - ``dlc_project.edit_config(bodyparts=<from bundle>)`` -- the
        project's bodypart list is *overwritten* by the bundle's
        ``metadata.bodyparts``. The model literally expects those
        output channels, so the project must match.
      - Creates ``<project>/dlc-models-pytorch/iteration-<N>/<modelfolder>/{train,test}/``,
        where ``<modelfolder>`` is the DLC convention
        ``<Task><date>-trainset<int(frac*100)>shuffle<shuffle>``
        (see ``deeplabcut/utils/auxiliaryfunctions.get_model_folder``).
      - Copies the snapshot into ``train/`` and writes
        ``train/pytorch_config.yaml`` with ``metadata.project_path``
        and ``metadata.pose_config_path`` rewritten to the destination.
      - Writes ``test/pose_cfg.yaml`` with its ``dataset`` field
        rewritten to the new project root.
      - Writes ``training-datasets/iteration-<N>/UnaugmentedDataSet_<Task><date>/metadata.yaml``
        with a single shuffle entry matching the manufactured folder.
        DLC's ``analyze_videos`` resolves shuffles through this file
        (``TrainingDatasetMetadata.get``), so without it inference
        raises ``Could not find a shuffle with trainingset fraction
        X and index N``.

    After this returns, ``dlc_project.analyze_videos(iteration_num=N)``
    will pick up the bundled snapshot via ``all_snapshots`` (which
    globs ``*train/snapshot*.pt``) and run inference against the
    new video without any training having happened.

    Args:
        dlc_project: A :class:`dustrack.dlcinterface.DLCProject`
            instance, just constructed (typically with an empty
            active manual layer -- ``bodyparts: []`` in
            ``config.yaml``, no iteration-N folder yet).
        bundle_path: Folder produced by
            :func:`extract_snapshot_for_seeding`.
        iteration: Iteration number to install the bundle under
            (default 0).
        shuffle: Shuffle number for the model folder name (default
            1, matching DLC's ``analyze_videos`` default).

    Returns:
        Path: The created modelfolder (``.../iteration-<N>/<modelfolder>``).

    Raises:
        FileNotFoundError, ValueError: from :func:`inspect_seed_bundle`.
    """
    info = inspect_seed_bundle(bundle_path)

    dlc_project.edit_config(bodyparts=info["bodyparts"])

    cfg = dlc_project.config
    proj_id = f"{cfg['Task']}{cfg['date']}"
    train_frac = cfg["TrainingFraction"][0]
    modelfolder_name = (
        f"{proj_id}-trainset{int(train_frac * 100)}shuffle{shuffle}"
    )

    project_root = Path(dlc_project.config_path).parent
    modelfolder = (
        project_root
        / "dlc-models-pytorch"
        / f"iteration-{iteration}"
        / modelfolder_name
    )
    train_dir = modelfolder / "train"
    test_dir = modelfolder / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot: byte-identical copy, original filename preserved so
    # the epoch number is visible.
    shutil.copy2(info["snapshot"], train_dir / info["snapshot"].name)

    # pytorch_config.yaml: rewrite the two absolute paths inside
    # metadata to the destination project. Other fields (model arch,
    # bodyparts, training hyperparameters) are preserved verbatim --
    # they describe the bundled snapshot's identity, not the project.
    with open(info["pytorch_config"]) as f:
        pytorch_cfg_data = yaml.safe_load(f)
    pytorch_cfg_data.setdefault("metadata", {})
    pytorch_cfg_data["metadata"]["project_path"] = str(project_root)
    pytorch_cfg_data["metadata"]["pose_config_path"] = str(
        train_dir / "pytorch_config.yaml"
    )
    with open(train_dir / "pytorch_config.yaml", "w") as f:
        yaml.safe_dump(pytorch_cfg_data, f, sort_keys=False)

    # pose_cfg.yaml: rewrite the ``dataset`` field (project root).
    # Inference's read_plainconfig at videos.py:425 only consumes a
    # handful of fields (dataset_type, num_joints, all_joints_names,
    # net_type); ``dataset`` is preserved as provenance.
    with open(info["pose_cfg"]) as f:
        pose_cfg_data = yaml.safe_load(f)
    pose_cfg_data["dataset"] = str(project_root)
    with open(test_dir / "pose_cfg.yaml", "w") as f:
        yaml.safe_dump(pose_cfg_data, f, sort_keys=False)

    # training-datasets/iteration-N/UnaugmentedDataSet_<Task><date>/metadata.yaml
    # registers the shuffle so DLC's inference path can resolve it.
    # ``analyze_videos`` calls ``TrainingDatasetMetadata.get(trainset_index,
    # index)`` (``deeplabcut/generate_training_dataset/metadata.py:186``)
    # to look up the shuffle by (train_fraction, index); without an
    # entry it raises ``Could not find a shuffle with trainingset
    # fraction X and index N``. We write a minimal record matching
    # the modelfolder we just produced. ``split`` is a local index
    # into a data_splits map that's only written when DLC owns the
    # save (``metadata.py:225``) and only consumed when
    # ``load_splits=True`` (training time); using ``split: 1`` here
    # is the same harmless value DLC writes for a single-shuffle
    # project.
    trainset_dir = (
        project_root
        / "training-datasets"
        / f"iteration-{iteration}"
        / f"UnaugmentedDataSet_{proj_id}"
    )
    trainset_dir.mkdir(parents=True, exist_ok=True)
    metadata_payload = {
        "shuffles": {
            modelfolder_name: {
                "train_fraction": float(train_frac),
                "index": shuffle,
                "split": 1,
                "engine": "pytorch",
            }
        }
    }
    with open(trainset_dir / "metadata.yaml", "w") as f:
        f.write("# Generated by dustrack.import_seed_bundle_into_project\n")
        f.write("---\n")
        yaml.safe_dump(metadata_payload, f, sort_keys=False)

    return modelfolder
