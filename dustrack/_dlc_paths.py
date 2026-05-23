"""DLC-project filesystem predicates + multi-video path validation.

Six helpers extracted from ``dlcinterface.py`` in the 1.2.0rc1
follow-up: they're pure filesystem / path predicates that don't
touch the :class:`DLCProject` class internals, and they're shared
between :mod:`._open`, :mod:`._bundle_swap`, :mod:`._bundle`,
:mod:`._workflow_gates`, and :mod:`.gui`.

Keeping them here (vs. in ``dlcinterface.py``) lets consumers
import them without dragging in DLC's ~7 s lazy load (the
DLCProject class triggers ``_ensure_dlc_loaded`` on construction).
:func:`_resolve_multi_video_from_list` is the one function that
needs ``DLCProject`` and lazy-imports it locally.

The names keep their leading underscore prefix so consumers that
previously imported from ``dustrack.dlcinterface`` keep working
via the PEP 562 ``__getattr__`` proxy at the tail of
``dlcinterface.py``.

Extracted from ``dlcinterface.py`` in the 1.2.0rc1 follow-up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def _is_dlc_config_yaml(path) -> bool:
    """True iff ``path`` is a DLC ``config.yaml`` file (case-insensitive
    on the basename; the file's parent must exist but we don't structurally
    validate it as a full project here -- DLCProject construction will
    surface a clearer error if the config is malformed).
    """
    p = Path(path)
    if not p.is_file():
        return False
    return p.name.lower() == "config.yaml"


def _is_dlc_project_root(folder) -> bool:
    """Cheap structural check for a DLC project folder.

    DLC's ``create_new_project`` always lays down ``config.yaml`` next to
    ``videos/`` and ``labeled-data/``; requiring all three avoids matching
    a stray ``config.yaml`` that belongs to something else. No YAML
    parsing -- pure filesystem.
    """
    f = Path(folder)
    return (
        (f / "config.yaml").is_file()
        and (f / "videos").is_dir()
        and (f / "labeled-data").is_dir()
    )


def _find_dlc_config(path):
    """Resolve ``path`` to the DLC ``config.yaml`` that contains it, or None.

    Resolves four input shapes:

    - ``config.yaml`` file -> that path (only if the sibling project structure exists)
    - DLC project folder -> ``folder / 'config.yaml'``
    - Any file inside a project (notably a video under ``videos/``) -> walks up
      ancestors until a DLC-root is found
    - Anything else (a bare video outside any project, a non-existent path) -> None

    Returning None signals Phase 1 to :func:`open`. Note the walk-up stops
    at the filesystem root; in practice DLC's layout means it terminates
    after one step.
    """
    p = Path(path)
    if not p.exists():
        return None

    if p.is_file() and p.name.lower() == "config.yaml":
        return p if _is_dlc_project_root(p.parent) else None

    if p.is_dir() and _is_dlc_project_root(p):
        return p / "config.yaml"

    if p.is_file():
        for ancestor in p.parents:
            if _is_dlc_project_root(ancestor):
                return ancestor / "config.yaml"

    return None


def _find_video_index(project, video_path):
    """Look up a video's index in ``project.video_list`` by filename stem.

    Stem matching (rather than full-path equality) is robust to the
    drive-letter / UNC / posix shuffling that :func:`rebase_to_config`
    already handles inside :class:`DLCProject`. Returns ``None`` if the
    video isn't part of the project.
    """
    target_stem = Path(video_path).stem
    for i, name in enumerate(project.video_names):
        if name == target_stem:
            return i
    return None


def _session_inside_dlc_project(dustrack) -> Optional[Path]:
    """Return the DLC project root the session sits inside, or None.

    Reuses :func:`_find_dlc_config` for the filesystem walk-up so the
    structural check (``config.yaml + videos/ + labeled-data/``) stays
    in one place. ``dustrack._dlcproject`` is checked first as the cheap
    short-circuit: a session that was opened via ``dustrack.open(<project>)``
    or that survived a successful ``create_dlc_project`` already knows
    its project; we only fall back to walking up ``dustrack.fname``'s
    ancestors when the attribute is unset (e.g. a video opened bare
    that happens to live inside an existing project tree).
    """
    proj = getattr(dustrack, "_dlcproject", None)
    if proj is not None:
        config_path = getattr(proj, "config_path", None)
        if config_path is not None:
            return Path(config_path).parent
    fname = getattr(dustrack, "fname", None)
    if fname is None:
        return None
    config = _find_dlc_config(fname)
    return config.parent if config is not None else None


def _resolve_multi_video_from_list(path_list: list) -> tuple:
    """Validate that every entry of ``path_list`` resolves to one
    shared DLC project, returning ``(DLCProject, list[Path])``.

    Strict-single-project contract (Roadmap *Next 1.2.0* item 3,
    1.2.0a3 cut): every video in a multi-video session must belong to
    the same DLC project. Bare-video entries, mixed projects, and
    ``config.yaml`` paths all raise ``ValueError`` so the user can fix
    the input rather than landing in an undefined state.

    The returned video-path list is the input order (the user's
    queue), NOT the project's canonical order. Bundle indexing follows
    the queue.

    Raises:
        ImportError: ``deeplabcut`` isn't installed.
        ValueError: Any entry isn't inside a DLC project, or entries
            span multiple projects, or a non-video entry sneaks in.
    """
    # Lazy import to avoid cycle: dlcinterface imports nothing from
    # _dlc_paths (the predicates above are pure), but DLCProject's
    # constructor runs _ensure_dlc_loaded() which pulls in deeplabcut.
    # Keeping it lazy means callers that only need the cheap predicates
    # don't pay the DLC import cost.
    from .dlcinterface import DLCProject
    from .dlcloader import HAS_DLC

    if not HAS_DLC:
        raise ImportError(
            "dustrack.open: multi-video sessions require deeplabcut "
            "(every video must belong to a single DLC project)."
        )
    resolved: list[Path] = []
    config_paths: set = set()
    for p in path_list:
        if not p.is_file():
            raise ValueError(
                f"dustrack.open: multi-video entry {p!s} is not a file. "
                "Multi-video sessions accept videos inside one DLC project; "
                "pass a project folder to open every video in the project."
            )
        if _is_dlc_config_yaml(p):
            raise ValueError(
                f"dustrack.open: multi-video entry {p!s} is a DLC "
                "config.yaml. To open every video in a project, pass the "
                "config.yaml (or the project folder) by itself -- not "
                "as a list entry alongside videos."
            )
        cp = _find_dlc_config(p)
        if cp is None:
            raise ValueError(
                f"dustrack.open: multi-video entry {p!s} is not inside a "
                "DLC project. Multi-video sessions require every video to "
                "belong to one shared project."
            )
        config_paths.add(Path(cp).resolve())
        resolved.append(p)
    if len(config_paths) > 1:
        raise ValueError(
            "dustrack.open: multi-video entries span multiple DLC projects "
            f"({sorted(str(c) for c in config_paths)}). All entries must "
            "belong to one shared project."
        )
    project = DLCProject(str(next(iter(config_paths))))
    return project, resolved
