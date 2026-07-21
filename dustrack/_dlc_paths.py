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

import contextlib
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence


@contextlib.contextmanager
def symlink_as_hardlink():
    """Make ``os.symlink`` hard-link instead, for the duration of the block.

    DLC's ``create_new_project(copy_videos=False)`` places videos by
    trying ``os.symlink``, then ``mklink``, and **falling back to a full
    copy**. On Windows both link attempts need a privilege the user
    typically doesn't have, so the copy is the normal path -- which
    defeats the whole point of
    :func:`link_or_copy_videos_into_project`. DUSTrack does eventually
    notice the copy, delete it, and hard-link over it, so the end state
    was always correct; the cost was paying a full copy first.

    That cost is not academic at pia02 scale: its telemed exports run
    3-16 GB each, so a 10-video project wrote ~65 GB and took ~10
    minutes before being replaced by links. Across a 61-participant
    cohort that is multiple TB of pointless I/O.

    A hard link satisfies DLC's need (a real file at the destination
    path) strictly better than a symlink would, so this simply lets its
    first attempt succeed. Cross-volume -- where ``os.link`` genuinely
    cannot work -- falls through to the real ``os.symlink`` and DLC's
    original ladder.
    """
    real_symlink = os.symlink

    def _link_instead(src, dst, *args, **kwargs):
        try:
            os.link(src, dst)
        except OSError:
            real_symlink(src, dst, *args, **kwargs)

    os.symlink = _link_instead
    try:
        yield
    finally:
        os.symlink = real_symlink


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


_DEFAULT_LINK_SIDECARS: tuple[str, ...] = (".dnav-toc",)


def link_or_copy_videos_into_project(
    project_videos_dir: Path,
    source_videos: Sequence[Path],
    link_videos: Optional[bool] = None,
    link_sidecars: tuple[str, ...] = _DEFAULT_LINK_SIDECARS,
    dest_names: Optional[Sequence[str]] = None,
) -> list[Path]:
    """Place each source video inside ``<project>/videos/`` and return the
    new in-project paths (suitable for rewriting DLC's ``video_sets``).

    On the same volume, hard links the source via :func:`os.link` so the
    file's bytes exist once on disk while DLC sees a real file at the
    in-project path. Cross-volume (or any :class:`OSError` from
    ``os.link``) falls back to :func:`shutil.copy2` so a project on a
    different drive still gets a self-contained ``videos/`` folder. This
    replaces DLC's ``copy_videos=True`` flow: the lossless h265 mp4s
    produced by the new ``telemed.process()`` pipeline (~1-2 GB per
    20k-frame recording) make per-project copies impractical at scale.

    Sidecars listed in ``link_sidecars`` (default: ``.dnav-toc``) are
    placed alongside if present beside the source. Linking the dnav TOC
    saves the in-project copy from rebuilding it on first read.

    Args:
        project_videos_dir: Destination directory (``<project>/videos/``).
            Created if missing.
        source_videos: Source video paths to place.
        link_videos: ``None`` (default) auto-picks: link, fall back to
            copy on :class:`OSError` with a stderr warning. ``True``
            requires linking and raises ``OSError`` on failure. ``False``
            always copies.
        link_sidecars: Extensions appended to each source video filename
            to look for sidecars (e.g. ``("video.mp4", ".dnav-toc")``
            picks up ``video.mp4.dnav-toc``). Sidecars use the same
            link/copy policy as the video; absence is silent.
        dest_names: Optional in-project filenames, 1:1 with
            ``source_videos``. ``None`` (default) keeps each source's own
            name. Renaming at link time exists because a hard link is
            just a second directory entry for the same bytes -- the
            in-project name is free to differ from the source's, at no
            storage cost and with no copy.

            The pia02 case is why: its telemed exports are named
            ``pia02_s061_003 fav piece 20251210 145114_b2.mp4`` --
            spaces, and long enough that DLC's ~58-character prediction
            suffix pushes the full path toward Windows' 260-character
            limit. Linking them in as ``pia02_s061_003_LFAc.mp4`` keeps
            the corpus convention and the path budget without touching
            the source tree.

    Returns:
        In-project paths corresponding 1:1 to ``source_videos``. Caller
        is responsible for rewriting ``config['video_sets']`` keys
        (DLC's ``create_new_project(copy_videos=False)`` registers the
        source paths; we want the in-project paths so downstream code
        sees the videos as project-local).

    Notes:
        Idempotent on re-entry: if a target already exists and is the
        same inode as the source (already hard-linked), it's left
        alone. If a target exists but is a different file (e.g. a real
        copy from a prior ``copy_videos=True`` project), it is also
        left alone -- we don't overwrite, since the existing file is
        functionally equivalent for DLC's purposes.
    """
    project_videos_dir = Path(project_videos_dir)
    project_videos_dir.mkdir(parents=True, exist_ok=True)
    if dest_names is not None and len(dest_names) != len(source_videos):
        raise ValueError(
            f"dest_names has {len(dest_names)} entries for "
            f"{len(source_videos)} source videos -- they must correspond 1:1"
        )
    in_project_paths: list[Path] = []
    for i, src in enumerate(source_videos):
        src = Path(src)
        dst = project_videos_dir / (src.name if dest_names is None else dest_names[i])
        _place_one(src, dst, link_videos=link_videos)
        for ext in link_sidecars:
            sidecar_src = src.with_name(src.name + ext)
            if sidecar_src.exists():
                sidecar_dst = dst.with_name(dst.name + ext)
                _place_one(sidecar_src, sidecar_dst, link_videos=link_videos)
        in_project_paths.append(dst)
    return in_project_paths


def _place_one(src: Path, dst: Path, *, link_videos: Optional[bool]) -> None:
    """Hard-link or copy ``src`` to ``dst`` per the ``link_videos`` mode.

    See :func:`link_or_copy_videos_into_project` for the mode semantics.

    If ``dst`` exists:

    - same inode as ``src`` -> already linked, no-op.
    - same bytes (file size match -- coarse but cheap) -> a copy of
      ``src`` is there (typically DLC's own ``copy_videos=False``
      fall-back path put it there: DLC tries a symlink and falls back
      to a real copy on Windows without symlink privilege). Replace
      with a hard link per ``link_videos`` mode so disk usage drops
      back to one inode.
    - different bytes -> raise. The caller's contract is "place
      ``src`` at ``dst``"; silently leaving a different file at
      ``dst`` would lie about the result.
    """
    if dst.exists():
        if _is_same_inode(src, dst):
            return
        if _looks_like_existing_copy(src, dst):
            dst.unlink()
        else:
            raise FileExistsError(
                f"{dst} already exists with different size than {src}. "
                "Refusing to overwrite -- remove the file manually if "
                "the replacement is intentional."
            )
    if link_videos is False:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError as e:
        if link_videos is True:
            raise
        print(
            f"[link_or_copy_videos_into_project] cross-volume or "
            f"filesystem-unsupported hard link for {src.name!r}: "
            f"copying instead ({e.strerror})",
            file=sys.stderr,
        )
        shutil.copy2(src, dst)


def _is_same_inode(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def _looks_like_existing_copy(src: Path, dst: Path) -> bool:
    """Cheap heuristic for "``dst`` is a copy of ``src``": same size.

    Same size on its own is weak evidence but fits the DLC-side flow:
    DLC's ``copy_videos=False`` fall-back uses :func:`shutil.copy`, so
    a freshly-created project's ``videos/<stem>.<ext>`` is either
    byte-equal or absent. We don't md5 here because the typical telemed
    h265 file is ~1-2 GB and hashing it adds minutes of I/O to project
    creation. A future stricter mode could md5 small samples.
    """
    try:
        return src.stat().st_size == dst.stat().st_size
    except OSError:
        return False
