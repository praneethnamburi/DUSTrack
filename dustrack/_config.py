"""
Configuration for DUSTrack.

Two layers:

- **Module-level globals** (``EXPERIMENTER``, ``DLC3_USE_LAST_SNAPSHOT``)
  -- code-level defaults baked at import time.
- **Per-user JSON store at** ``~/.dustrack/config.json`` -- cross-session
  state that needs to survive Python restarts. Today: seed-bundles
  root (read/written by ``dustrack.seed``); recent-video / recent-
  folder history (read/written here, consumed by the no-arg picker
  flow). All accessors are import-cheap and fail safely on missing /
  unreadable files (empty dict).

The user-config helpers live here, not in ``seed.py``, because they
are general-purpose. ``seed.py`` continues to re-export
``_read_user_config`` / ``_write_user_config`` for back-compat with
any direct callers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

# Experimenter name used when creating DeepLabCut projects.
# This identifier is embedded in project paths and configuration files.
EXPERIMENTER = "x"

# For DeepLabCut 3.x: whether to use the last trained snapshot instead of
# the snapshot marked as "best" during evaluation.
# - True: Use the most recent snapshot (last training iteration)
# - False: Use the snapshot with the lowest test error (best performance)
DLC3_USE_LAST_SNAPSHOT = True


# ---------------------------------------------------------------------
# Per-user JSON store
# ---------------------------------------------------------------------

# Path stays out of the project tree so every dustrack session on this
# machine shares it. JSON for hand-inspectability.
_USER_CONFIG_DIR = Path.home() / ".dustrack"
_USER_CONFIG_PATH = _USER_CONFIG_DIR / "config.json"

# Cap on history lists. Keeps the JSON readable and bounds the
# stale-on-disk filter cost. The future "Open recent" modal can page
# beyond this if needed.
_RECENT_LIST_CAP = 25


def _read_user_config() -> dict:
    """Return the per-user config dict, or ``{}`` on missing /
    unreadable file. Never raises -- a corrupt config must not strand
    callers."""
    if not _USER_CONFIG_PATH.is_file():
        return {}
    try:
        return json.loads(_USER_CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _write_user_config(cfg: dict) -> None:
    """Write ``cfg`` atomically (well: write-then-rename would be safer,
    but a partial write here can only happen on disk-full / power-loss,
    and a malformed config is recovered by ``_read_user_config``)."""
    _USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _USER_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


# ---------------------------------------------------------------------
# Recent-video / recent-folder history
# ---------------------------------------------------------------------

def _dedupe_prepend(lst: list[str], item: str, cap: int) -> list[str]:
    """Move ``item`` to the front of ``lst`` (or insert if absent),
    drop later duplicates, cap at ``cap``. Comparison is on the
    resolved string the caller passes -- normalisation is the caller's
    job."""
    out = [item]
    for x in lst:
        if x == item:
            continue
        out.append(x)
        if len(out) >= cap:
            break
    return out


def _filter_existing(paths: list[str], want_dir: bool) -> list[Path]:
    """Resolve string paths to ``Path`` objects, keeping only the ones
    that still exist on disk and match the requested kind. Used by the
    ``get_recent_*`` accessors so the future "Open recent" modal never
    shows entries that would 404 on click."""
    out: list[Path] = []
    for s in paths:
        p = Path(s)
        if want_dir and p.is_dir():
            out.append(p)
        elif not want_dir and p.is_file():
            out.append(p)
    return out


def record_recent_video(path: Union[str, Path]) -> None:
    """Push ``path`` to the front of the recent-videos list and
    persist. Called from the DUSTrack close-guard on every successful
    session close. Dedupes case-sensitively on the resolved string."""
    resolved = str(Path(path).resolve())
    cfg = _read_user_config()
    current = list(cfg.get("recent_videos") or [])
    cfg["recent_videos"] = _dedupe_prepend(current, resolved, _RECENT_LIST_CAP)
    _write_user_config(cfg)


def get_recent_videos() -> list[Path]:
    """Return the recent-videos list, most-recent first, filtered to
    paths that still exist on disk. The on-disk JSON keeps stale
    entries (cheap re-resolution at read time), so a mounted-then-
    unmounted network drive doesn't permanently lose history."""
    cfg = _read_user_config()
    return _filter_existing(list(cfg.get("recent_videos") or []), want_dir=False)


def record_recent_folder(path: Union[str, Path]) -> None:
    """Push ``path`` (a folder) to the front of the recent-folders
    list. Used for multi-video sessions: when ``dustrack.open([...])``
    is called with paths sharing a common parent, that parent is
    recorded here so the future picker can offer the whole folder as
    a one-click choice."""
    resolved = str(Path(path).resolve())
    cfg = _read_user_config()
    current = list(cfg.get("recent_folders") or [])
    cfg["recent_folders"] = _dedupe_prepend(current, resolved, _RECENT_LIST_CAP)
    _write_user_config(cfg)


def get_recent_folders() -> list[Path]:
    """Return the recent-folders list, most-recent first, filtered to
    folders that still exist on disk."""
    cfg = _read_user_config()
    return _filter_existing(list(cfg.get("recent_folders") or []), want_dir=True)


def get_last_video_picker_dir() -> Optional[Path]:
    """Derive "where the file picker should land next time".

    Resolution order:
    1. Parent of the most-recent existing entry in ``recent_videos``.
    2. Most-recent existing entry in ``recent_folders``.
    3. ``None`` -- caller falls back to the OS default (typically the
       user's home / Documents).

    Derivation avoids a duplicate ``last_picker_dir`` key that could
    drift out of sync with the history lists.
    """
    vids = get_recent_videos()
    if vids:
        return vids[0].parent
    folders = get_recent_folders()
    if folders:
        return folders[0]
    return None