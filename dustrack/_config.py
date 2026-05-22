"""
Configuration for DUSTrack.

Two layers:

- **Module-level globals** (``EXPERIMENTER``, ``DLC3_USE_LAST_SNAPSHOT``)
  -- code-level defaults baked at import time.
- **Per-user JSON store at** ``~/.dustrack/config.json`` -- cross-session
  state that needs to survive Python restarts. Today: seed-bundles
  root (read/written by ``dustrack.seed``); recent-session history
  (read/written here, consumed by the no-arg picker flow). All
  accessors are import-cheap and fail safely on missing / unreadable
  files (empty dict).

Recent-session shape (1.2.0a3 seed-window cut): one JSON key,
``recent_sessions``, holding a list of path-tuples. A single-video
session writes ``[v.mp4]``; a multi-video session writes the full
bundle list ``[v0.mp4, v1.mp4, ...]``; a project-folder open writes
the resolved video list. Click-to-reopen calls
``dustrack.open(<the stored list>)`` so cardinality is preserved.

Pre-1.2.0a3 history (separate ``recent_videos`` / ``recent_folders``
keys) is migrated on first read: every old video string becomes a
1-element session, every old folder string becomes a 1-element
session pointing at the folder. The old keys are dropped from disk
once the unified list is written; back-compat accessors
(``get_recent_videos`` / ``get_recent_folders``) project the unified
list back down for legacy callers.

The user-config helpers live here, not in ``seed.py``, because they
are general-purpose. ``seed.py`` continues to re-export
``_read_user_config`` / ``_write_user_config`` for back-compat with
any direct callers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

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

# Cap on the unified session list. 20 (down from the pre-1.2.0a3 cap of
# 25) keeps the seed-modal's recent column short enough to scan at a
# glance.
_RECENT_LIST_CAP = 20


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
# Recent-session history (unified list-of-path-tuples)
# ---------------------------------------------------------------------

def _migrate_legacy_recent_keys(cfg: dict) -> dict:
    """Fold pre-1.2.0a3 ``recent_videos`` / ``recent_folders`` keys into
    the unified ``recent_sessions`` list and drop them from the dict.

    Order preservation: the legacy ``recent_videos`` list (most-recent
    first) lands at the front of ``recent_sessions``; the legacy
    ``recent_folders`` list follows. Each legacy string becomes a
    1-element session (the unified entry shape).

    Idempotent: if neither legacy key exists, returns ``cfg`` unchanged.
    The migrated dict is what the caller persists; this function does
    NOT write to disk.
    """
    legacy_videos = cfg.pop("recent_videos", None)
    legacy_folders = cfg.pop("recent_folders", None)
    if legacy_videos is None and legacy_folders is None:
        return cfg
    existing = list(cfg.get("recent_sessions") or [])
    # Wrap legacy strings as 1-element sessions. Validate shape so a
    # half-migrated config (someone hand-edited the JSON) doesn't crash.
    migrated: list[list[str]] = []
    for s in legacy_videos or []:
        if isinstance(s, str):
            migrated.append([s])
    for s in legacy_folders or []:
        if isinstance(s, str):
            migrated.append([s])
    # Dedupe against ``existing`` so a re-migration doesn't double-up.
    seen_keys = {tuple(e) for e in existing if isinstance(e, list)}
    for entry in migrated:
        key = tuple(entry)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        existing.append(entry)
    cfg["recent_sessions"] = existing[:_RECENT_LIST_CAP]
    return cfg


def _is_polluted_entry(entry: list[str]) -> bool:
    """Return True if ``entry`` is the packaged seed asset, which
    is never a useful history entry.

    Pre-1.2.0a3-cleanup the seed video leaked into the history store
    via some test / fallback paths. Real users don't open the
    synthetic seed asset on purpose; auto-drop it on read.

    Other forms of stale entries (network drive offline, deleted
    file, transient temp paths from test fixtures) are NOT pruned
    here: a re-mounted drive recovers them, and deleted files are
    already filtered from the UI surface by
    :func:`_filter_existing_entries`. Aggressive temp-dir pruning
    would catch legitimate pytest ``tmp_path`` fixtures in test
    runs that exercise the recorder against the real config.
    """
    if not entry:
        return True
    first = entry[0]
    if not isinstance(first, str) or not first:
        return True
    try:
        resolved = Path(first).resolve(strict=False)
    except (OSError, ValueError):
        return False
    seed_path = Path(__file__).resolve().parent / "_data" / "seed_video.mp4"
    if resolved == seed_path:
        return True
    return False


def _prune_polluted_entries(cfg: dict) -> tuple[dict, bool]:
    """Drop never-useful pollution from the ``recent_sessions`` list.

    Returns ``(cfg, changed)`` -- callers persist when ``changed`` is
    True. The pollution heuristics live in :func:`_is_polluted_entry`;
    keep them narrow so user-meaningful entries (stale network paths,
    moved files) survive.
    """
    raw = cfg.get("recent_sessions")
    if not isinstance(raw, list):
        return cfg, False
    kept: list = []
    changed = False
    for entry in raw:
        if not isinstance(entry, list):
            changed = True
            continue
        if _is_polluted_entry(entry):
            changed = True
            continue
        kept.append(entry)
    if changed:
        cfg["recent_sessions"] = kept
    return cfg, changed


def _read_sessions_raw() -> list[list[str]]:
    """Read the on-disk ``recent_sessions`` list, migrating legacy keys
    in-place if present. Returns the raw string list-of-lists; callers
    resolve paths + filter for existence."""
    cfg = _read_user_config()
    needs_write = False
    if "recent_videos" in cfg or "recent_folders" in cfg:
        cfg = _migrate_legacy_recent_keys(cfg)
        needs_write = True
    cfg, pruned = _prune_polluted_entries(cfg)
    needs_write = needs_write or pruned
    if needs_write:
        # Persist the migration / prune so the next read is a no-op.
        try:
            _write_user_config(cfg)
        except OSError:
            # Best-effort; a read-only home directory must not block
            # the read path. The in-memory cfg is correct; next launch
            # will retry the write.
            pass
    raw = cfg.get("recent_sessions") or []
    out: list[list[str]] = []
    for entry in raw:
        if isinstance(entry, list) and all(isinstance(s, str) for s in entry):
            out.append(entry)
    return out


def _dedupe_prepend(lst: list[list[str]], item: list[str], cap: int) -> list[list[str]]:
    """Move ``item`` to the front of ``lst`` (or insert if absent),
    drop later duplicates, cap at ``cap``. Comparison is on the
    list-of-strings shape -- callers normalise to resolved strings
    before calling."""
    out = [item]
    for x in lst:
        if x == item:
            continue
        out.append(x)
        if len(out) >= cap:
            break
    return out


def _filter_existing_entries(entries: list[list[str]]) -> list[list[Path]]:
    """Resolve each entry's strings to ``Path`` objects, keeping only
    entries whose first element still exists on disk (file OR
    directory). Stale individual paths inside a multi-element entry
    are kept -- a re-mounted network drive recovers them; dropping
    the whole entry on one missing path is too aggressive.

    The on-disk JSON keeps every recorded entry; this filter only
    trims what the picker UI surfaces.
    """
    out: list[list[Path]] = []
    for entry in entries:
        paths = [Path(s) for s in entry]
        if not paths:
            continue
        first = paths[0]
        if first.exists():
            out.append(paths)
    return out


def record_recent_session(paths: Sequence[Union[str, Path]]) -> None:
    """Push ``paths`` (the full bundle list of the session that just
    closed) to the front of the unified recent-sessions list and
    persist.

    Called from the DUSTrack close-guard on every successful session
    close. Dedupes case-sensitively on the resolved string tuple --
    re-opening the same exact session moves it to the front rather
    than duplicating. The first element of the tuple is the "active"
    video (the one ``dustrack.open()`` will land on when this entry
    is clicked from the recent list).
    """
    resolved = [str(Path(p).resolve()) for p in paths]
    if not resolved:
        return
    # Defense-in-depth: pollution skip even at the write side. The
    # seed-tracker's close-guard already short-circuits on
    # ``_is_seed_session = True``, but a future caller wiring its own
    # save path should not re-introduce the seed asset / temp-dir
    # pattern that the read-side prune would just drop again.
    if _is_polluted_entry(resolved):
        return
    cfg = _read_user_config()
    if "recent_videos" in cfg or "recent_folders" in cfg:
        cfg = _migrate_legacy_recent_keys(cfg)
    cfg, _ = _prune_polluted_entries(cfg)
    current = list(cfg.get("recent_sessions") or [])
    # Normalise existing entries to the same shape (list[str]).
    current = [list(e) for e in current if isinstance(e, list)]
    cfg["recent_sessions"] = _dedupe_prepend(current, resolved, _RECENT_LIST_CAP)
    _write_user_config(cfg)


def get_recent_sessions() -> list[list[Path]]:
    """Return the unified recent-sessions list, most-recent first,
    filtered to entries whose first (active) path still exists on
    disk.

    Each entry is a ``list[Path]``: single-video sessions are
    1-element; multi-video sessions are N-element. Caller renders
    each entry as one row in the picker.
    """
    return _filter_existing_entries(_read_sessions_raw())


# ---------------------------------------------------------------------
# Back-compat accessors (pre-1.2.0a3 surface)
# ---------------------------------------------------------------------

def record_recent_video(path: Union[str, Path]) -> None:
    """Pre-1.2.0a3 single-video recorder. Forwarded to the unified
    writer as a 1-element session so old callers keep working without
    touching the underlying storage layout."""
    record_recent_session([path])


def get_recent_videos() -> list[Path]:
    """Pre-1.2.0a3 video accessor. Projects the unified list down to
    the per-entry active path, filtered to entries whose active path
    is a file (not a directory). Multi-video sessions surface as the
    bundle-0 video here."""
    sessions = get_recent_sessions()
    out: list[Path] = []
    for entry in sessions:
        first = entry[0]
        if first.is_file():
            out.append(first)
    return out


def record_recent_folder(path: Union[str, Path]) -> None:
    """Pre-1.2.0a3 folder recorder. Forwarded as a 1-element session.
    Multi-video sessions today record via :func:`record_recent_session`
    with the full bundle list -- the close-guard no longer calls this
    helper directly, but external callers (and tests) still can."""
    record_recent_session([path])


def get_recent_folders() -> list[Path]:
    """Pre-1.2.0a3 folder accessor. Projects the unified list down to
    1-element entries whose path is a directory."""
    sessions = get_recent_sessions()
    out: list[Path] = []
    for entry in sessions:
        if len(entry) != 1:
            continue
        first = entry[0]
        if first.is_dir():
            out.append(first)
    return out


def get_last_video_picker_dir() -> Optional[Path]:
    """Derive "where the file picker should land next time".

    Resolution order:
    1. Parent of the most-recent existing entry whose active path is a
       file.
    2. Most-recent existing entry whose active path is a directory.
    3. ``None`` -- caller falls back to the OS default (typically the
       user's home / Documents).

    Derivation avoids a duplicate ``last_picker_dir`` key that could
    drift out of sync with the history list.
    """
    sessions = get_recent_sessions()
    for entry in sessions:
        first = entry[0]
        if first.is_file():
            return first.parent
    for entry in sessions:
        first = entry[0]
        if first.is_dir():
            return first
    return None
