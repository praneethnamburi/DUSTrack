"""Lazy DeepLabCut loader: state machine + callback fan-out.

``import deeplabcut`` runs torch + several heavy submodules; on the
dlc3rc14 env it costs ~7 s of wall time. Paying that during
``import dustrack`` blocks the picker and the GUI cold-open even
though nothing on the annotation-only path actually touches DLC
(Create DLC Project, Train DLC model, the Phase 2 resume path).

``HAS_DLC`` is settled cheaply via ``importlib.util.find_spec`` so
the button-gating decision in ``DUSTrack.__init__`` doesn't need the
real import; the actual import runs through :func:`_ensure_dlc_loaded`
(synchronous, idempotent, thread-safe) and is normally kicked off by
:func:`_ensure_dlc_loaded_async` -- a daemon thread fired from
``dustrack.open()`` so DLC loads in parallel with the picker /
DUSTrack-construction / user-annotation work. The module globals
:data:`deeplabcut`, :data:`VideoWriter`, :data:`ScannerError` and
:data:`DLC3` stay ``None`` / ``False`` until the loader populates
them; every DLC-using method calls :func:`_ensure_dlc_loaded` as its
first line so the names are guaranteed bound at use.

Tests can short-circuit the loader by setting the module-level
:data:`_DLC_LOAD_STATE` to ``"done"`` and pre-binding the names; see
``tests/test_lazy_dlc_loader.py``.

Extracted from ``dlcinterface.py`` in dustrack 1.2.0rc1. The names
are re-imported by ``dlcinterface.py`` so existing
``from dustrack.dlcinterface import HAS_DLC`` paths keep working.
"""

from __future__ import annotations

import importlib
import importlib.util
import threading
import traceback
import warnings
from typing import Optional


HAS_DLC: bool = importlib.util.find_spec("deeplabcut") is not None
DLC3: bool = False
deeplabcut = None  # populated by _ensure_dlc_loaded
VideoWriter = None  # populated by _ensure_dlc_loaded
ScannerError = None  # populated by _ensure_dlc_loaded

_DLC_LOAD_LOCK = threading.Lock()
_DLC_LOAD_STATE: str = "pending"  # "pending" | "loading" | "done" | "missing"
_DLC_LOAD_THREAD: Optional[threading.Thread] = None
_DLC_LOAD_CALLBACKS: list = []  # called on the loader thread once "done" / "missing"


def _ensure_dlc_loaded() -> bool:
    """Import ``deeplabcut`` (and friends) on first call; idempotent.

    Returns ``True`` if DLC is available after the call, ``False`` if
    the package isn't installed. Thread-safe: concurrent callers block
    on a single import; the second-and-later calls return immediately.

    On success, the module globals ``deeplabcut``, ``VideoWriter``,
    ``ScannerError`` and ``DLC3`` are bound to the real values. On
    failure (no DLC installed) the globals stay ``None`` / ``False``.
    """
    global deeplabcut, VideoWriter, ScannerError, DLC3, _DLC_LOAD_STATE
    if _DLC_LOAD_STATE == "done":
        return True
    if _DLC_LOAD_STATE == "missing":
        return False
    if not HAS_DLC:
        with _DLC_LOAD_LOCK:
            _DLC_LOAD_STATE = "missing"
        _fire_dlc_load_callbacks()
        return False
    with _DLC_LOAD_LOCK:
        if _DLC_LOAD_STATE == "done":
            return True
        if _DLC_LOAD_STATE == "missing":
            return False
        # We hold the lock; nobody else can flip state until we either
        # finish the import or fail.
        try:
            _dlc = importlib.import_module("deeplabcut")
            _vw = importlib.import_module("deeplabcut.utils.auxfun_videos").VideoWriter
            _se = importlib.import_module("ruamel.yaml.scanner").ScannerError
        except ImportError:
            _DLC_LOAD_STATE = "missing"
            _fire_dlc_load_callbacks()
            return False
        deeplabcut = _dlc
        VideoWriter = _vw
        ScannerError = _se
        DLC3 = bool(getattr(_dlc, "__version__", "").startswith("3."))
        _DLC_LOAD_STATE = "done"
    _fire_dlc_load_callbacks()
    return True


def _ensure_dlc_loaded_async() -> Optional[threading.Thread]:
    """Fire-and-forget background DLC import. Idempotent.

    Spawns a daemon thread that calls ``_ensure_dlc_loaded()`` on first
    invocation; later calls (or calls after the load already finished)
    are no-ops. Returns the loader thread (the existing one on repeat
    calls) or ``None`` when there is no work to do (DLC missing, or the
    load already finished synchronously).

    Safe to call from any thread, including before any Qt application
    exists. The loader thread doesn't touch Qt -- DLC's own
    ``deeplabcut/__init__.py`` runs in light mode (``DLC loaded in
    light mode; you cannot use any GUI``) so there's no
    cross-thread-Qt hazard.
    """
    global _DLC_LOAD_THREAD
    if _DLC_LOAD_STATE in ("done", "missing"):
        return None
    if not HAS_DLC:
        # find_spec said no DLC; flip the state so subsequent sync
        # callers short-circuit without acquiring the lock.
        _ensure_dlc_loaded()
        return None
    with _DLC_LOAD_LOCK:
        if _DLC_LOAD_STATE in ("done", "missing"):
            return None
        if _DLC_LOAD_THREAD is not None and _DLC_LOAD_THREAD.is_alive():
            return _DLC_LOAD_THREAD
        _DLC_LOAD_THREAD = threading.Thread(
            target=_ensure_dlc_loaded,
            name="dustrack-dlc-preload",
            daemon=True,
        )
        _DLC_LOAD_THREAD.start()
        return _DLC_LOAD_THREAD


def _dlc_load_state() -> str:
    """Return the current loader state.

    Public-shaped (single underscore) for the workflow-button gate +
    tests; the value is one of ``"pending"`` (no import attempted yet),
    ``"loading"`` is reserved for an in-flight background import (set
    when ``_ensure_dlc_loaded_async`` is running and the sync call
    hasn't yet entered the lock), ``"done"`` (DLC is bound), or
    ``"missing"`` (find_spec returned None, or import raised).

    Today the lock-holding sync path doesn't publish a transitional
    ``"loading"`` state separately from ``"pending"``; the bg thread
    transitions ``pending -> done|missing`` once the import returns.
    Callers should treat ``"loading"`` as a superset of ``"pending"``
    (i.e. "not yet known to be ready").
    """
    if _DLC_LOAD_STATE == "pending" and _DLC_LOAD_THREAD is not None:
        return "loading"
    return _DLC_LOAD_STATE


def register_dlc_load_callback(cb) -> None:
    """Register a callback to fire when the lazy DLC load resolves.

    Callbacks run on the loader thread (typically the background
    daemon) once ``_DLC_LOAD_STATE`` flips to ``"done"`` or
    ``"missing"``. If the load has already resolved by the time the
    callback is registered, it fires immediately on the caller's
    thread.

    Used by ``DUSTrack.__init__`` to schedule a Qt-side refresh of the
    Workflow-button gates once the bg preload completes. Callbacks
    must be cheap and exception-safe; an exception in any one callback
    won't prevent the others from running.
    """
    fire_now = False
    with _DLC_LOAD_LOCK:
        if _DLC_LOAD_STATE in ("done", "missing"):
            fire_now = True
        else:
            _DLC_LOAD_CALLBACKS.append(cb)
    if fire_now:
        try:
            cb()
        except (
            Exception
        ):  # noqa: BLE001 -- defensive; callback errors must not propagate.
            traceback.print_exc()


def _fire_dlc_load_callbacks() -> None:
    """Internal: drain ``_DLC_LOAD_CALLBACKS`` after the loader resolves."""
    with _DLC_LOAD_LOCK:
        callbacks = list(_DLC_LOAD_CALLBACKS)
        _DLC_LOAD_CALLBACKS.clear()
    for cb in callbacks:
        try:
            cb()
        except (
            Exception
        ):  # noqa: BLE001 -- defensive; one bad callback can't block the rest.
            traceback.print_exc()


if not HAS_DLC:
    warnings.warn(
        "deeplabcut is not installed. You can still use the optical flow functions with DUSTrack.",
        stacklevel=2,
    )
