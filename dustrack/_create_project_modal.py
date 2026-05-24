"""Qt UI for the Create DLC Project options dialog (1.3.0a2).

One modal that lets the user override the three things that were
previously implicit when clicking Create DLC Project: the project
**name** (was ``f"{video}_{layer}"`` with no way to edit), the
project **folder** (was always the video's parent -- no way to put
projects in a dedicated ``M:\\DLC_MODELS``), and the **experimenter**.

Pre-populated from :func:`_default_create_project_options` so OK-ing
straight through reproduces the old defaults (minus the seed-video
name bug). The default folder is the last project root the user
created into (persisted in ``~/.dustrack/config.json`` via
:func:`dustrack._config.get_last_project_root`), falling back to the
active video's parent.

No "link videos" toggle: hard-linking with copy fall-back is the
always-right default (see ``DLCProject.__init__``'s ``link_videos``
kwarg for the scripted override).

The default-options builder + validator live in :mod:`._overlays`
next to the dialog class factory, mirroring the Train-modal split.
"""

from __future__ import annotations

from typing import Optional

from . import _config
from ._overlays import (
    _default_create_project_options,
    _make_create_project_options_class,
)


def prompt_create_project_options(
    qt_window,
    *,
    video_fname,
    layer_name,
    experimenter,
) -> Optional[dict]:
    """Show the Create DLC Project modal and return the user's choices.

    Builds the initial state via :func:`_default_create_project_options`
    (seeding the folder from the remembered last project root), runs
    ``CreateProjectOptionsDialog`` synchronously, and -- on Create --
    persists the chosen folder as the new last project root so the
    next project defaults there too.

    Returns:
        dict | None: ``{"name", "path", "experimenter"}`` on Create,
        or ``None`` if the user clicked Cancel (caller returns without
        creating a project).
    """
    CreateProjectOptionsDialog = _make_create_project_options_class()
    initial_state = _default_create_project_options(
        video_fname=video_fname,
        layer_name=layer_name,
        experimenter=experimenter,
        last_project_root=_config.get_last_project_root(),
    )
    result = CreateProjectOptionsDialog(
        qt_window,
        initial_state=initial_state,
    ).exec_()
    if result is None:
        return None
    # Remember where the user put this project so the next Create DLC
    # Project modal defaults to the same root.
    _config.record_project_root(result["path"])
    return result
