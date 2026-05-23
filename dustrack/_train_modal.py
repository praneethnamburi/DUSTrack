"""Qt UI for the Train DLC options dialog.

One modal that surfaces :meth:`DLCProject.train_iteration`'s arg
surface to the user: refine_mode (scratch / in-project / external),
source iteration/snapshot picker for in-project, Browse... for
external ``.pt``, training epochs (DLC3) / iterations (DLC2), and
a create-labeled-video toggle. Cancel returns ``None``; on Accept,
the user's choices are translated into kwargs ready to splat into
:meth:`DLCProject.train_iteration`.

Pairs with :mod:`dustrack.dlcinterface` (the logic side --
``DLCProject.train_iteration`` itself). The default-options
builder + kwarg translator live in :mod:`._overlays` next to the
dialog class factory.

Extracted from ``gui.DUSTrack`` in the 1.2.0rc1 follow-up.
"""

from __future__ import annotations

from ._overlays import (
    _default_training_options,
    _make_training_options_class,
    _training_options_to_train_iteration_kwargs,
)


def prompt_training_options(qt_window, dlcproject):
    """Show the Training options modal and return kwargs ready to
    splat into :meth:`DLCProject.train_iteration`.

    Builds the initial state via :func:`_default_training_options`
    from the live ``DLCProject``, runs ``TrainingOptionsDialog``
    synchronously, and translates the user's choices via
    :func:`_training_options_to_train_iteration_kwargs`.

    Returns:
        dict | None: kwargs for ``train_iteration``, or ``None``
        if the user clicked Cancel (caller returns without kicking
        off training).
    """
    TrainingOptionsDialog = _make_training_options_class()
    initial_state = _default_training_options(dlcproject)
    options = TrainingOptionsDialog(
        qt_window,
        initial_state=initial_state,
    ).exec_()
    if options is None:
        return None
    return _training_options_to_train_iteration_kwargs(options)
