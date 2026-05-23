"""Qt UI for the seed-bundle pick / confirm flow.

Pairs with :mod:`dustrack.seed` (the logic side -- bundle
inspection, root path management, project install). The modals here
drive the user through:

1. **Pick** -- list-picker against the remembered bundles root
   (:func:`pick_from_seed_bundles`), OR file-dialog Browse when no
   root is set (:func:`browse_for_seed_bundle`).
2. **Confirm** -- show detected info (bodyparts, snapshot name,
   net type, description) and ask once before kicking off the
   create-and-seed flow (:func:`confirm_seed_bundle`).
3. **Remember root** -- after the first successful Browse, ask
   whether the picked parent should become the bundles root so
   the next session opens directly into the list-picker
   (:func:`maybe_remember_seed_bundles_root`).

:func:`prompt_seed_bundle` is the top-level orchestrator that loops
between the picker and Browse paths as the user navigates.

Extracted from ``gui.DUSTrack`` in the 1.2.0rc1 follow-up.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ._overlays import (
    _make_confirm_overlay_class,
    _make_seed_bundle_picker_class,
)
from .seed import (
    get_seed_bundles_root,
    inspect_seed_bundle,
    list_seed_bundles,
    set_seed_bundles_root,
)


def prompt_seed_bundle(qt_window, active_layer_name: str) -> Optional[str]:
    """Multi-step modal sequence that fires when ``Create DLC Project``
    is clicked with an empty active manual layer.

    Two entry points depending on whether a seed-bundles root has
    been remembered:

    - **Root set + non-empty**: opens a list-picker showing every
      valid bundle under the root with its name + bodyparts +
      description. Quick-select; no file-dialog navigation required.
    - **No root set / empty root**: opens the legacy intent
      overlay -> ``QFileDialog`` -> confirm path. After a successful
      pick, offers to remember the picked bundle's parent as the
      bundles root so next session uses the picker.

    Returns the validated bundle folder path on Accept, or ``None``
    on any Cancel / invalid bundle path. Caller (``create_dlc_project``)
    treats ``None`` as "user bailed -- leave the UI alone".
    """
    # Loop so the picker's "Change bundles root" action can re-open
    # the dialog against the new root, and "Browse elsewhere" can
    # fall through to the file-dialog branch.
    while True:
        root = get_seed_bundles_root()
        if root is not None and root.is_dir():
            bundles = list_seed_bundles(root)
        else:
            bundles = []

        if bundles:
            action = pick_from_seed_bundles(qt_window, root, bundles)
            if action is None:
                return None
            kind = action[0]
            if kind == "use":
                info = action[1]
                bundle_path = str(info["path"])
                if confirm_seed_bundle(qt_window, bundle_path, info):
                    return bundle_path
                return None
            if kind == "set_root":
                set_seed_bundles_root(action[1])
                continue  # re-list against the new root
            if kind == "browse":
                # Fall through to legacy Browse flow below.
                pass

        # Legacy flow: explain + Browse + validate + confirm.
        picked = browse_for_seed_bundle(qt_window, active_layer_name)
        if picked is None:
            return None
        # First-time-Browse polite ask: remember the parent as the
        # root so the picker takes over next session. Skip if a
        # root is already configured.
        if get_seed_bundles_root() is None:
            maybe_remember_seed_bundles_root(qt_window, picked)
        return picked


def pick_from_seed_bundles(qt_window, root, bundles):
    """Drive :class:`SeedBundlePickerDialog`. Returns the dialog's
    raw result tuple (or ``None`` on cancel).
    """
    PickerDialog = _make_seed_bundle_picker_class()
    return PickerDialog(qt_window, root=root, bundles=bundles).exec_()


def browse_for_seed_bundle(qt_window, active_layer_name: str) -> Optional[str]:
    """Legacy seed-bundle flow used when no bundles root is set
    (or the picker user clicked Browse elsewhere): intent
    overlay -> ``QFileDialog`` -> validate -> confirm. Returns
    the validated bundle path on accept, ``None`` on any cancel
    or invalid bundle.
    """
    from qtpy.QtWidgets import QFileDialog

    ConfirmOverlay = _make_confirm_overlay_class()

    # Only show the intent overlay when there's no remembered root --
    # if the user got here via "Browse elsewhere" from the picker,
    # they already understand the situation.
    if get_seed_bundles_root() is None:
        result = ConfirmOverlay(
            qt_window,
            title="No annotations in active layer",
            message=(
                f"Active layer {active_layer_name!r} has no labels. "
                "To create a DLC project from this session, "
                "either annotate frames manually first, or seed "
                "iteration-0 from a pre-trained snapshot bundle "
                "(a folder containing snapshot-*.pt + "
                "pytorch_config.yaml + pose_cfg.yaml).\n\n"
                "Inference from the bundled snapshot will run on "
                "the current video and load as a dense reference "
                "overlay; your manual refinements then become "
                "iteration-1."
            ),
            buttons=[
                ("Browse for seed bundle…", "primary"),
                ("Cancel", "neutral"),
            ],
            default="Cancel",
            severity="warning",
        ).exec_()
        if result != "Browse for seed bundle…":
            return None

    bundle_dir = QFileDialog.getExistingDirectory(
        qt_window,
        "Choose seed bundle folder",
        "",
        QFileDialog.ShowDirsOnly,
    )
    if not bundle_dir:
        return None

    try:
        info = inspect_seed_bundle(bundle_dir)
    except (FileNotFoundError, ValueError) as exc:
        ConfirmOverlay(
            qt_window,
            title="Invalid seed bundle",
            message=(
                f"The selected folder is not a usable seed bundle:\n\n"
                f"{exc}\n\n"
                "Re-click 'Create DLC Project' to try again."
            ),
            buttons=[("OK", "neutral")],
            default="OK",
            severity="error",
        ).exec_()
        return None

    if confirm_seed_bundle(qt_window, bundle_dir, info):
        return bundle_dir
    return None


def confirm_seed_bundle(qt_window, bundle_path, info) -> bool:
    """Final confirm-with-detected-info overlay shared by the
    picker and Browse paths. Returns True iff the user clicked
    ``Create and seed``.
    """
    ConfirmOverlay = _make_confirm_overlay_class()
    description = info.get("description") or "(no description)"
    result = ConfirmOverlay(
        qt_window,
        title="Confirm seed bundle",
        message=(
            f"Bundle: {bundle_path}\n"
            f"Snapshot: {info['snapshot'].name}\n"
            f"Bodyparts ({len(info['bodyparts'])}): {info['bodyparts']}\n"
            f"Net type: {info.get('net_type') or '(unset)'}\n"
            f"Description: {description}\n\n"
            "Create the project, install this snapshot as iteration-0, "
            "and run inference on the current video?"
        ),
        buttons=[
            ("Create and seed", "primary"),
            ("Cancel", "neutral"),
        ],
        default="Cancel",
        severity="info",
    ).exec_()
    return result == "Create and seed"


def maybe_remember_seed_bundles_root(qt_window, bundle_path) -> None:
    """After the first successful Browse pick, ask the user if
    they want to remember the bundle's parent folder as the
    seed-bundles root so the next session opens the list-picker
    directly. No-op if they say no (the next Browse will ask
    again).
    """
    ConfirmOverlay = _make_confirm_overlay_class()
    parent = Path(bundle_path).parent
    result = ConfirmOverlay(
        qt_window,
        title="Remember bundles location?",
        message=(
            f"Use this folder as your seed-bundles root?\n\n"
            f"{parent}\n\n"
            "Next time you click Create DLC Project on an empty "
            "layer, DUSTrack will list every bundle in this "
            "folder so you don't have to browse."
        ),
        buttons=[
            ("Remember it", "primary"),
            ("Not now", "neutral"),
        ],
        default="Remember it",
        severity="info",
    ).exec_()
    if result == "Remember it":
        set_seed_bundles_root(parent)
