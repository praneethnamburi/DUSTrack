"""DLC bodypart / annotation-layer name helpers.

Pure-helper concerns shared by the GUI, the file manager, and the
DLC project wrapper. Five name predicates and one filename
constructor; no Qt, no I/O (except :func:`get_fname_annotations`
which only assembles a path string).

* :func:`_dlc_bodyparts_to_layer_labels` -- given a DLC project's
  ``bodyparts``, produce the labels a new manual annotation layer
  should carry. Single source of truth for the
  ``["point0", "point1"]`` -> ``["0", "1"]`` conversion.

* :func:`_is_dense_layer_name` -- True if a layer name implies dense
  per-frame coverage (DLC inference output, ``dlccorr`` splice, or any
  LK-RSTC jitter-reduced output). Controls default plot type
  (line vs scatter) and overlay-pin behavior.

* :func:`is_manual_layer_name` / :func:`is_manual_annotation_layer` --
  name-only / name+file predicates for "is this a manual annotation
  layer (vs. a DLC trace / dlccorr / buffer / etc.)?". Used by the
  preflight scan and the save-on-close guard.

* :func:`get_fname_annotations` -- assemble the canonical
  ``<video_stem>_annotations_<layer>.<suffix>`` filename next to a
  video. The file-pattern is the inverse of
  :func:`is_manual_annotation_layer`'s match.

Extracted from ``dlcinterface.py`` in dustrack 1.2.0rc1; the manual-
layer predicates + filename constructor folded in during the 1.2.0rc1
follow-up refactor (lifted from ``gui.DUSTrack``).
"""

from __future__ import annotations

import os
from pathlib import Path


_DENSE_LAYER_PREFIXES = ("dlc_", "dlccorr", "lk_", "deblip")
_DENSE_LAYER_SUBSTRINGS = ("lkmovavg",)

#: Prefixes that mark a *derived* layer -- a model/flow prediction or a
#: corrected output that must NEVER be extracted as DLC training labels. Splits
#: cleanly from manual (hand-labelled) layers, which are the training feed.
#: Members: ``dlc`` (inference + LK-RSTC jitter outputs), ``lk_`` (the
#: flow-prediction layer paired with each DLC trace), ``blips`` (the flagged
#: blip frames, for inspection), ``deblip`` (the de-blipped corrected trace).
#: Generalizes the pre-2026-07 hard-coded ``startswith("dlc")`` exclusion.
_DERIVED_LAYER_PREFIXES = ("dlc", "lk_", "blips", "deblip")


def _dlc_bodyparts_to_layer_labels(bodyparts: list[str]) -> list[str]:
    """Convert DLC ``bodyparts`` to DUSTrack annotation-layer ``labels``.

    Mirrors :meth:`VideoAnnotation._dlc_trace_to_annotation_dict`
    (the h5-trace loader) and inverts ``DLCProject.__init__``'s
    ``[f'point{x}' for x in annotation_names]`` synthesis at
    project-creation time:

    - If every bodypart strips cleanly to a digit after removing
      the ``"point"`` prefix, the labels are the bare digits
      (``["point0", "point1"]`` -> ``["0", "1"]``; ``["point1",
      "point3"]`` -> ``["1", "3"]``).
    - Otherwise (e.g. ``["nose", "ear"]`` from a non-DUSTrack
      project), the labels are consecutive indices starting at 0.

    Single source of truth for "given a project's bodyparts, what
    labels should a new manual annotation layer carry?".
    """
    prefix = "point"
    stripped = [bp.removeprefix(prefix) for bp in bodyparts]
    if stripped and all(s.isdigit() for s in stripped):
        return stripped
    return [str(i) for i in range(len(bodyparts))]


def _is_dense_layer_name(name: str) -> bool:
    """True if ``name`` is a layer that should render as a line plot
    by default (DLC inference, the ``dlccorr`` manual-corrections
    splice, or any LK-RSTC jitter-reduced output).

    The LK output of a non-DLC source (e.g. ``dlccorr``) lands at a
    name like ``dlccorr_lkmovavg_0.500`` via
    :meth:`VideoFileManager.canonical_layer_name`'s ``_annotations``
    branch -- dense like a DLC trace, but it doesn't start with
    ``dlc_``. The substring match catches it without widening the
    prefix list. ``dlccorr`` itself is dense because it's the
    overlay's per-frame DLC trace with the active layer's sparse
    manual edits spliced in -- per-frame coverage is inherited from
    the overlay.
    """
    return any(name.startswith(p) for p in _DENSE_LAYER_PREFIXES) or any(
        s in name for s in _DENSE_LAYER_SUBSTRINGS
    )


def is_manual_layer_name(
    ann_name: str,
    special_names: tuple = ("dlccorr", "buffer"),
) -> bool:
    """Name-only predicate for "is this a manual annotation layer?".

    Pure string check on the layer name -- excludes ``dlccorr``
    (terminal output of apply_manual_corrections), ``buffer``
    (workspace scratch), and any layer whose name starts with
    ``"dlc"`` (DLC trace + process_with_lk LK outputs). Symmetric
    with the name-side of :func:`is_manual_annotation_layer`,
    which adds an on-disk file-pattern check on top.

    Lives separately so callers that care about incomplete-frame
    scanning -- which only needs the in-memory ``ann.data`` --
    can include layers that aren't yet saved to disk (``ann.fname
    is None``). The Train pre-flight uses this for inclusion and
    then guards the disk-diff portion on ``ann.fname`` being set;
    save-on-close uses the stricter file-aware predicate because
    a layer with no disk file has nothing to diff against.
    """
    if ann_name in special_names:
        return False
    if any(ann_name.startswith(p) for p in _DERIVED_LAYER_PREFIXES):
        return False
    return True


def is_manual_annotation_layer(
    video_fname,
    ann_fname,
    ann_name: str,
    special_names: tuple = ("dlccorr", "buffer"),
) -> bool:
    """Identify a manual annotation ``.json`` layer that feeds
    :meth:`DLCProject.extract_frames`.

    Rule: ``.json`` file alongside the video, matching the
    ``<video_stem>_annotations*.json`` pattern, AND the layer name
    passes :func:`is_manual_layer_name`. Excludes ``dlccorr`` /
    ``buffer`` / ``dlc*`` by name.

    File-based detection -- doesn't rely on the
    ``iteration-N`` naming convention, so a layer the user
    renamed to ``iter1`` or seeded with experimenter initials
    (e.g. ``pn``) is still picked up.
    """
    if ann_fname is None or video_fname is None:
        return False
    fname_path = Path(ann_fname)
    if fname_path.suffix != ".json":
        return False
    video_path = Path(video_fname)
    if fname_path.parent != video_path.parent:
        return False
    video_stem = video_path.stem
    stem = fname_path.stem
    if stem != f"{video_stem}_annotations" and not stem.startswith(
        f"{video_stem}_annotations_"
    ):
        return False
    return is_manual_layer_name(ann_name, special_names)


def get_fname_annotations(
    video_fname, annotation_name: str, suffix: str = ".json"
) -> str:
    """Construct the canonical filename for an annotation layer named
    ``annotation_name`` next to ``video_fname``.

    Pattern: ``<video_stem>_annotations_<annotation_name><suffix>``
    in the video's parent directory. The inverse of
    :func:`is_manual_annotation_layer`'s file-pattern match -- a
    file written here is recognised as a manual annotation layer on
    next discovery.

    Empty ``annotation_name`` produces ``<video_stem>_annotations<suffix>``
    (no trailing underscore).
    """
    video_path = Path(video_fname)
    return os.path.join(
        video_path.parent,
        video_path.stem
        + "_annotations"
        + (f"_{annotation_name}" if annotation_name else "")
        + suffix,
    )
