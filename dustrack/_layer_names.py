"""DLC bodypart / annotation-layer name helpers.

Two pure-helper concerns shared by the GUI, the file manager, and the
DLC project wrapper:

* :func:`_dlc_bodyparts_to_layer_labels` -- given a DLC project's
  ``bodyparts``, produce the labels a new manual annotation layer
  should carry. Single source of truth for the
  ``["point0", "point1"]`` -> ``["0", "1"]`` conversion.

* :func:`_is_dense_layer_name` -- True if a layer name implies dense
  per-frame coverage (DLC inference output, ``dlccorr`` splice, or any
  LK-RSTC jitter-reduced output). Controls default plot type
  (line vs scatter) and overlay-pin behavior.

Extracted from ``dlcinterface.py`` in dustrack 1.2.0rc1.
"""
from __future__ import annotations


_DENSE_LAYER_PREFIXES = ("dlc_", "dlccorr")
_DENSE_LAYER_SUBSTRINGS = ("lkmovavg",)


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
    return (
        any(name.startswith(p) for p in _DENSE_LAYER_PREFIXES)
        or any(s in name for s in _DENSE_LAYER_SUBSTRINGS)
    )
