"""Preflight: scan + diff + remediation logic for the Train DLC click.

Pure logic — no Qt. Two concerns fold together here:

1. **Unsaved-changes diff.** For each manual annotation layer in the
   session, compare the in-memory ``.data`` to whatever's on disk and
   report the per-(label, frame) added/removed/modified set. Used by
   both the Train pre-flight (so training consumes the user's edits)
   and the close-guard (so the user is prompted before losing data).

2. **Incomplete-frame scan.** For each manual layer, find frames that
   are missing one or more of the required bodyparts -- "required"
   resolved either from the layer's own touched-label set (no DLC
   project) or from the DLC project's ``config['bodyparts']`` (post-
   1.2.0a2 project-aware mode). Training would fail downstream on
   such frames, so the user is offered a "save and clean" remediation
   that drops them after writing a recovery sidecar next to the layer.

Companion module :mod:`._preflight_modal` is the Qt UI on top.

Extracted from ``gui.DUSTrack`` in the 1.2.0rc1 follow-up; all
functions are testable from synthetic dicts without instantiating
the GUI.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from ._layer_names import (
    _dlc_bodyparts_to_layer_labels,
    is_manual_annotation_layer,
    is_manual_layer_name,
)


# ---------------------------------------------------------------------
# Incomplete-frame scan
# ---------------------------------------------------------------------


def scan_incomplete_frames(
    data: dict,
    target_labels: "list[str] | None" = None,
) -> dict:
    """Find frames missing one or more required bodyparts in an
    annotation ``data`` dict (``{label: {frame: [x, y]}}``).

    Two modes:

    - ``target_labels=None`` (legacy, project-unaware): "required"
      = labels that have at least one annotation. Empty labels
      are treated as UI placeholders so they don't fail every
      frame. This is the right behavior for a session without a
      DLC project, where the user's declared label set may be the
      default ``[" 0"]`` bootstrap with no project bodyparts to
      anchor against.
    - ``target_labels=<list>`` (project-aware): "required" =
      exactly that list. Used when a ``DLCProject`` exists and
      its ``config['bodyparts']`` are the load-bearing label
      set -- training fails if a frame is missing any of them,
      regardless of whether the user has touched that label
      anywhere in the layer yet. Closes the case where a
      seeded project has bodyparts ``["point0", "point1"]``
      but the user annotated only the ``"0"`` label, and
      pre-flight wrongly reported the layer as complete.

    Returns ``{frame: [missing_label, ...]}`` for incomplete
    frames, frame-sorted with missing-labels lists in the same
    order as the required-label list. Empty dict iff every
    annotated frame has every required label.

    Pure data-in / data-out; testable from synthetic dicts.
    """
    if target_labels is None:
        required = [L for L, frames in data.items() if frames]
    else:
        required = list(target_labels)
    if not required:
        return {}
    all_frames: set = set()
    for L, frames in data.items():
        all_frames.update(frames.keys())
    incomplete: dict = {}
    for frame in sorted(all_frames):
        missing = [L for L in required if frame not in data.get(L, {})]
        if missing:
            incomplete[frame] = missing
    return incomplete


def scan_stray_labels(
    data: dict, target_labels: "list[str] | None",
) -> dict:
    """Find non-empty labels on the layer that are not in the
    project's required-label set.

    "Stray label" = a label that has at least one annotation but
    isn't a project bodypart. Two typical sources:

    1. User added a label intending to track another point, and
       hasn't promoted it to a project bodypart yet (the "Fork
       project" workflow extends bodyparts by spawning a new DLC
       project with ``bodyparts ∪ strays``; see the design rule
       in the dustrack spec).
    2. User added a label as a working / scratch tag and never
       intended it for training.

    Returns ``{label: [frame, ...]}`` for each stray label, with
    frames sorted. Returns ``{}`` when ``target_labels is None``
    (project-unaware mode has no external truth to call something
    "stray" against) or when no annotated labels lie outside the
    target set.

    Pure data-in / data-out; testable from synthetic dicts.
    """
    if target_labels is None:
        return {}
    target = set(target_labels)
    strays: dict = {}
    for label, frames in data.items():
        if label in target:
            continue
        if not frames:
            continue
        strays[label] = sorted(frames.keys())
    return strays


def build_dropped_incomplete_payload(data: dict, incomplete_frames: dict) -> dict:
    """Build the JSON payload for the dropped-incomplete sidecar.

    Each entry is ``{label: [x, y]}`` for the labels that *were*
    present at the dropped frame (the missing ones are the
    incompleteness). Frame keys are stringified for JSON compat.
    """
    payload: dict = {}
    for frame in incomplete_frames:
        present = {}
        for L, frames in data.items():
            if frame in frames:
                present[L] = [float(x) for x in frames[frame]]
        payload[str(frame)] = present
    return payload


def build_dropped_incomplete_sidecar_name(ann_fname, ts: str) -> str:
    """``<fstem>.dustrack-dropped-incomplete-<ts>`` in the
    annotation's directory.

    Intentionally avoids `.json` so DUSTrack's annotation-discovery
    glob (``{video_stem}*_annotations*.json`` in
    :meth:`DLCProject.extract_frames`) does not re-ingest the
    sidecar on a subsequent training run.
    """
    p = Path(ann_fname)
    return str(p.parent / f"{p.stem}.dustrack-dropped-incomplete-{ts}")


def save_dropped_incomplete_sidecar(ann, incomplete_frames: dict):
    """Persist the dropped-frame contents next to the given layer.

    Returns the sidecar path on success, ``None`` if the layer
    has no on-disk filename (in-memory only).
    """
    if ann.fname is None:
        return None
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    sidecar = build_dropped_incomplete_sidecar_name(ann.fname, ts)
    payload = build_dropped_incomplete_payload(ann.data, incomplete_frames)
    Path(sidecar).write_text(json.dumps(payload, indent=2))
    return sidecar


# ---------------------------------------------------------------------
# In-memory vs disk diff
# ---------------------------------------------------------------------


def normalize_layer_data(data: dict) -> dict:
    """Canonical form for diff comparison: int frame keys, float
    ``[x, y]`` values, empty labels filtered.

    Empty-label filtering exists for *diff* symmetry: a label
    with no frames contributes no entries to added / removed /
    modified regardless of whether it's present on one side
    only. With dnav 1.4.0rc2's first-class-label schema, both
    on-disk JSON and in-memory data may legitimately carry
    ``"label": {}`` entries (whereas pre-rc2,
    :meth:`VideoAnnotation.save` pruned them on the way out); the
    diff still works correctly because both inputs are filtered
    the same way here.
    """
    out: dict = {}
    for label, frames in data.items():
        if not frames:
            continue
        out[label] = {
            int(frame): [float(x) for x in xy] for frame, xy in frames.items()
        }
    return out


def load_layer_disk_data(ann_fname) -> dict:
    """Read the on-disk JSON for a layer and return its data in
    canonical form. Empty dict if the file does not exist or
    cannot be parsed (treated as "fully unsaved").
    """
    if ann_fname is None:
        return {}
    p = Path(ann_fname)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return normalize_layer_data(raw)


def diff_ann_vs_disk(mem_data: dict, disk_data: dict) -> dict:
    """Compare two canonical data dicts. Returns
    ``{"added": [...], "removed": [...], "modified": [...]}``
    where each list contains ``(label, frame)`` tuples in label-
    then frame-sorted order.

    Both inputs are assumed normalized via :func:`normalize_layer_data`.
    """
    added: list = []
    removed: list = []
    modified: list = []
    all_labels = set(mem_data) | set(disk_data)
    for label in sorted(all_labels):
        mem_frames = mem_data.get(label, {})
        disk_frames = disk_data.get(label, {})
        for frame in sorted(set(mem_frames) - set(disk_frames)):
            added.append((label, frame))
        for frame in sorted(set(disk_frames) - set(mem_frames)):
            removed.append((label, frame))
        for frame in sorted(set(mem_frames) & set(disk_frames)):
            if mem_frames[frame] != disk_frames[frame]:
                modified.append((label, frame))
    return {"added": added, "removed": removed, "modified": modified}


# ---------------------------------------------------------------------
# Cross-layer sweeps (combine diff + incomplete)
# ---------------------------------------------------------------------


def scan_unsaved_layers(annotations, video_fname) -> dict:
    """Across every manual annotation layer, return
    ``{layer_name: diff}`` for layers whose in-memory data differs
    from disk. Sibling of :func:`scan_unsaved_and_incomplete`
    scoped to the data-loss concern only -- the close-event guard
    does not care about incomplete-frame quality (that surfaces
    next time the user trains).
    """
    unsaved: dict = {}
    for ann in annotations:
        if not is_manual_annotation_layer(video_fname, ann.fname, ann.name):
            continue
        mem_data = normalize_layer_data(ann.data)
        disk_data = load_layer_disk_data(ann.fname)
        diff = diff_ann_vs_disk(mem_data, disk_data)
        if any(diff.values()):
            unsaved[ann.name] = diff
    return unsaved


def scan_unsaved_and_incomplete(
    annotations,
    video_fname,
    dlcproject=None,
) -> dict:
    """Across every manual annotation layer in the session, find
    in-memory-vs-disk diffs AND/OR incomplete frames. Returns
    ``{layer_name: {"diff": ..., "incomplete": ...}}`` for
    layers with at least one issue; layers with neither are
    omitted.

    Inclusion is name-based (see :func:`is_manual_layer_name`)
    so layers that haven't been saved yet -- ``ann.fname is None``,
    the state after a user opens a fresh video, annotates a
    partial layer, and clicks Train without saving first -- are
    still scanned for incompleteness. The disk-diff portion is
    guarded on ``ann.fname`` being set AND matching the
    ``<video_stem>_annotations*.json`` pattern; without a disk
    file there's nothing to diff against.

    Project-aware incomplete scan (1.2.0a2): when ``dlcproject``
    is not None, derives the required-label set from
    ``config['bodyparts']`` (mapped through
    :func:`_dlc_bodyparts_to_layer_labels`) and hands it to
    :func:`scan_incomplete_frames` as ``target_labels``.

    Stray-label scan (project-aware only): the same
    ``target_labels`` set is used to find labels with annotations
    that lie outside the project's bodyparts -- see
    :func:`scan_stray_labels`. When the modal-side asks the user
    whether to keep strays in the saved file (a separate checkbox;
    default unchecked), the remediation honours that decision via
    :func:`apply_pre_flight_remediations`'s ``strip_strays`` arg.
    Project-unaware mode yields no strays (no external truth).
    """
    target_labels: "list[str] | None" = None
    if dlcproject is not None:
        bodyparts = dlcproject.config.get("bodyparts") or []
        if bodyparts:
            target_labels = _dlc_bodyparts_to_layer_labels(bodyparts)

    issues: dict = {}
    for ann in annotations:
        if not is_manual_layer_name(ann.name):
            continue
        incomplete = scan_incomplete_frames(
            ann.data,
            target_labels=target_labels,
        )
        strays = scan_stray_labels(ann.data, target_labels)
        diff = {"added": [], "removed": [], "modified": []}
        if is_manual_annotation_layer(video_fname, ann.fname, ann.name):
            mem_data = normalize_layer_data(ann.data)
            disk_data = load_layer_disk_data(ann.fname)
            diff = diff_ann_vs_disk(mem_data, disk_data)
        if any(diff.values()) or incomplete or strays:
            issues[ann.name] = {
                "diff": diff,
                "incomplete": incomplete,
                "strays": strays,
            }
    return issues


# ---------------------------------------------------------------------
# Formatters (for modal bodies)
# ---------------------------------------------------------------------


def format_incomplete_breakdown(incomplete_frames: dict, max_rows: int = 200) -> str:
    """Multi-line per-bodypart breakdown for the pre-flight modal's
    detailed-text panel. Truncates very wide reports.
    """
    rows = []
    total = len(incomplete_frames)
    for i, (frame, missing) in enumerate(sorted(incomplete_frames.items())):
        if i >= max_rows:
            rows.append(f"... ({total - max_rows} more frames)")
            break
        rows.append(f"Frame {frame}: missing {', '.join(missing)}")
    return "\n".join(rows)


def format_unsaved_summary(unsaved: dict) -> str:
    """Per-layer +added / -removed / ~modified counts for the
    save-on-close modal's informative text.
    """
    lines = []
    for layer_name, diff in unsaved.items():
        a = len(diff.get("added", []))
        r = len(diff.get("removed", []))
        m = len(diff.get("modified", []))
        pieces = []
        if a:
            pieces.append(f"+{a} added")
        if r:
            pieces.append(f"-{r} removed")
        if m:
            pieces.append(f"~{m} modified")
        lines.append(f"  {layer_name!r}: " + ", ".join(pieces))
    return "\n".join(lines)


def format_pre_flight_summary(
    issues: dict,
    max_incomplete_examples: int = 3,
) -> str:
    """Per-layer breakdown for the unified pre-flight modal's
    detailed-text panel.
    """
    blocks = []
    for layer_name, info in issues.items():
        lines = [f"Layer {layer_name!r}:"]
        diff = info.get("diff", {})
        a = len(diff.get("added", []))
        r = len(diff.get("removed", []))
        m = len(diff.get("modified", []))
        if a or r or m:
            pieces = []
            if a:
                pieces.append(f"+{a} added")
            if r:
                pieces.append(f"-{r} removed")
            if m:
                pieces.append(f"~{m} modified")
            lines.append("  Unsaved changes: " + ", ".join(pieces))
        else:
            lines.append("  (no unsaved changes)")
        incomplete = info.get("incomplete", {})
        if incomplete:
            n = len(incomplete)
            lines.append(f"  Incomplete frames: {n}")
            for i, (frame, missing) in enumerate(sorted(incomplete.items())):
                if i >= max_incomplete_examples:
                    lines.append(f"    ... ({n - max_incomplete_examples} more)")
                    break
                lines.append(f"    frame {frame}: missing {', '.join(missing)}")
        else:
            lines.append("  (no incomplete frames)")
        strays = info.get("strays", {})
        if strays:
            n = len(strays)
            total_annotations = sum(len(frames) for frames in strays.values())
            lines.append(
                f"  Labels not in this DLC project: {n} "
                f"({total_annotations} annotation"
                f"{'s' if total_annotations != 1 else ''} across "
                "these label(s) will not be trained)"
            )
            for label, frames in sorted(strays.items(), key=lambda kv: kv[0]):
                lines.append(
                    f"    label {label!r}: {len(frames)} annotation"
                    f"{'s' if len(frames) != 1 else ''}"
                )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def has_strays(issues: dict) -> bool:
    """True iff any layer in the issues dict has at least one stray
    label with at least one annotation.

    Used by the modal layer to decide whether to render the "Save
    labels not in this DLC project" checkbox.
    """
    return any(info.get("strays") for info in issues.values())


# ---------------------------------------------------------------------
# Remediation (drop incomplete frames + save)
# ---------------------------------------------------------------------


def apply_pre_flight_remediations(
    annotations,
    video_fname,
    issues: dict,
    *,
    make_annotation_file_name,
    strip_strays: bool = True,
) -> None:
    """For each layer with issues, drop incomplete frames (with
    recovery sidecar), optionally strip stray labels, and save the
    (possibly trimmed) layer.

    Layers whose ``ann.fname`` is ``None`` (in-session unsaved
    layers, the first-time-training case) get a canonical fname
    derived from the video stem + layer name before save:
    ``<video_stem>_annotations_<layer_name>.json``. The recovery
    sidecar needs the same path resolved.

    ``make_annotation_file_name`` is injected to avoid an import cycle
    (``_file_management.py`` imports ``_layer_names`` already, and
    the canonical fname builder lives there).

    ``strip_strays`` -- when True (default), remove every annotation
    in stray labels (labels not in the project's bodyparts) before
    save. When False, preserve strays in the saved JSON so the user
    can promote them via the "Fork project" workflow later. The
    pre-flight modal exposes this as a single checkbox: default
    unchecked (strip) -- the user must deliberately opt in to keep
    extra-label work. Project-unaware mode has no strays so the
    arg is a no-op there.
    """
    for layer_name, info in issues.items():
        ann = annotations[layer_name]
        if ann.fname is None:
            ann.fname = str(
                make_annotation_file_name(
                    Path(video_fname),
                    annotation_suffix=ann.name,
                )
            )
        incomplete = info.get("incomplete") or {}
        if incomplete:
            save_dropped_incomplete_sidecar(ann, incomplete)
            # Drop the incomplete frames directly. Routing mutations
            # through ``ann.remove(label, frame)`` keeps the revision
            # counter consistent (see
            # ``feedback_revision_counter_invalidation_pattern``).
            for frame in incomplete:
                for label in list(ann.data.keys()):
                    if frame in ann.data[label]:
                        ann.remove(label, frame)
        if strip_strays:
            strays = info.get("strays") or {}
            for label, frames in strays.items():
                if label not in ann.data:
                    continue
                for frame in list(frames):
                    if frame in ann.data[label]:
                        ann.remove(label, frame)
        ann.save()


def has_trainable_labels(annotations, dlcproject=None) -> bool:
    """True if the project has *any* source of labels training
    could consume: at least one non-empty manual annotation layer
    in the session, or at least one ``.h5`` under the project's
    ``labeled-data/`` folder (already-extracted labels from
    prior iterations).

    Pure predicate -- no side effects.
    """
    for ann in annotations:
        if not is_manual_layer_name(ann.name):
            continue
        if any(ann.data.values()):
            return True
    if dlcproject is not None:
        labels_dir = Path(dlcproject.paths["labels"])
        if labels_dir.is_dir():
            for _ in labels_dir.rglob("*.h5"):
                return True
    return False
