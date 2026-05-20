# Key Design Concepts

## Core Data Model

DUSTrack organizes point tracking data using three hierarchical concepts: *Annotations*, *Labels*, and *Layers*.

### Annotations

An **annotation** is a single data point consisting of a frame number (integer) and a 2D pixel location (x, y coordinates as floats).

Example: `5: [120.3, 240.7]` means at frame 5, the point is located at pixel coordinates (120.3, 240.7).

Annotations are the atomic units of tracking data—each represents one observation of a point at one moment in time.

### Labels

A **label** is the identifier for one anatomical point being tracked throughout the video. Each label contains a collection of annotations across different frames. Labels are numeric strings for keyboard efficiency: `"0"`, `"1"`, `"2"`, ..., `"9"`: press number keys to switch between labels during annotation. When tracking more than 10 points, access extended ranges (`"10"`-`"19"`, `"20"`-`"29"`, etc.) accessed via `Shift+,` and `Shift+.`

While numbers are used to for efficiency of manual annotations, we recommend creating a separate `.txt` file containing **Human-readable names** for each anatomical landmark. Create a `dlc_trackermap.txt` file to map numeric labels to anatomical names:
```
point0 - muscle_boundary
point1 - fascia
point2 - bone_surface
```

**Data structure**:
Each label stores a dictionary mapping frame numbers to pixel coordinates, representing the pixel location of the same anatomical landmark in three different frames:
```python
"0": {5: [120.3, 240.7], 10: [122.1, 241.5], 15: [123.8, 242.1]}  # label "0": 3 annotations
```

### Layers

A **layer** is a complete annotation set stored in one file. Each layer contains multiple labels (typically tracking different anatomical landmarks). 

**Complete example**:
```python
layer.data = {
    "0": {5: [120.3, 240.7], 10: [122.1, 241.5], 15: [123.8, 242.1]},  # label "0" with 3 annotations
    "1": {5: [130.5, 250.2], 10: [131.8, 251.0]},                      # label "1" with 2 annotations
}
```

In this structure:
- **Layer**: The entire `layer.data` dictionary plus metadata (video path, etc.)
- **Labels**: `"0"` and `"1"` (two different anatomical points)
- **Annotations**: Individual frame-coordinate pairs like `5: [120.3, 240.7]`


**Understanding Layers and File Naming**

Layers in DUSTrack represent individual annotation sessions or annotators. Each layer corresponds to a *single file* and is managed internally by the `VideoAnnotation` class. The layer naming convention typically uses the annotator's initials.

When you save your annotations using the `s` keyboard shortcut, DUSTrack generates a JSON file following the pattern: `{video_name}_annotations_{layer_name}.json`.

DeepLabCut (DLC) uses the `.h5` file format. DUSTrack's annotations are formatted into a `.h5` file when preparing this data to train a ResNet model using DLC. DLC's model predictions are also in `.h5` formatand can be directly loaded into DUSTrack (see [Working with Layers](#working-with-layers)). 

**Tip**: Ese the `VideoAnnotation` class independently for analysis scripts to read annotation stored in `.json` or `.h5` files. 

**Why multiple layers?**

DUSTrack supports simultaneous layers to enable:

1. **Iterative Refinement Workflow**
   - Layer `"manual"`: Initial manual annotations on sparse frames
   - Layer `"dlc_iter1"`: First DLC model predictions
   - Layer `"dlc_iter2"`: Refined model after correcting mislabeled frames
   - Layer `"lkmovavg_0.500"`: Post-processed results with jitter reduction

2. **Comparison Between Sources**
   - Compare manual ground truth vs. automated predictions
   - Evaluate different post-processing window sizes
   - Visually assess inter-rater reliability between annotators

3. **Non-destructive Editing**
   - Keep original annotations while experimenting with refinements
   - Roll back to previous versions if needed

### Current Layer vs. Overlay Layer

The GUI displays two layers simultaneously:

- **Current Layer** (opaque markers): The layer you're actively editing. Press `-`/`=` to cycle through layers.
- **Overlay Layer** (translucent trace): A reference layer for comparison. Press `[`/`]` to cycle through overlays or set to `None`.

**Common patterns**:
- **Manual refinement**: Current = `"manual"`, Overlay = `"dlc_iter1"` (copy good predictions with `c` key)
- **Quality assessment**: Current = `"lkmovavg_0.500"`, Overlay = `"dlc_iter1"` (compare smoothed vs. raw)

### Working with Layers

**Creating layers**:
```python
import dustrack

# Initialize with single layer
tracker = dustrack.open('video.mp4', "manual")

# If video_annotations_manual.json file exists in the same folder as video.mp4, 
# then annotations from this file will be loaded into the "manual" layer.
# Otherwise, an "empty" layer will be created.

# Initialize with multiple layers
tracker = dustrack.open('video.mp4', ["manual", "dlc_iter1"])

# Initialize with specific file paths
tracker = dustrack.open('video.mp4', {
    'manual': 'path/to/manual_annotations.json',
    'predictions': 'path/to/dlc_predictions.h5'
})
```

**Copying between layers** (keyboard shortcuts):
- `c`: Copy current label's annotation from overlay to current layer (current frame only)
- `Alt+C`: Copy annotations at marked frames of interest from overlay
- `Ctrl+Alt+C`: Copy annotations in selected interval from overlay

**The Buffer Layer**: Every DUSTrack session includes a special `"buffer"` layer for temporary storage and experimentation. It serves as a scratch space.

**Derived / dense layers**: a few layers are produced *from* other layers by built-in operations and are rendered as continuous lines (not dots) by default:
- `dlc_iteration-N_M` — DLC inference for iteration `N` at step `M`. Produced by **Train DLC model**; added live to the session after training completes.
- `dlccorr` — Applied-manual-corrections layer. Produced by **Apply manual corrections**: the active sparse manual layer's edits are spliced into the current DLC overlay's per-frame trace, yielding a dense annotation. Excluded from DLC training input.
- `lkmovavg_<window>` — Lucas-Kanade RSTC jitter-reduction output. Produced by **Reduce jitter** on a dense source layer (typical input: `dlccorr` after corrections).

The line-vs-dot rendering picks itself: any layer whose name starts with `dlc_`, equals `dlccorr`, or contains `lkmovavg` renders as a line; everything else (manual annotations) renders as dots.

---

## Workflow buttons and sidebar groups

The workflow panel groups its buttons by task into five sections separated by horizontal lines, with a pastel per-group palette so visual scanning matches the workflow phase:

| Group | Buttons | What they do |
|-------|---------|--------------|
| **Workflow** | Create DLC project · Train DLC model · Apply manual corrections · Reduce jitter · Save annotation as... | The five-step pipeline: scaffold a DLC project from your manual annotations → train ResNet → splice manual corrections into DLC predictions → smooth jitter with LK-RSTC → export. |
| **Display** | Trace: line / Trace: dot · Freeze/Unfreeze plot axes · (EnhanceWidget) | Visual controls. The **EnhanceWidget** is two sliders — CLAHE clip and Gamma — plus a `[None | Auto]` row; *None* resets to raw image (pass-through), *Auto* infers per-frame heuristics from the current frame histogram. |
| **Niche** | Replace existing from overlay · Remove layer | Layer-mutating affordances. *Replace existing from overlay* copies the overlay's annotations into the current layer at frames the current layer is missing. *Remove layer* drops a layer from the session (file on disk is **not** deleted — to fully discard, save annotation as... then delete the original). |
| **Utilities** | Refresh UI (F5) · Keyboard shortcuts | *Refresh UI* re-renders all annotations from `.data` — recovery affordance if you mutated data from the Python REPL behind DUSTrack's back. *Keyboard shortcuts* opens the live cheatsheet (modeless dialog, grouped by section). |
| **Swap layers** | (single button) | Toggles foreground ↔ overlay. Positioned directly above the statevars widget it manipulates so the cause/effect is one glance. |

The workflow order in the **Workflow** group matches the steady-state DUSTrack workflow:
1. Manually annotate sparse frames (no button — use the GUI directly).
2. **Create DLC project** — scaffolds `config.yaml`, `videos/`, `labeled-data/`. Migrates your manual-annotation layer files to the project folder.
3. **Train DLC model** — runs DLC `extract_frames → train → evaluate → analyze_videos` in a worker thread under a modal progress overlay (DLC stdout/stderr stream live into the overlay log and to the launching terminal). On completion the new `dlc_iteration-N_M` trace layer is added live to the session and a fresh `iteration-{N+1}` manual layer is created as the destination for the next round of corrections.
4. **Apply manual corrections** — splices the active manual layer's sparse edits into the DLC overlay's dense per-frame trace, producing a `dlccorr` layer.
5. **Reduce jitter** — runs LK-RSTC over a dense source layer (usually `dlccorr`), producing `lkmovavg_<window>`.
6. **Save annotation as...** — Qt file dialog seeded with `{video_stem}_annotations_{layer_name}.json`; useful for one-off exports outside the auto-save pattern.

### Discarding unsaved annotations

If you've made manual edits to the active layer but want to revert without closing the window, click **Discard unsaved annotations** (or use the keyboard shortcut from the cheatsheet). DUSTrack confirms via a severity-aware modal (the title is red if the diff is destructive), then either reloads the layer's JSON from disk or — if no backing file exists — resets the layer to empty.

This is for **manually authored** layers; the operation refuses on `dlccorr` / `dlc_*` / dense layers, which are regenerable from upstream and should be removed and re-applied rather than reloaded.

### Closing the window with unsaved changes

DUSTrack guards window close (X button, Alt+F4, `plt.close()`) by scanning every manual annotation layer for in-memory diffs vs disk. If any layer has unsaved changes, a modal lists the per-layer breakdown (`+added / -removed / ~modified` counts) and offers **Save all / Discard / Cancel** — with *Cancel* as the default, so accidental Enter/Esc won't lose data.

---

## Video Frame Indexing

DUSTrack uses **zero-based frame indexing** following Python conventions:
- First frame: `frame 0`
- For a 100-frame video: frames `0` to `99`

**Frame numbers in the GUI**:
- The current frame number appears in the interface state panel
- Click on trajectory plots (right-click) to jump to specific frames
- Frame markers show your current position in the x and y trajectory plots

**Temporal navigation**:
DUSTrack treats videos as sequences of discrete frames. When you:
- Annotate frame 10, then frame 50, you're creating a **sparse annotation**
- Run optical flow interpolation, it fills frames 11-49 based on the boundary conditions
- Load DLC predictions, you typically get **dense annotations** (one prediction per frame)

---

## Frames of Interest

When working with videos, you often need to focus on (e.g. jump back and forth between) **specific frames** rather than sequentially reviewing all frames. DUSTrack's "frames of interest" feature lets you mark and rapidly navigate between important frames.

### What are Frames of Interest?

Frames of interest are **user-marked frames** that deserve special attention. Common use cases:

1. **Evaluating model predictions**: Mark frames where DLC predictions look questionable
2. **Assessing annotation consistency**: Mark frames to compare across multiple layers
3. **Sparse manual annotation**: Mark candidate frames for labeling before running optical flow interpolation
4. **Quality control**: Mark frames for systematic review during iterative refinement

### Marking and Navigation

**Marking frames**:
- Press `m` while viewing a frame to toggle it as a frame of interest
- Marked frames appear as vertical gray bars in the trajectory plots

**Navigation shortcuts**:
- `Alt+,`: Jump to previous frame of interest
- `Alt+.`: Jump to next frame of interest

**Batch operations**:
- `Alt+C`: Copy annotations at all frames of interest from overlay to current layer

### Workflow Example

To review DLC predictions:
1. Set current layer to `"manual"`, overlay to `"dlc_iter1"`
2. Step through video and press `m` on frames with poor predictions
3. Use `Alt+,` / `Alt+.` to cycle between marked frames
4. Press `c` to copy good predictions from overlay, or manually correct bad ones

Without frames of interest, reviewing predictions in a 1000-frame video means clicking through linearly or manually noting frame numbers. With frames of interest, you create a **focused review queue** that streamlines quality assessment and refinement.

---

## Event Intervals for Optical Flow Interpolation

DUSTrack provides an interval selection system that works seamlessly with the overlay layer concept to enable **optical flow interpolation** between sparse annotations.

### What are Event Intervals?

An **event interval** is a range of frames defined by a **start frame** and **end frame**. This interval specifies where optical flow algorithms should interpolate point positions based on boundary conditions from the overlay layer.

### How It Works

The typical workflow combines three elements:
1. **Overlay layer** containing sparse annotations (e.g., manual labels at frames 10 and 50)
2. **Event interval** marking the range to interpolate (frames 10-50)
3. **Optical flow interpolation** filling in frames 11-49 using Lucas-Kanade RSTC

### The `z-z-a` Pattern

The most common operation is the **`z`, `z`, `a` sequence**:

1. **First `z`**: Hover your mouse over the trajectory plot at the start frame and press `z` to mark interval start
2. **Second `z`**: Hover over the end frame and press `z` to mark interval end
   - A gray shaded region appears showing the selected interval
3. **Press `a`**: Triggers Lucas-Kanade RSTC interpolation for the current label
   - Uses overlay layer positions at start/end frames as boundary conditions
   - Fills intermediate frames in the current layer

**Example workflow**:
```python
import dustrack

# Setup: manual layer has labels at frames 10 and 50
tracker = dustrack.open('video.mp4', "manual")

# In GUI:
# 1. Set overlay to "manual" (to use as reference)
# 2. Create new layer for interpolated results
# 3. Hover over trajectory plot at x=10, press 'z'
# 4. Hover over trajectory plot at x=50, press 'z'
# 5. Press 'a' to interpolate current label
# 6. Result: frames 11-49 now have smoothly interpolated positions
```

### Overlay Layer as Boundary Conditions

The event interval system is designed to work **with the overlay layer**:
- The overlay layer provides the **source annotations** (start and end points)
- The current layer receives the **interpolated results**
- This allows non-destructive interpolation: original annotations remain unchanged

**Common pattern**: Use manual annotations as overlay, create a new layer for interpolated results:
1. Current layer: `"manual_interpolated"` (empty or partially filled)
2. Overlay layer: `"manual"` (sparse annotations)
3. Select interval spanning two manual annotations
4. Press `a` to fill intermediate frames in current layer

### Interpolating Multiple Labels

- `Alt+A`: Interpolate **all labels** in the selected interval (not just current label)
- Useful when you've manually labeled multiple points at the interval boundaries

### Design Rationale

Without event intervals, you'd need to:
- Manually annotate every frame (tedious for 1000+ frame videos)
- Write custom scripts to specify frame ranges
- Risk overwriting original annotations during interpolation

With event intervals + overlay system:
- Visually select ranges on trajectory plots
- Preserve original data in overlay layer
- Rapidly fill sparse annotations with optical flow

---

## Architecture: Building on datanavigator

DUSTrack is built on top of the `datanavigator` package, which provides the foundational video browsing and annotation framework -- specifically `datanavigator.VideoBrowser`, the asset-manager + button-row + state-variable widgets, and the events / `_qt` scaffolding. Understanding this relationship helps when extending DUSTrack or troubleshooting issues. More importantly, to develop a better understanding of design concepts in DUSTrack's user interface, you need to dig into the `datanavigator` package.

### Inheritance Hierarchy

```python
datanavigator.VideoBrowser           # Foundational video-browsing GUI
    └── dustrack.pointtracking._DUSTrackBase   # Internal: point-annotation
        │                                       # primitives (no DLC)
        └── dustrack.DUSTrack                   # Public: DLC workflow + UI

dustrack.VideoAnnotation             # Annotation data container; ships with
                                     # the LK-RSTC postprocess hook attached
                                     # at import time
```

Since DUSTrack inherits from datanavigator:
- All datanavigator keyboard shortcuts work in DUSTrack
- datanavigator documentation applies to basic operations

Note: the point-annotation UI primitives (formerly
`datanavigator.VideoPointAnnotator`) relocated to `dustrack` in
1.2.0a1 and the class was renamed to `_DUSTrackBase`. Use
`dustrack.DUSTrack` directly; direct construction of the internal
base is discouraged (it's the implementation of DUSTrack, not a
public API).

---
