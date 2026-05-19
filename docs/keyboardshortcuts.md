# GUI Keyboard Shortcuts

The most reliable cheatsheet is the **live one inside DUSTrack**:
click the **Keyboard shortcuts** button in the workflow panel's
*Utilities* group, and a modeless dialog opens with every binding
that's currently active, grouped by section (Annotation / Frame
navigation / Layer-label / LK-interpolate / View / File / Other).
Each button in the sidebar also shows its shortcut suffix (e.g. the
*Refresh UI* button reads `Refresh UI  (f5)`).

A static reference for the most common shortcuts is below — the live
cheatsheet is authoritative when the two disagree.

## Annotation

| Key | Action |
|-----|--------|
| `0`–`9` | Switch to label 0–9 in the current layer |
| `Shift+,` / `Shift+.` | Cycle through extended label ranges (`10`–`19`, `20`–`29`, ...) |
| `s` | Save the current layer to disk |
| `c` | Copy current label's annotation from overlay to current layer at the current frame |
| `Alt+C` | Copy annotations at all *frames of interest* from overlay to current layer |
| `Ctrl+Alt+C` | Copy annotations in the selected event interval from overlay |

## Frame navigation

| Key | Action |
|-----|--------|
| `←` / `→` | Previous / next frame |
| `,` / `.` | Step backwards / forwards (larger step) |
| `m` | Toggle current frame as a *frame of interest* |
| `Alt+,` / `Alt+.` | Jump to previous / next frame of interest |

## Layer

| Key | Action |
|-----|--------|
| `-` / `=` | Cycle current layer |
| `[` / `]` | Cycle overlay layer (or set to `None`) |
| `v` | Run LK interpolation on the current label at the current frame |

## LK interpolation (event interval)

| Key | Action |
|-----|--------|
| `z` (first press, over a trajectory plot) | Mark interval start |
| `z` (second press) | Mark interval end |
| `a` | Run LK-RSTC interpolation for current label across the selected interval |
| `Alt+A` | Run LK-RSTC for *all* labels in the interval |

## View

| Key | Action |
|-----|--------|
| `r` | Reset view. Hovering over the image pane → resets image zoom/pan. Hovering over a trajectory plot → resets time axis to full video range and autoscales y to visible data. |
| `Alt+R` | Reset trajectory plots to the data extent (zoom to the annotated region). |
| `F5` | Refresh UI — re-render all annotations from `.data`. Recovery affordance for direct REPL mutations. |
| `Ctrl+C` | Copy the entire DUSTrack window (sidebar + image pane + traces) to the clipboard as an image. |
