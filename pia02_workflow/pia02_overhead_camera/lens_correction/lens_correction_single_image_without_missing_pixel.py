from pathlib import Path
import math

import cv2
import numpy as np
import lensfunpy

# Camera / lens constants
CAMERA_MAKER = "Sony"
CAMERA_MODEL = "ILCE-6400"  # proxy for FX30 (same APS-C 1.5x crop)
LENS_MAKER = "Sigma"
LENS_MODEL = "16mm F1.4 DC DN"
FOCAL_LENGTH = 16.0  # mm
APERTURE = 1.4
DISTANCE = 1000.0  # metres (infinity; irrelevant for geometry distortion)

SAFETY_MARGIN = 10  # extra pixels beyond computed minimum on each side

INPUT_IMAGE = Path(r"\\192.168.1.104\home\piano\data\overhead_camera\last_frames\009-0-fx30_2_2274.png")


def build_modifier(width, height):
    """Create a lensfunpy Modifier calibrated to the original image dimensions.

    Uses scale=1.0 to avoid auto-scaling that would increase edge pixel loss.
    Returns the Modifier (not coords) so apply_geometry_distortion can be
    called with different xu/yu/width/height parameters.
    """
    db = lensfunpy.Database()
    cam = db.find_cameras(CAMERA_MAKER, CAMERA_MODEL)[0]
    lens = db.find_lenses(cam, LENS_MAKER, LENS_MODEL)[0]

    mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
    mod.initialize(FOCAL_LENGTH, APERTURE, DISTANCE,
                   scale=1.0, pixel_format=np.uint8)
    return mod


def compute_padding(coords, width, height):
    """Determine padding needed so every source pixel is reachable.

    Analyses the coordinate map at each edge to find the worst-case offset,
    then divides by the local gradient to get the number of extra output
    pixels required.

    Returns (pad_left, pad_right, pad_top, pad_bottom).
    """
    # -- LEFT edge: output col 0 maps to src_x > 0; need extra cols to reach 0
    left_src_x = coords[:, 0, 0]
    offset_left = float(left_src_x.max())  # worst-case (largest positive)
    if offset_left > 0:
        row = int(np.argmax(left_src_x))
        grad = float(coords[row, 1, 0] - coords[row, 0, 0])
        pad_left = math.ceil(offset_left / grad) + SAFETY_MARGIN
    else:
        pad_left = SAFETY_MARGIN

    # -- RIGHT edge: output col -1 maps to src_x < W-1; need extra cols
    right_src_x = coords[:, -1, 0]
    gap_right = float((width - 1) - right_src_x.min())  # worst-case
    if gap_right > 0:
        row = int(np.argmin(right_src_x))
        grad = float(coords[row, -1, 0] - coords[row, -2, 0])
        pad_right = math.ceil(gap_right / grad) + SAFETY_MARGIN
    else:
        pad_right = SAFETY_MARGIN

    # -- TOP edge: output row 0 maps to src_y; need extra rows to reach 0
    top_src_y = coords[0, :, 1]
    offset_top = float(top_src_y.max())
    if offset_top > 0:
        col = int(np.argmax(top_src_y))
        grad = float(coords[1, col, 1] - coords[0, col, 1])
        pad_top = math.ceil(offset_top / grad) + SAFETY_MARGIN
    else:
        pad_top = SAFETY_MARGIN

    # -- BOTTOM edge: output row -1 maps to src_y; need extra rows to reach H-1
    bot_src_y = coords[-1, :, 1]
    gap_bot = float((height - 1) - bot_src_y.min())
    if gap_bot > 0:
        col = int(np.argmin(bot_src_y))
        grad = float(coords[-1, col, 1] - coords[-2, col, 1])
        pad_bottom = math.ceil(gap_bot / grad) + SAFETY_MARGIN
    else:
        pad_bottom = SAFETY_MARGIN

    return pad_left, pad_right, pad_top, pad_bottom


def build_expanded_coords(mod, width, height, padding):
    """Generate a coordinate map for an expanded output canvas.

    The modifier was initialised with the ORIGINAL image dimensions.
    By passing negative xu/yu and larger width/height we get a bigger
    output grid whose source coordinates still reference the original image.
    """
    pad_left, pad_right, pad_top, pad_bottom = padding
    new_w = width + pad_left + pad_right
    new_h = height + pad_top + pad_bottom

    expanded = mod.apply_geometry_distortion(
        xu=-float(pad_left),
        yu=-float(pad_top),
        width=new_w,
        height=new_h,
    )
    if expanded is None:
        raise RuntimeError("No distortion calibration data available.")
    return expanded, (new_w, new_h)


def auto_crop_bounds(expanded_coords, width, height):
    """Find the tight bounding box of output pixels with valid source coords.

    Returns (row_min, row_max, col_min, col_max) inclusive, or None.
    """
    valid = (
        (expanded_coords[:, :, 0] >= 0) &
        (expanded_coords[:, :, 0] <= width - 1) &
        (expanded_coords[:, :, 1] >= 0) &
        (expanded_coords[:, :, 1] <= height - 1)
    )
    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)
    if not rows.any():
        return None
    rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    return rmin, rmax, cmin, cmax


def print_edge_analysis(coords, width, height, label=""):
    """Print source coordinate coverage at each edge for diagnostics."""
    if label:
        print(f"\n--- {label} ---")
    print(f"  Left   edge  max src_x: {coords[:, 0, 0].max():.2f}  (want <= 0)")
    print(f"  Right  edge  min src_x: {coords[:, -1, 0].min():.2f}  (want >= {width - 1})")
    print(f"  Top    edge  max src_y: {coords[0, :, 1].max():.2f}  (want <= 0)")
    print(f"  Bottom edge  min src_y: {coords[-1, :, 1].min():.2f}  (want >= {height - 1})")


def correct_image(input_path, auto_crop=True):
    """Apply lens distortion correction without losing any edge pixels."""
    input_path = Path(input_path)
    frame = cv2.imread(str(input_path))
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    h, w = frame.shape[:2]
    print(f"Input: {input_path.name}  ({w} x {h})")

    # 1. Analyse standard coords to understand edge loss
    mod = build_modifier(w, h)
    standard_coords = mod.apply_geometry_distortion()
    if standard_coords is None:
        raise RuntimeError("No distortion calibration data.")
    print_edge_analysis(standard_coords, w, h, "Standard coords (before expansion)")

    # 2. Compute required padding
    padding = compute_padding(standard_coords, w, h)
    print(f"Padding: L={padding[0]}, R={padding[1]}, T={padding[2]}, B={padding[3]}")

    # 3. Generate expanded coordinate map
    mod2 = build_modifier(w, h)
    expanded_coords, (new_w, new_h) = build_expanded_coords(mod2, w, h, padding)
    print(f"Expanded canvas: {new_w} x {new_h}")
    print_edge_analysis(expanded_coords, w, h, "Expanded coords (after expansion)")

    # 4. Remap
    corrected = cv2.remap(
        frame, expanded_coords, None,
        cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # 5. Auto-crop excessive black borders
    if auto_crop:
        bounds = auto_crop_bounds(expanded_coords, w, h)
        if bounds is not None:
            rmin, rmax, cmin, cmax = bounds
            corrected = corrected[rmin:rmax + 1, cmin:cmax + 1]
            print(f"Auto-cropped: {corrected.shape[1]} x {corrected.shape[0]}")

    # 6. Save
    out_path = input_path.parent / f"{input_path.stem}-lens_corrected_new.png"
    cv2.imwrite(str(out_path), corrected)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    if not INPUT_IMAGE.exists():
        print(f"Input image not found: {INPUT_IMAGE}")
        exit(1)
    correct_image(INPUT_IMAGE, auto_crop=True)
