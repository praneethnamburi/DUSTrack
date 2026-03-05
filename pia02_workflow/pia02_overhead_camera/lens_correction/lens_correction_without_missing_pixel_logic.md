# Lens Correction Without Missing Pixels — Logic Explained

## The Problem

The Sigma 16mm f/1.4 DC DN lens introduces **barrel distortion**: straight lines near the image edges bow outward. To correct this, we apply the inverse transformation (a pincushion warp) using calibration data from the Lensfun database.

However, barrel distortion correction **loses edge pixels**. Here's why:

### How lensfunpy's coordinate map works

lensfunpy uses **reverse mapping** (also called backward mapping). It generates a coordinate array `coords` of shape `(H, W, 2)` where:

```
coords[y, x] = (src_x, src_y)
```

This means: "To fill output pixel (x, y), sample from the original image at (src_x, src_y)."

### Why edges are lost

For barrel distortion correction, the transformation "stretches" the image outward. In reverse-mapping terms, this means output edge pixels sample from **inner** source pixels:

```
Original image (1920 x 1080)            Corrected image (1920 x 1080)
+----------------------------------+    +----------------------------------+
|  these pixels (x=0..26)         |    |                                  |
|  are NEVER sampled by any       |    |  output col 0 samples from       |
|  output pixel — they are lost   |    |  source x=27, not x=0           |
|                                  |    |                                  |
+----------------------------------+    +----------------------------------+
```

The output image has the **same dimensions** as the input. Since every output edge pixel maps to an inner source pixel, the source edge pixels have no corresponding output pixel — they vanish.

Concrete example from our setup:
- `coords[:, 0, 0].max() ≈ 17` — leftmost output column samples from source x ~17
- `coords[:, -1, 0].min() ≈ 1902` — rightmost output column samples from source x ~1902
- Source columns 0–16 and 1903–1919 are **never sampled** (lost)


## The Solution: Expanding the Output Canvas

### Key insight

lensfunpy's `apply_geometry_distortion()` accepts optional parameters:

```python
mod.apply_geometry_distortion(xu=0, yu=0, width=-1, height=-1)
```

- `xu`, `yu` — starting position of the output grid (can be **negative**)
- `width`, `height` — dimensions of the output grid (can be **larger** than original)

By passing negative `xu`/`yu` and larger dimensions, we generate a coordinate map for an **expanded** output canvas. The distortion model stays calibrated to the original image size, but the output grid extends beyond the original bounds, reaching those previously-lost source edge pixels.

```
Expanded output canvas (1920 + padding)
+------+----------------------------------+------+
| pad  |     original 1920 region         | pad  |
| left |                                  | right|
|      |                                  |      |
| NEW output pixels at x < 0 now         |      |
| sample from source x = 0..16           |      |
+------+----------------------------------+------+
```


## Step-by-Step Logic

### Step 1: `build_modifier(width, height)`

Creates a lensfunpy `Modifier` calibrated to the **original** image dimensions.

**Why return the Modifier instead of coords?** We need to call `apply_geometry_distortion()` twice — once with default parameters (to analyse edge offsets) and once with expanded parameters (to generate the final map).

**Why `scale=1.0`?** The default `scale=0.0` enables auto-scaling, which zooms in to avoid black borders in pincushion correction cases. For barrel correction, this auto-scaling actually makes edge loss **worse** (e.g., 23px lost instead of 17px). Using `scale=1.0` gives the raw distortion model.


### Step 2: `compute_padding(coords, width, height)`

Analyses the standard coordinate map to determine how many extra output pixels are needed on each side to reach source pixels at the image boundary (x=0, x=W-1, y=0, y=H-1).

**Logic for each edge (using LEFT as example):**

```
1. Look at all source x-coordinates along the left output column:
   left_src_x = coords[:, 0, 0]    # shape (H,)

2. Find the worst case — the row where the offset is largest:
   offset_left = max(left_src_x)

   If offset_left = 17, the leftmost output pixel at that row
   samples from source x=17. We need to "reach" source x=0.

3. Measure the local gradient — how many source pixels per output pixel:
   grad = coords[row, 1, 0] - coords[row, 0, 0]  ≈ 0.89

   This means each additional output pixel to the left covers ~0.89
   source pixels.

4. Compute padding:
   pad_left = ceil(17 / 0.89) + SAFETY_MARGIN
            = ceil(19.1) + 10
            = 30 pixels
```

The same logic applies to right (gap from last output to source W-1), top, and bottom.

**Why worst-case?** Barrel distortion is radial — it's strongest at the corners. The worst-case row/column ensures even corner pixels are covered.

**Why SAFETY_MARGIN = 10?** The gradient estimate is local (linear approximation of a nonlinear function). The margin accounts for floating-point imprecision and slight nonlinearity.


### Step 3: `build_expanded_coords(mod, width, height, padding)`

Generates the expanded coordinate map by calling `apply_geometry_distortion` with shifted origin and larger dimensions:

```python
expanded = mod.apply_geometry_distortion(
    xu=-pad_left,     # start the output grid pad_left pixels to the LEFT of origin
    yu=-pad_top,      # start pad_top pixels ABOVE origin
    width=new_w,      # W + pad_left + pad_right
    height=new_h,     # H + pad_top + pad_bottom
)
```

The modifier was initialized with the original W x H, so the distortion model is unchanged. We're simply asking: "What source coordinates correspond to output pixels in this larger grid?"

The new output pixels at the edges (x < 0 or x >= W in the original coordinate system) now map to source pixels near x=0 or x=W-1 — the previously-lost content.


### Step 4: `cv2.remap` with expanded coords

```python
corrected = cv2.remap(
    frame,                    # original image (H x W)
    expanded_coords,          # expanded map (new_H x new_W x 2)
    None,
    cv2.INTER_LANCZOS4,       # high-quality interpolation
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0),    # black for out-of-bounds source pixels
)
```

`cv2.remap` produces an output of size `(new_H, new_W)`. For each output pixel, it looks up the source coordinate from `expanded_coords` and samples from `frame`. If a source coordinate falls outside the original image bounds, it fills with black (border constant).


### Step 5: `auto_crop_bounds(expanded_coords, width, height)`

The expanded canvas may have a few rows/columns of pure black at the very edges (from the safety margin overshooting). This function finds the tight bounding box of output pixels whose source coordinates are within the valid range [0, W-1] x [0, H-1]:

```python
valid = (coords_x >= 0) & (coords_x <= W-1) & (coords_y >= 0) & (coords_y <= H-1)
```

It returns `(row_min, row_max, col_min, col_max)` — the crop rectangle that removes pure-black borders while **keeping all original content**.

Result: the final image is slightly larger than the original (e.g., ~1964 x 1104 for a 1920 x 1080 input) because the barrel distortion correction genuinely pushes content outward.


## Summary

```
Original approach:
  1920x1080 input  →  coords (1920x1080)  →  remap  →  1920x1080 output (edges lost)

New approach:
  1920x1080 input  →  standard coords (analyse edges)
                   →  compute padding (L=30, R=34, T=23, B=23)
                   →  expanded coords (1984x1126)
                   →  remap  →  1984x1126 output
                   →  auto-crop  →  ~1964x1104 output (all pixels preserved)
```
