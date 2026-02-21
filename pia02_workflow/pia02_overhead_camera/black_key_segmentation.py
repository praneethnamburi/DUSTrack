import cv2
import numpy as np


def segment_black_keys(img, threshold=0, min_aspect_ratio=2.0, min_area=500,
                       min_width=10, max_y_frac=0.7,
                       clahe_clip=2.0, clahe_grid=(8, 8)):
    """Segment black keys from an overhead piano keyboard image.

    Args:
        img: BGR image of the piano keyboard.
        threshold: Grayscale threshold for dark regions. 0 (default) uses
            Otsu's method to auto-pick the threshold. A manual value can
            be passed to override.
        min_aspect_ratio: Minimum height/width ratio for a contour to be
            considered a black key.
        min_area: Minimum contour area to keep (rejects noise).
        min_width: Minimum bounding-box width in pixels. Rejects thin gaps
            between white keys (typically 2-5px wide).
        max_y_frac: Black keys must start above this fraction of image height.
        clahe_clip: CLAHE clip limit for local contrast normalization.
        clahe_grid: CLAHE tile grid size.

    Returns:
        mask: Binary mask (uint8, 0/255) of the black keys.
        intermediates: Dict of intermediate images for debugging.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape

    # Step 1: CLAHE — normalize brightness locally so black keys are
    # uniformly dark even under uneven illumination.
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    equalized = clahe.apply(gray)

    # Step 2: Threshold on the CLAHE-equalized image.
    if threshold == 0:
        _, dark_mask = cv2.threshold(equalized, 0, 255,
                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, dark_mask = cv2.threshold(equalized, threshold, 255,
                                     cv2.THRESH_BINARY_INV)

    # Step 3: Morphological close to bridge small reflection gaps within a key.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
    closed = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

    # Step 4: Filter contours by width, aspect ratio, area, and position.
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw all contours before filtering (for debugging)
    all_contours_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(all_contours_img, contours, -1, (0, 255, 0), 1)

    mask = np.zeros_like(gray)
    filtered_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / w if w > 0 else 0
        area = cv2.contourArea(cnt)

        # Black keys: tall, narrow rectangles, wider than white-key gaps,
        # located in the upper portion of the image.
        if (aspect_ratio > min_aspect_ratio
                and area > min_area
                and w > min_width
                and y < h_img * max_y_frac):
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            filtered_contours.append(cnt)

    # Smooth mask edges
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Overlay: red highlight on original image
    overlay = img.copy()
    overlay[mask > 0] = (0, 0, 255)
    blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

    # Filtered contours drawn on the original image
    filtered_contours_img = img.copy()
    cv2.drawContours(filtered_contours_img, filtered_contours, -1, (0, 255, 0), 2)

    intermediates = {
        "1_grayscale": gray,
        "2_clahe": equalized,
        "3_dark_mask_threshold": dark_mask,
        "4_morphological_close": closed,
        "5_all_contours": all_contours_img,
        "6_filtered_contours": filtered_contours_img,
        "7_final_mask": mask,
        "8_overlay": blended,
    }

    return mask, intermediates


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Segment black keys from an overhead piano keyboard image.")
    parser.add_argument("image", nargs="?", default=str(Path(__file__).parent / "image.png"),
                        help="Path to the keyboard image (default: image.png next to this script).")
    parser.add_argument("--threshold", type=int, default=0,
                        help="Grayscale threshold (0 = Otsu auto).")
    parser.add_argument("--min-aspect-ratio", type=float, default=2.0)
    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--min-width", type=int, default=10)
    parser.add_argument("--max-y-frac", type=float, default=0.7)
    parser.add_argument("--clahe-clip", type=float, default=2.0,
                        help="CLAHE clip limit.")
    parser.add_argument("--clahe-grid", type=int, nargs=2, default=[8, 8],
                        help="CLAHE tile grid size (two ints).")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    mask, intermediates = segment_black_keys(
        img,
        threshold=args.threshold,
        min_aspect_ratio=args.min_aspect_ratio,
        min_area=args.min_area,
        min_width=args.min_width,
        max_y_frac=args.max_y_frac,
        clahe_clip=args.clahe_clip,
        clahe_grid=tuple(args.clahe_grid),
    )

    # Save all intermediate results into a results folder
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    for name, image in intermediates.items():
        out_path = results_dir / f"{name}.png"
        cv2.imwrite(str(out_path), image)
        print(f"Saved {out_path}")

    print(f"\nAll {len(intermediates)} intermediate results saved to {results_dir}")
