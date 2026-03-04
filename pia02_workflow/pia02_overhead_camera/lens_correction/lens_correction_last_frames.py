from pathlib import Path

import cv2
import numpy as np
import lensfunpy
from tqdm import tqdm

# Camera / lens constants
CAMERA_MAKER = "Sony"
CAMERA_MODEL = "ILCE-6400"  # proxy for FX30 (same APS-C 1.5x crop)
LENS_MAKER = "Sigma"
LENS_MODEL = "16mm F1.4 DC DN"
FOCAL_LENGTH = 16.0  # mm
APERTURE = 1.4
DISTANCE = 1000.0  # metres (infinity; irrelevant for geometry distortion)

INPUT_DIR = Path(r"\\192.168.1.104\home\piano\data\overhead_camera\last_frames")

# check if the input directory exists
if not INPUT_DIR.exists():
    print(f"Input directory {INPUT_DIR} does not exist")
    exit()


def build_distortion_maps(width, height):
    """Look up the camera/lens in Lensfun and return distortion remap coordinates."""
    db = lensfunpy.Database()
    cam = db.find_cameras(CAMERA_MAKER, CAMERA_MODEL)[0]
    lens = db.find_lenses(cam, LENS_MAKER, LENS_MODEL)[0]

    mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
    mod.initialize(FOCAL_LENGTH, APERTURE, DISTANCE, pixel_format=np.uint8)

    coords = mod.apply_geometry_distortion()
    if coords is None:
        raise RuntimeError(
            f"No distortion calibration data for {LENS_MAKER} {LENS_MODEL} "
            f"on {CAMERA_MAKER} {CAMERA_MODEL}."
        )
    return coords


def correct_frame(frame, coords):
    """Apply lens distortion correction to a single frame."""
    return cv2.remap(frame, coords, None, cv2.INTER_LANCZOS4)


def correct_images(input_dir):
    """Correct all images in a directory and save as <name>-corrected.png."""
    input_dir = Path(input_dir)
    images = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    )

    if not images:
        print(f"No images found in {input_dir}")
        return

    # Build maps once from the first image's dimensions
    first = cv2.imread(str(images[0]))
    if first is None:
        raise FileNotFoundError(f"Cannot read image: {images[0]}")
    h, w = first.shape[:2]
    coords = build_distortion_maps(w, h)

    for img_path in tqdm(images, desc="Correcting"):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Skipping unreadable file: {img_path}")
            continue
        corrected = correct_frame(frame, coords)
        out_path = img_path.parent / f"{img_path.stem}-lens_corrected.png"
        cv2.imwrite(str(out_path), corrected)

    print(f"Done. Corrected {len(images)} images in {input_dir}")


if __name__ == "__main__":
    correct_images(INPUT_DIR)
