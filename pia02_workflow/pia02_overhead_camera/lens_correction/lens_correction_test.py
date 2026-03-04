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

# Input / output paths
INPUT_VIDEO = r"\\192.168.1.104\home\piano\data\overhead_camera\lens_correction_test\fx30_2_0894.MP4"
OUTPUT_VIDEO = r"\\192.168.1.104\home\piano\data\overhead_camera\lens_correction_test\fx30_2_0894_corrected.mp4"


def build_distortion_maps(width, height):
    """Look up the camera/lens in Lensfun and return distortion remap coordinates.

    Returns:
        ndarray of shape (height, width, 2) with float32 remap coordinates.
    """
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


def correct_video(input_path, output_path):
    """Apply lens distortion correction to every frame and write the result."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    coords = build_distortion_maps(width, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    last_original = None
    last_corrected = None

    for _ in tqdm(range(total_frames), desc="Correcting"):
        ret, frame = cap.read()
        if not ret:
            break
        corrected = cv2.remap(frame, coords, None, cv2.INTER_LANCZOS4)
        out.write(corrected)
        last_original = frame
        last_corrected = corrected

    cap.release()
    out.release()
    print(f"Saved undistorted video: {output_path}")

    # Save last frame of both input and output as images
    out_dir = Path(output_path).parent
    stem = Path(output_path).stem

    if last_original is not None:
        orig_img = out_dir / f"{stem}_last_frame_original.png"
        cv2.imwrite(str(orig_img), last_original)
        print(f"Saved last original frame: {orig_img}")

    if last_corrected is not None:
        corr_img = out_dir / f"{stem}_last_frame_corrected.png"
        cv2.imwrite(str(corr_img), last_corrected)
        print(f"Saved last corrected frame: {corr_img}")


if __name__ == "__main__":
    correct_video(INPUT_VIDEO, OUTPUT_VIDEO)
