import cv2
import lensfunpy

# ----------------------------
# 1. Load Lensfun database
# ----------------------------
db = lensfunpy.Database()

# FX30 is not in Lensfun, so use the closest match: Sony A6400 (same sensor)
cam = db.find_cameras("Sony", "ILCE-6400")[0]

# Sigma 16mm f/1.4 DC DN (Sony E)
lens = db.find_lenses(cam, "Sigma", "16mm F1.4 DC DN")[0]

# ----------------------------
# 2. Distortion parameters
# ----------------------------
focal_length = 16.0   # mm
aperture = 1.4
distance = 0          # distance has no effect on geometric distortion

# ----------------------------
# 3. Input/output videos — USER PUTS PATHS HERE
# ----------------------------
input_video = r"\\192.168.1.104\home\piano\data\overhead_camera\lens_correction_test\fx30_2_0894.MP4"
output_video = r"\\192.168.1.104\home\piano\data\overhead_camera\lens_correction_test\fx30_2_0894_corrected.mp4"

cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ----------------------------
# 4. Precompute lensfun correction maps
# ----------------------------
mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
mod.initialize(focal_length, aperture, distance)

coords = mod.apply_geometry_distortion()

# OpenCV requires float32 maps split into x/y
mapx = coords[:, :, 0].astype("float32")
mapy = coords[:, :, 1].astype("float32")

# ----------------------------
# 5. Create output video writer
# ----------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# ----------------------------
# 6. Process video frame-by-frame
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply precomputed distortion correction
    corrected = cv2.remap(frame, mapx, mapy, cv2.INTER_LANCZOS4)

    out.write(corrected)

cap.release()
out.release()

print("Saved undistorted video:", output_video)