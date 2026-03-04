import cv2
import numpy as np

# === INPUT / OUTPUT PATHS — USER EDITS HERE ===
input_video = "input path"
output_video = "output path"

# === LOAD CALIBRATION FRAME ===
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {input_video}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)  # last frame (clean)
ret, frame = cap.read()
if not ret:
    raise IOError("Could not read calibration frame.")
cap.release()

disp = frame.copy()


# ================================
# STEP 1 — USER SELECTS 4 CORNERS
# ================================
print("Click the 4 approximate keyboard corners:")
print("1) top-left, 2) top-right, 3) bottom-right, 4) bottom-left")

clicked = []

def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked.append((x, y))
        cv2.circle(disp, (x, y), 8, (0,255,0), -1)
        cv2.imshow("Select 4 corners", disp)

cv2.imshow("Select 4 corners", disp)
cv2.setMouseCallback("Select 4 corners", mouse_click)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(clicked) != 4:
    raise ValueError("You must click exactly 4 points.")

clicked = np.float32(clicked)
print("User clicks:", clicked)


# ========================================
# STEP 2 — AUTOMATIC CORNER REFINEMENT
# ========================================
# Canny edges + Hough lines
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 80, 200)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                        minLineLength=200, maxLineGap=30)

if lines is None:
    raise RuntimeError("No strong lines detected; refine the calibration frame.")


def line_to_params(line):
    """ Convert line segment to ax + by + c = 0 """
    x1, y1, x2, y2 = line
    a = y1 - y2
    b = x2 - x1
    c = x1*y2 - x2*y1
    return (a, b, c)

def intersect(L1, L2):
    """ Solve intersection of two lines in ax+by+c = 0 form """
    a1, b1, c1 = L1
    a2, b2, c2 = L2
    d = a1*b2 - a2*b1
    if abs(d) < 1e-5:
        return None
    x = (b1*c2 - b2*c1) / d
    y = (c1*a2 - c2*a1) / d
    return np.array([x, y], dtype=np.float32)


def refine_corner(rough_point):
    rx, ry = rough_point
    # Find lines whose endpoints are within 150 px of the click
    close_lines = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        d = min(np.hypot(rx-x1, ry-y1), np.hypot(rx-x2, ry-y2))
        if d < 150:  # adjust tolerance if needed
            close_lines.append((x1, y1, x2, y2))

    # Need at least two lines to intersect
    if len(close_lines) < 2:
        return rough_point

    # Convert to ax+by+c=0 form
    params = [line_to_params(l) for l in close_lines]

    # Try all intersections and pick the one closest to the rough click
    best = None
    best_dist = 999999

    for i in range(len(params)):
        for j in range(i+1, len(params)):
            p = intersect(params[i], params[j])
            if p is None:
                continue
            dist = np.hypot(p[0]-rx, p[1]-ry)
            if dist < best_dist:
                best_dist = dist
                best = p

    if best is None:
        return rough_point
    return best


refined = np.float32([refine_corner(pt) for pt in clicked])
print("Refined corners:\n", refined)


# ========================================
# STEP 3 — WARPING SETUP
# ========================================
# Choose output size (adjust if needed)
out_w, out_h = 1920, 260

dst_pts = np.float32([
    [0, 0],
    [out_w, 0],
    [out_w, out_h],
    [0, out_h]
])

M = cv2.getPerspectiveTransform(refined, dst_pts)


# ========================================
# STEP 4 — PROCESS ENTIRE VIDEO
# ========================================
cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (out_w, out_h))

frame_i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    warped = cv2.warpPerspective(frame, M, (out_w, out_h))
    out.write(warped)

    frame_i += 1
    if frame_i % 400 == 0:
        print(f"{frame_i}/{total_frames} processed...")

cap.release()
out.release()
print("Done! Saved:", output_video)
