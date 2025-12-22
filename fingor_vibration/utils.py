


import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import ffmpeg

def get_video_info(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"FPS: {fps}, Frame count: {frame_count}, Width: {width}, Height: {height}")
    return fps, frame_count, width, height



def show_frame_with_bbox(video_path: str, frame_idx: int, x: int, y: int, w: int, h: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Jump to the desired frame index
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    ok, frame_bgr = cap.read()
    cap.release()

    if not ok or frame_bgr is None:
        raise RuntimeError(f"Could not read frame {frame_idx} (out of range?)")

    # Convert BGR (OpenCV) -> RGB (matplotlib)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    H, W = frame_rgb.shape[:2]

    # Optional: clamp bbox to image bounds
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)
    w_clamped = max(0, x1 - x0)
    h_clamped = max(0, y1 - y0)

    fig, ax = plt.subplots()
    ax.imshow(frame_rgb)
    rect = patches.Rectangle((x0, y0), w_clamped, h_clamped, linewidth=2, edgecolor="r", facecolor="none")
    ax.add_patch(rect)
    ax.set_title(f"Frame {frame_idx} with bbox (x={x}, y={y}, w={w}, h={h})")
    ax.axis("off")
    plt.show()


def crop_mp4_ffmpeg_python(in_path, out_path, start_sec, end_sec, x, y, w, h):
    duration = max(0.0, float(end_sec) - float(start_sec))
    (
        ffmpeg
        .input(in_path, ss=start_sec, t=duration)
        .filter("crop", w, h, x, y)
        .output(out_path, vcodec="libx264", acodec="aac", crf=18, preset="medium")
        .overwrite_output()
        .run()
    )



if __name__ == "__main__":

    video_folder = r"C:\Users\haowe\OneDrive\Desktop\MIT\PianoProject\Code\DUSTrack\pia02_workflow\fingor_torque"
    video_name = "s061.MP4"
    video_path = os.path.join(video_folder, video_name)
    out_path = os.path.join(video_folder, f"cropped_{video_name}")

    x = 1300
    y = 100
    w = 600
    h = 600

    start_sec = 45
    end_sec = 2*60+45

    get_video_info(video_path)
    show_frame_with_bbox(video_path, frame_idx=0, x=x, y=y, w=w, h=h)
    crop_mp4_ffmpeg_python(video_path, out_path, start_sec, end_sec, x, y, w, h)
    get_video_info(out_path)











# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     raise RuntimeError(f"Failed to open: {video_path}")

# fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
# duration_s = (frame_count / fps) if fps > 0 else float("nan")

# print(f"Frames: {frame_count}")
# print(f"FPS: {fps:.10f}")
# print(f"Duration (s): {duration_s:.10f}")
# print(f"Resolution: {width}x{height}")

# cap.release()



# import ffmpeg

# def crop_mp4_ffmpeg_python(in_path, out_path, start_sec, end_sec, x, y, w, h):
#     duration = max(0.0, float(end_sec) - float(start_sec))
#     (
#         ffmpeg
#         .input(in_path, ss=start_sec, t=duration)
#         .filter("crop", w, h, x, y)
#         .output(out_path, vcodec="libx264", acodec="aac", crf=18, preset="medium")
#         .overwrite_output()
#         .run()
#     )

# crop_mp4_ffmpeg_python(
#     "input.mp4", "output.mp4",
#     start_sec=2.5, end_sec=10.0,
#     x=100, y=50, w=640, h=480
# )
