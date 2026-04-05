import os
import re
import math
import cv2
import numpy as np

# COCO skeleton pairs (0-based index)
COCO_EDGES = [
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
    (5, 6),              # shoulders
    (11, 12),            # hips
    (5, 11), (6, 12),    # torso sides
    (0, 1), (0, 2),      # nose to eyes
    (1, 3), (2, 4),      # eyes to ears
]

def draw_pose(frame, keypoints, edges=COCO_EDGES, conf_threshold=0.1):
    """
    keypoints shape:
      - (J, 2) for x,y
      - (J, 3) for x,y,conf
    """
    kpts = keypoints.copy()

    # draw joints
    for i, pt in enumerate(kpts):
        if len(pt) == 2:
            x, y = pt
            conf = 1.0
        else:
            x, y, conf = pt

        if conf >= conf_threshold:
            # Check NaN or inf
            if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
                continue
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

    # draw bones
    for a, b in edges:
        if a >= len(kpts) or b >= len(kpts):
            continue

        pa = kpts[a]
        pb = kpts[b]

        if len(pa) == 2:
            xa, ya = pa
            ca = 1.0
        else:
            xa, ya, ca = pa

        if len(pb) == 2:
            xb, yb = pb
            cb = 1.0
        else:
            xb, yb, cb = pb

        if ca >= conf_threshold and cb >= conf_threshold:
            if (np.isnan(xa) or np.isnan(ya) or np.isinf(xa) or np.isinf(ya) or
                np.isnan(xb) or np.isnan(yb) or np.isinf(xb) or np.isinf(yb)):
                continue
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (255, 0, 0), 2)

    return frame


def make_grid(images, cols=None):
    n = len(images)
    if cols is None:
        cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    h, w = images[0].shape[:2]
    blank = np.zeros_like(images[0])

    padded = images + [blank] * (rows * cols - n)
    row_imgs = []

    for r in range(rows):
        row = padded[r * cols:(r + 1) * cols]
        row_imgs.append(np.hstack(row))

    return np.vstack(row_imgs)


def load_axel_views(root, split, subject, sequence, pose_type="coco"):
    """
    Example:
    root/data/train_set/S1/Axel_1_cam_1.mp4
    root/data/train_set/S1/Axel_1_cam_1_coco.npy
    """
    folder = os.path.join(root, "data", split, subject)
    pattern = re.compile(rf"^{re.escape(sequence)}_cam_(\d+)\.mp4$")

    views = []
    for fname in sorted(os.listdir(folder)):
        m = pattern.match(fname)
        if not m:
            continue

        cam_id = m.group(1)
        base = f"{sequence}_cam_{cam_id}"
        video_path = os.path.join(folder, f"{base}.mp4")

        if pose_type == "coco":
            pose_path = os.path.join(folder, f"{base}_coco.npy")
        elif pose_type == "h36m":
            pose_path = os.path.join(folder, f"{base}_h36m.npy")
        else:
            pose_path = os.path.join(folder, f"{base}.npy")

        if not os.path.exists(pose_path):
            print(f"Missing pose file for {base}: {pose_path}")
            continue

        poses = np.load(pose_path)
        views.append({
            "cam": f"cam_{cam_id}",
            "video_path": video_path,
            "pose_path": pose_path,
            "poses": poses,
        })

    return views


def show_all_views_with_annotation(
    root,
    split="train_set",
    subject="S1",
    sequence="Axel_1",
    pose_type="coco",
    output_path=None,
    resize_to=(480, 270),
    show_window=True
):
    views = load_axel_views(root, split, subject, sequence, pose_type=pose_type)
    if not views:
        raise ValueError("No views found.")

    # open all videos
    caps = []
    min_frames = float("inf")
    fps_list = []

    for v in views:
        cap = cv2.VideoCapture(v["video_path"])
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {v['video_path']}")

        caps.append(cap)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps_list.append(fps)

        pose_frames = len(v["poses"])
        min_frames = min(min_frames, frame_count, pose_frames)

    fps_out = fps_list[0] if fps_list else 25

    writer = None
    if output_path is not None:
        sample_n = len(views)
        cols = math.ceil(math.sqrt(sample_n))
        rows = math.ceil(sample_n / cols)
        out_w = resize_to[0] * cols
        out_h = resize_to[1] * rows
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_out, (out_w, out_h))

    for t in range(min_frames):
        frames_for_grid = []

        for cap, v in zip(caps, views):
            ret, frame = cap.read()
            if not ret:
                break

            pose = v["poses"][t]
            frame = draw_pose(frame, pose)

            cv2.putText(
                frame,
                f"{sequence} | {v['cam']} | frame {t}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            frame = cv2.resize(frame, resize_to)
            frames_for_grid.append(frame)

        if len(frames_for_grid) != len(views):
            break

        grid = make_grid(frames_for_grid)

        if show_window:
            cv2.imshow("Multiview Axel", grid)
            key = cv2.waitKey(int(1000 / fps_out)) & 0xFF
            if key == 27 or key == ord('q'):
                break

        if writer is not None:
            writer.write(grid)

    for cap in caps:
        cap.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_all_views_with_annotation(
        root="AthletePose3D",
        split="train_set",
        subject="S2",
        sequence="Axel_2",
        pose_type="coco",
        output_path="Axel_2_multiview.mp4",
        show_window=True
    )