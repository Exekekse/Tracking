"""Utilities for computing and rendering body part boxes from pose keypoints."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import cv2
except Exception:  # pragma: no cover - cv2 may be missing in test env
    cv2 = None
import os
import time

from detection import Detection, PartBox


# Classes for export mapping
PART_CLASS_MAP = {"head": 0, "upper_body": 1, "body": 2}


@dataclass
class ImgShape:
    height: int
    width: int


def _clamp_box(box: Tuple[float, float, float, float], shape: ImgShape) -> PartBox:
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(shape.width - 1, x1)))
    x2 = int(max(0, min(shape.width - 1, x2)))
    y1 = int(max(0, min(shape.height - 1, y1)))
    y2 = int(max(0, min(shape.height - 1, y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def compute_parts_for_detection(det: Detection, img_shape: Tuple[int, int]) -> Optional[Dict[str, PartBox]]:
    """Compute head/upper/body boxes for a detection based on pose keypoints.

    The heuristics favour available keypoints but gracefully fall back to the
    detection bounding box if data is missing.
    """

    if not det.keypoints:
        return None

    h, w = img_shape[:2]
    shape = ImgShape(h, w)

    def pt(idx: int) -> Optional[Tuple[float, float]]:
        if idx >= len(det.keypoints):
            return None
        x, y, c = det.keypoints[idx]
        if c <= 0:
            return None
        return x, y

    head_pts = [pt(i) for i in range(5) if pt(i) is not None]
    shoulders = [pt(5), pt(6)]
    hips = [pt(11), pt(12)]

    # Shoulder line y
    shoulder_y_vals = [p[1] for p in shoulders if p is not None]
    if shoulder_y_vals:
        shoulder_y = median(shoulder_y_vals)
    else:
        shoulder_y = det.y1 + (det.y2 - det.y1) * 0.25

    # Hip line y
    hip_y_vals = [p[1] for p in hips if p is not None]
    if hip_y_vals:
        hip_y = median(hip_y_vals)
    else:
        hip_y = det.y1 + (det.y2 - det.y1) * 0.6

    # Head box
    head_x_vals = [p[0] for p in head_pts]
    if head_x_vals:
        head_cx = median(head_x_vals)
    elif shoulder_y_vals:
        sx_vals = [p[0] for p in shoulders if p is not None]
        head_cx = median(sx_vals)
    else:
        head_cx = det.cx

    if len(shoulders) == 2 and shoulders[0] and shoulders[1]:
        head_w = abs(shoulders[0][0] - shoulders[1][0])
    else:
        head_w = (det.x2 - det.x1) * 0.3
    head_w = max(20, head_w)

    head_top_vals = [p[1] for p in head_pts]
    head_top = min(head_top_vals) if head_top_vals else det.y1

    head_box = _clamp_box(
        (
            head_cx - head_w / 2,
            head_top,
            head_cx + head_w / 2,
            shoulder_y,
        ),
        shape,
    )

    # Upper body box
    side_x_vals = [p[0] for p in shoulders + hips if p is not None]
    if side_x_vals:
        x1u = min(side_x_vals)
        x2u = max(side_x_vals)
    else:
        x1u, x2u = det.x1, det.x2
    pad = (det.x2 - det.x1) * 0.05
    upper_box = _clamp_box((x1u - pad, shoulder_y, x2u + pad, hip_y), shape)

    # Body box
    all_pts = [p for p in [pt(i) for i in range(len(det.keypoints))] if p is not None]
    if all_pts:
        x1b = min(p[0] for p in all_pts)
        y1b = min(p[1] for p in all_pts)
        x2b = max(p[0] for p in all_pts)
        y2b = max(p[1] for p in all_pts)
        body_box = _clamp_box((x1b, y1b, x2b, y2b), shape)
    else:
        body_box = _clamp_box((det.x1, det.y1, det.x2, det.y2), shape)

    return {"head": head_box, "upper_body": upper_box, "body": body_box}


def draw_parts_on_image(img, parts: Dict[str, PartBox]) -> None:
    """Draw semi transparent overlays for computed part boxes."""

    colors = {
        "head": (0, 255, 255),  # yellow
        "upper_body": (255, 0, 0),  # blue
        "body": (0, 255, 0),  # green
    }
    overlay = img.copy()
    for name, box in parts.items():
        x1, y1, x2, y2 = box
        if cv2 is not None:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), colors.get(name, (255, 255, 255)), -1)
    if cv2 is not None:
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, dst=img)


def rescale_keypoints(kpts: List[Tuple[float, float, float]], sx: float, sy: float) -> List[Tuple[float, float, float]]:
    """Rescale keypoints with separate x/y scale factors."""

    scaled = []
    for x, y, c in kpts:
        scaled.append((x * sx, y * sy, c))
    return scaled


def export_parts_for_training(frame_bgr, detections: List[Detection], out_dir: str) -> Optional[str]:
    """Export frame and part annotations for training in YOLO format."""

    if not any(d.parts for d in detections):
        print("[WARN] Keine Keypoints – Export abgebrochen.")
        return None

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join(out_dir, f"cap_{ts}.jpg")
    txt_path = os.path.join(out_dir, f"cap_{ts}.txt")

    if cv2 is not None:
        cv2.imwrite(img_path, frame_bgr)

    h, w = frame_bgr.shape[:2]
    with open(txt_path, "w", encoding="utf-8") as f:
        for d in detections:
            if not d.parts:
                continue
            for name, (x1, y1, x2, y2) in d.parts.items():
                cls_id = PART_CLASS_MAP[name]
                cx = (x1 + x2) / 2.0 / w
                cy = (y1 + y2) / 2.0 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"[INFO] Parts exportiert: {img_path} + {txt_path}")
    return txt_path
