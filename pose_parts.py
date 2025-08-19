"""Utilities to compute human body part regions from pose keypoints.

This module provides helper functions to derive head / upper body /
body boxes from YOLOv8 pose keypoints and to render or export these
regions.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np

from detection import Detection

PART_COLORS = {
    "head": (0, 255, 255),      # yellow
    "upper_body": (255, 0, 0),  # blue
    "body": (0, 255, 0),        # green
}


def safe_mkdir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def rescale_keypoints(kpts: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """Rescale keypoints from resized inference back to original resolution."""
    out = np.asarray(kpts, dtype=float).copy()
    out[..., 0] *= scale_x
    out[..., 1] *= scale_y
    return out


def _clamp_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    return x1, y1, x2, y2


def compute_parts_for_detection(d: Detection, img_shape: Tuple[int, int]) -> Dict[str, Tuple[int, int, int, int]] | None:
    """Compute head/upper/body boxes for a detection.

    Returns ``None`` if the detection lacks keypoints.
    """

    if not d.keypoints:
        return None

    kpts = np.asarray(d.keypoints, dtype=float)
    if kpts.shape[0] < 17:
        # not enough keypoints – give up
        return None

    conf = kpts[:, 2]
    valid = conf > 0.2
    if not np.any(valid):
        return None

    h, w = img_shape

    # Body box from keypoint extrema
    k_valid = kpts[valid]
    body_x1, body_y1 = k_valid[:, 0].min(), k_valid[:, 1].min()
    body_x2, body_y2 = k_valid[:, 0].max(), k_valid[:, 1].max()
    body = _clamp_box(body_x1, body_y1, body_x2, body_y2, w, h)

    # Shoulders
    ls = kpts[5] if conf[5] > 0.2 else None
    rs = kpts[6] if conf[6] > 0.2 else None
    shoulder_y = (
        np.median([p[1] for p in [ls, rs] if p is not None])
        if (ls is not None or rs is not None)
        else d.y1 + d.wh[1] * 0.25
    )
    if ls is not None and rs is not None:
        shoulder_dist = abs(ls[0] - rs[0])
        head_cx = np.median([ls[0], rs[0]])
    else:
        shoulder_dist = d.wh[0] * 0.5
        head_cx = d.cx

    # Hips
    lh = kpts[11] if conf[11] > 0.2 else None
    rh = kpts[12] if conf[12] > 0.2 else None
    hip_y = (
        np.median([p[1] for p in [lh, rh] if p is not None])
        if (lh is not None or rh is not None)
        else d.y1 + d.wh[1] * 0.75
    )

    x_candidates = [p[0] for p in [ls, rs, lh, rh] if p is not None]
    if x_candidates:
        x_left, x_right = min(x_candidates), max(x_candidates)
    else:
        x_left, x_right = d.x1, d.x2
    padding = (x_right - x_left) * 0.1

    upper_body = _clamp_box(x_left - padding, shoulder_y, x_right + padding, hip_y, w, h)

    head_pts = [kpts[i] for i in (0, 1, 2, 3, 4) if conf[i] > 0.2]
    head_top = min((p[1] for p in head_pts), default=d.y1)
    head_w = max(shoulder_dist, 20)
    head_y1 = head_top - 0.2 * head_w
    head = _clamp_box(head_cx - head_w / 2, head_y1, head_cx + head_w / 2, shoulder_y, w, h)

    parts = {
        "head": head,
        "upper_body": upper_body,
        "body": body,
    }
    return parts


def draw_parts_on_image(img: np.ndarray, parts: Dict[str, Tuple[int, int, int, int]]) -> None:
    """Draw semi transparent rectangles for the given parts on ``img``."""

    overlay = img.copy()
    for name, (x1, y1, x2, y2) in parts.items():
        color = PART_COLORS.get(name, (255, 255, 255))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)


def export_parts_labels(frame_bgr: np.ndarray, parts_list: List[Dict[str, Tuple[int, int, int, int]]], out_dir: str) -> None:
    """Export frame + YOLO label file for body parts."""

    valid = [p for p in parts_list if p]
    if not valid:
        print("[WARN] Keine Keypoints vorhanden – Export abgebrochen.")
        return

    safe_mkdir(out_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join(out_dir, f"cap_{ts}.jpg")
    txt_path = os.path.join(out_dir, f"cap_{ts}.txt")

    cv2.imwrite(img_path, frame_bgr)

    h, w = frame_bgr.shape[:2]
    lines = 0
    with open(txt_path, "w", encoding="utf-8") as f:
        for parts in valid:
            for cls, key in enumerate(["head", "upper_body", "body"]):
                if key in parts:
                    x1, y1, x2, y2 = parts[key]
                    cx = (x1 + x2) / 2.0 / w
                    cy = (y1 + y2) / 2.0 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    lines += 1
    print(f"[INFO] Export parts: {img_path} + {txt_path} ({lines} boxes)")
