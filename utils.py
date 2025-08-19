import cv2
import os
import json
import numpy as np
import pathlib
import tempfile
import time
import logging
import logging.handlers
import io


def _get_attr(mod, name):
    return getattr(mod, name) if hasattr(mod, name) else None


def create_tracker() -> cv2.Tracker:
    """Erzeuge einen OpenCV-Tracker mit robusten Fallbacks (benötigt opencv-contrib-python)."""
    candidates = [
        (cv2, "TrackerMOSSE_create"),
        (getattr(cv2, "legacy", cv2), "TrackerMOSSE_create"),
        (cv2, "TrackerKCF_create"),
        (getattr(cv2, "legacy", cv2), "TrackerKCF_create"),
        (cv2, "TrackerCSRT_create"),
        (getattr(cv2, "legacy", cv2), "TrackerCSRT_create"),
        (cv2, "TrackerMIL_create"),
        (getattr(cv2, "legacy", cv2), "TrackerMIL_create"),
    ]
    for mod, name in candidates:
        ctor = _get_attr(mod, name)
        if callable(ctor):
            try:
                return ctor()
            except Exception:
                continue
    raise RuntimeError(
        "Kein geeigneter OpenCV-Tracker verfügbar. Installiere 'opencv-contrib-python' (nicht headless, wenn Overlay)."
    )


def xyxy_to_xywh(x1, y1, x2, y2):
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)


def xywh_to_xyxy(x, y, w, h):
    return float(x), float(y), float(x + w), float(y + h)


def clamp_box_xywh(x, y, w, h, width, height):
    x = min(max(0.0, x), max(0.0, width - 1.0))
    y = min(max(0.0, y), max(0.0, height - 1.0))
    w = max(1.0, min(w, width - x))
    h = max(1.0, min(h, height - y))
    return x, y, w, h


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return float(inter / (area_a + area_b - inter))


def ema(prev_xywh, new_xywh, alpha):
    if prev_xywh is None:
        return new_xywh
    px, py, pw, ph = prev_xywh
    nx, ny, nw, nh = new_xywh
    return (
        alpha * nx + (1 - alpha) * px,
        alpha * ny + (1 - alpha) * py,
        alpha * nw + (1 - alpha) * pw,
        alpha * nh + (1 - alpha) * ph,
    )


# --- Persistenz & Hilfsfunktionen -------------------------------------------------


def _atomic_write(path: pathlib.Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def read_json(path: pathlib.Path, default):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path: pathlib.Path, obj) -> None:
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    _atomic_write(path, data)


def get_storage_paths():
    base = os.environ.get("LOCALAPPDATA")
    if base:
        root = pathlib.Path(base) / "TrackingAI" / "Valorant"
    else:
        root = pathlib.Path.home() / ".local" / "share" / "TrackingAI" / "Valorant"
    paths = {
        "root": root,
        "config": root / "config",
        "data": root / "data",
        "logs": root / "logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def load_calibration(paths, defaults):
    cfg_path = paths["config"] / "calibration.json"
    cfg = read_json(cfg_path, {})
    merged = {**defaults, **cfg}
    return merged, cfg_path


def save_calibration(cfg_path, cfg):
    cfg["updated_utc"] = int(time.time())
    write_json(cfg_path, cfg)


def load_heatmap(path: pathlib.Path, shape):
    try:
        arr = np.load(str(path))
        if arr.shape == shape:
            return arr
    except Exception:
        pass
    return np.zeros(shape, dtype=np.float32)


def save_heatmap(path: pathlib.Path, arr):
    data = io.BytesIO()
    np.save(data, arr)
    _atomic_write(path, data.getvalue())


def enforce_size_limit(paths, limit_bytes):
    files = []
    for folder in (paths["data"], paths["logs"]):
        for p in folder.glob("**/*"):
            if p.is_file():
                files.append(p)
    total = sum(p.stat().st_size for p in files)
    if total <= limit_bytes:
        return
    files.sort(key=lambda p: p.stat().st_mtime)
    while files and total > limit_bytes:
        p = files.pop(0)
        try:
            size = p.stat().st_size
            p.unlink()
            total -= size
        except Exception:
            break


class Heatmap:
    def __init__(self, shape):
        self.grid = np.zeros(shape, dtype=np.float32)
        self.shape = shape
        self.counter = 0

    def accumulate(self, cx, cy, w, h):
        x = int(np.clip(cx / w * self.shape[1], 0, self.shape[1] - 1))
        y = int(np.clip(cy / h * self.shape[0], 0, self.shape[0] - 1))
        self.grid[y, x] += 1.0
        self.counter += 1

    def build_mask(self, percentile=0.6, min_y_ratio=0.6):
        if self.counter == 0:
            return np.zeros(self.shape, dtype=np.uint8)
        thr = np.max(self.grid) * percentile
        mask = (self.grid >= thr).astype(np.uint8)
        cutoff = int(self.shape[0] * min_y_ratio)
        mask[:cutoff, :] = 0
        return mask

    def reset(self):
        self.grid.fill(0.0)
        self.counter = 0


def candidate_valid(box, frame_w, frame_h, ar_range, ignore_mask, area_ratio, viewmodel_y):
    if box is None:
        return False
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return False
    area = w * h
    if area > area_ratio * frame_w * frame_h:
        return False
    cx = x + w * 0.5
    cy = y + h * 0.5
    if cy > viewmodel_y * frame_h:
        return False
    ar = w / h
    if not (ar_range[0] <= ar <= ar_range[1]):
        return False
    if ignore_mask is not None:
        mh, mw = ignore_mask.shape
        mx = int(cx / frame_w * mw)
        my = int(cy / frame_h * mh)
        if 0 <= mx < mw and 0 <= my < mh and ignore_mask[my, mx] > 0:
            return False
    return True


def head_above_body(head, body):
    if head is None or body is None:
        return False
    hx, hy, hw, hh = head
    bx, by, bw, bh = body
    return hy + hh <= by


def setup_logger(paths):
    log_file = paths["logs"] / "session.log"
    handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=256 * 1024, backupCount=3, encoding="utf-8"
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    return logging.getLogger("tracker")
