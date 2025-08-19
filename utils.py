import cv2


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
