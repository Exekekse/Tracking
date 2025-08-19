import os
import math
import cv2
import numpy as np
import mss
from ultralytics import YOLO
import torch  # noqa: F401 - ensures torch is available

MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
EMA_ALPHA = float(os.getenv("EMA_ALPHA", 0.3))
DEV_OVERLAY = os.getenv("DEV_OVERLAY", "0") == "1"
DRIFT_CHECK_INTERVAL = int(os.getenv("DRIFT_CHECK_INTERVAL", 60))
DETECTION_DOWNSCALE = float(os.getenv("DETECTION_DOWNSCALE", 1.0))

# Determine if a specialised head model exists. If not, restrict to class 0.
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "yolov8n.pt"
    DETECTION_CLASSES = [0]
else:
    DETECTION_CLASSES = None

model = YOLO(MODEL_PATH)
model.fuse()


TRACKER_CANDIDATES = [
    ("legacy", "TrackerMOSSE_create"),
    (None, "TrackerMOSSE_create"),
    ("legacy", "TrackerKCF_create"),
    (None, "TrackerKCF_create"),
    ("legacy", "TrackerCSRT_create"),
    (None, "TrackerCSRT_create"),
    ("legacy", "TrackerMIL_create"),
    (None, "TrackerMIL_create"),
]


def init_tracker(frame, bbox):
    """Try to initialise a tracker using the fallback chain."""
    for module, name in TRACKER_CANDIDATES:
        mod = getattr(cv2, module, cv2) if module else cv2
        creator = getattr(mod, name, None)
        if not callable(creator):
            continue
        try:
            tracker = creator()
            tracker.init(frame, bbox)
            return tracker
        except Exception:
            continue
    raise RuntimeError(
        "No suitable OpenCV tracker available. Install opencv-contrib-python."
    )


def validate_box(box, frame_shape):
    """Validate and clamp box to frame bounds. Return tuple or None."""
    if box is None:
        return None
    try:
        x, y, w, h = [float(v) for v in box]
    except Exception:
        return None
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        return None
    height, width = frame_shape[:2]
    x = min(max(0.0, x), width - 1.0)
    y = min(max(0.0, y), height - 1.0)
    w = min(max(0.0, w), width - x)
    h = min(max(0.0, h), height - y)
    if w < 1.0 or h < 1.0:
        return None
    return (x, y, w, h)


def detect_head(frame):
    """Run YOLO detection and return the best bounding box (x, y, w, h)."""
    scale = DETECTION_DOWNSCALE if DETECTION_DOWNSCALE < 1.0 else 1.0
    inp = frame
    if scale != 1.0:
        new_size = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
        inp = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
    results = model(inp, verbose=False, classes=DETECTION_CLASSES)[0]
    if not results.boxes:
        return None
    boxes = results.boxes.xyxy.cpu().numpy() / scale
    conf = results.boxes.conf.cpu().numpy()
    idx = conf.argmax()
    x1, y1, x2, y2 = boxes[idx]
    raw = (x1, y1, x2 - x1, y2 - y1)
    return validate_box(raw, frame.shape)


def iou(box_a, box_b):
    xa, ya, wa, ha = box_a
    xb, yb, wb, hb = box_b
    x1 = max(xa, xb)
    y1 = max(ya, yb)
    x2 = min(xa + wa, xb + wb)
    y2 = min(ya + ha, yb + hb)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = wa * ha + wb * hb - inter
    return inter / union if union > 0 else 0.0


def ema(box, prev, alpha=EMA_ALPHA):
    arr = np.array(box, dtype=np.float32)
    if prev is None:
        return arr
    return alpha * arr + (1 - alpha) * prev


def main():
    tracker = None
    smooth_box = None
    frame_count = 0

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        while True:
            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            frame = np.ascontiguousarray(frame, dtype=np.uint8)

            if tracker is None:
                det = detect_head(frame)
                if det is not None:
                    try:
                        tracker = init_tracker(frame, det)
                        smooth_box = np.array(det, dtype=np.float32)
                    except Exception:
                        tracker = None
                        smooth_box = None
            else:
                success, box = tracker.update(frame)
                if not success:
                    tracker = None
                    smooth_box = None
                else:
                    box = validate_box(box, frame.shape)
                    if box is None:
                        tracker = None
                        smooth_box = None
                    else:
                        smooth_box = ema(box, smooth_box)

                        if frame_count % DRIFT_CHECK_INTERVAL == 0:
                            det = detect_head(frame)
                            if det is not None and iou(box, det) < 0.5:
                                try:
                                    tracker = init_tracker(frame, det)
                                    smooth_box = np.array(det, dtype=np.float32)
                                except Exception:
                                    tracker = None
                                    smooth_box = None

            if DEV_OVERLAY and smooth_box is not None:
                x, y, w, h = map(int, smooth_box)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
