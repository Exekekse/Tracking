import os
import cv2
import numpy as np
import mss
from ultralytics import YOLO
import torch  # noqa: F401 - ensures torch is available

MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
EMA_ALPHA = float(os.getenv("EMA_ALPHA", 0.3))
DEV_OVERLAY = os.getenv("DEV_OVERLAY", "0") == "1"
DRIFT_CHECK_INTERVAL = int(os.getenv("DRIFT_CHECK_INTERVAL", 60))

# Determine if a specialised head model exists. If not, restrict to class 0.
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "yolov8n.pt"
    DETECTION_CLASSES = [0]
else:
    DETECTION_CLASSES = None

model = YOLO(MODEL_PATH)
model.fuse()


def create_tracker():
    """Create a MOSSE tracker, falling back to KCF if unavailable."""
    candidates = [
        ("legacy", "TrackerMOSSE_create"),
        (None, "TrackerMOSSE_create"),
        ("legacy", "TrackerKCF_create"),
        (None, "TrackerKCF_create"),
    ]
    for module, name in candidates:
        mod = getattr(cv2, module, cv2) if module else cv2
        creator = getattr(mod, name, None)
        if callable(creator):
            return creator()
    raise RuntimeError("No suitable OpenCV tracker available")


def detect_head(frame):
    """Run YOLO detection and return the best bounding box (x, y, w, h)."""
    results = model(frame, verbose=False, classes=DETECTION_CLASSES)[0]
    if not results.boxes:
        return None
    boxes = results.boxes.xyxy.cpu().numpy()
    conf = results.boxes.conf.cpu().numpy()
    idx = conf.argmax()
    x1, y1, x2, y2 = boxes[idx]
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


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

            if tracker is None:
                det = detect_head(frame)
                if det is not None:
                    tracker = create_tracker()
                    tracker.init(frame, tuple(det))
                    smooth_box = np.array(det, dtype=np.float32)
            else:
                success, box = tracker.update(frame)
                if not success:
                    tracker = None
                    smooth_box = None
                else:
                    box = list(box)
                    smooth_box = ema(box, smooth_box)

                    if frame_count % DRIFT_CHECK_INTERVAL == 0:
                        det = detect_head(frame)
                        if det is not None and iou(box, det) < 0.5:
                            tracker = create_tracker()
                            tracker.init(frame, tuple(det))
                            smooth_box = np.array(det, dtype=np.float32)

            if DEV_OVERLAY and smooth_box is not None:
                x, y, w, h = map(int, smooth_box)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
