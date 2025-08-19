import cv2
import mss
import numpy as np
import torch
from ultralytics import YOLO


def create_tracker() -> cv2.Tracker:
    """Create a MOSSE tracker with cross-version compatibility."""
    return (
        cv2.TrackerMOSSE_create()
        if hasattr(cv2, "TrackerMOSSE_create")
        else cv2.legacy.TrackerMOSSE_create()
    )


class HeadTracker:
    """Detect once with YOLO, then track a single head using OpenCV."""

    def __init__(self, model_path: str, conf: float = 0.5, alpha: float = 0.3):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path).to(device)
        self.conf = conf
        self.alpha = alpha  # smoothing factor for bounding box
        self.tracker: cv2.Tracker | None = None
        self.prev_box: np.ndarray | None = None

    def detect(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """Return first head box as (x, y, w, h)."""
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        if not results.boxes:
            return None
        x1, y1, x2, y2 = results.boxes.xyxy[0].cpu().numpy().astype(int)
        return x1, y1, x2 - x1, y2 - y1

    def init_tracker(self, frame: np.ndarray, box: tuple[int, int, int, int]):
        self.tracker = create_tracker()
        self.tracker.init(frame, box)
        self.prev_box = np.array(box, dtype=np.float32)

    def update(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        if self.tracker is None or self.prev_box is None:
            return None
        ok, box = self.tracker.update(frame)
        if not ok:
            return None
        box = np.array(box, dtype=np.float32)
        self.prev_box = self.alpha * box + (1 - self.alpha) * self.prev_box
        return tuple(map(int, self.prev_box))


def main() -> None:
    model_path = "yolov8n-head.pt"  # path to head detection model
    ht = HeadTracker(model_path)

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        box = ht.detect(frame)
        if box is None:
            return
        ht.init_tracker(frame, box)

        while True:
            frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            box = ht.update(frame)
            if box:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Head Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
