import cv2
import mss
import numpy as np
import torch
from ultralytics import YOLO

# set True to visualise tracking for development
DEV_OVERLAY = False
CHECK_INTERVAL = 60  # frames between drift checks
IOU_THRESH = 0.5


def create_tracker() -> cv2.Tracker:
    """Create a MOSSE tracker with cross-version compatibility."""
    return (
        cv2.TrackerMOSSE_create()
        if hasattr(cv2, "TrackerMOSSE_create")
        else cv2.legacy.TrackerMOSSE_create()
    )


class HeadTracker:
    """Detect once with YOLO, then track a single head using OpenCV."""

    def __init__(
        self,
        model_path: str,
        conf: float = 0.5,
        alpha: float = 0.3,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path).to(device)
        self.conf = conf
        self.alpha = alpha  # smoothing factor for bounding box
        self.tracker: cv2.Tracker | None = None
        self.prev_box: np.ndarray | None = None
        self.frame_count = 0

    def detect(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """Return first head box as (x, y, w, h)."""
        results = self.model(frame, conf=self.conf, classes=0, verbose=False)[0]
        if not results.boxes:
            return None
        x1, y1, x2, y2 = results.boxes.xyxy[0].cpu().numpy().astype(int)
        return x1, y1, x2 - x1, y2 - y1

    def init_tracker(self, frame: np.ndarray, box: tuple[int, int, int, int]):
        self.tracker = create_tracker()
        self.tracker.init(frame, box)
        self.prev_box = np.array(box, dtype=np.float32)

    @staticmethod
    def _xywh_to_xyxy(box: np.ndarray | tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = map(float, box)
        return np.array([x, y, x + w, y + h], dtype=np.float32)

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        xa1, ya1, xa2, ya2 = a
        xb1, yb1, xb2, yb2 = b
        ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
        ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = (xa2 - xa1) * (ya2 - ya1)
        area_b = (xb2 - xb1) * (yb2 - yb1)
        return inter / (area_a + area_b - inter)

    def update(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        if self.tracker is None or self.prev_box is None:
            det = self.detect(frame)
            if det is not None:
                self.init_tracker(frame, det)
            return det

        self.frame_count += 1
        ok, box = self.tracker.update(frame)
        if not ok:
            det = self.detect(frame)
            if det is not None:
                self.init_tracker(frame, det)
                return det
            self.tracker = None
            return None

        box = np.array(box, dtype=np.float32)
        self.prev_box = self.alpha * box + (1 - self.alpha) * self.prev_box

        if self.frame_count % CHECK_INTERVAL == 0:
            det = self.detect(frame)
            if det is not None:
                iou = self._iou(
                    self._xywh_to_xyxy(self.prev_box),
                    self._xywh_to_xyxy(det),
                )
                if iou < IOU_THRESH:
                    self.init_tracker(frame, det)
                    return det
        return tuple(map(int, self.prev_box))


def main() -> None:
    model_path = "yolov8n-head.pt"  # path to head detection model
    ht = HeadTracker(model_path)

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        box = ht.update(frame)
        if box is None:
            return

        while True:
            frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            box = ht.update(frame)
            if DEV_OVERLAY and box:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Head Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
