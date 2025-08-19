import cv2
import mss
import numpy as np
import torch
from ultralytics import YOLO


def create_tracker():
    """Return a CSRT tracker instance compatible with current OpenCV version."""
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    return cv2.legacy.TrackerCSRT_create()


class HeadTracker:
    def __init__(self, model_path: str, conf: float = 0.5, smooth: float = 0.6):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.conf = conf
        self.smooth = smooth
        self.trackers: list[cv2.Tracker] = []
        self.prev_boxes: list[np.ndarray] = []

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Run YOLO and return head boxes as (x, y, w, h)."""
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        boxes = []
        if results.boxes is None:
            return boxes
        xyxy = results.boxes.xyxy.cpu().numpy().astype(int)
        for x1, y1, x2, y2 in xyxy:
            boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes

    def init_trackers(self, frame: np.ndarray, boxes: list[tuple[int, int, int, int]]):
        self.trackers = []
        self.prev_boxes = []
        for box in boxes:
            tracker = create_tracker()
            tracker.init(frame, box)
            self.trackers.append(tracker)
            self.prev_boxes.append(np.array(box, dtype=np.float32))

    def update(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Update all trackers and return smoothed boxes."""
        new_trackers = []
        new_prev = []
        smoothed_boxes = []
        for tracker, prev in zip(self.trackers, self.prev_boxes):
            ok, box = tracker.update(frame)
            if not ok:
                continue
            box = np.array(box, dtype=np.float32)
            smooth_box = self.smooth * box + (1 - self.smooth) * prev
            new_trackers.append(tracker)
            new_prev.append(smooth_box)
            smoothed_boxes.append(tuple(map(int, smooth_box)))
        self.trackers = new_trackers
        self.prev_boxes = new_prev
        return smoothed_boxes


def main():
    model_path = "yolov8n-head.pt"  # replace with your head-specific model
    ht = HeadTracker(model_path)
    detect_interval = 30
    frame_idx = 0

    with mss.mss() as sct:
        monitor = sct.monitors[1]  # primary monitor
        while True:
            frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            if not ht.trackers or frame_idx % detect_interval == 0:
                boxes = ht.detect(frame)
                if boxes:
                    ht.init_trackers(frame, boxes)
            boxes = ht.update(frame)

            for x, y, w, h in boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Head Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_idx += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
