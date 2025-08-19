import time
import cv2
import numpy as np
import mss
from ultralytics import YOLO
import torch

from utils import (
    create_tracker,
    clamp_box_xywh,
    ema,
    xywh_to_xyxy,
    iou_xyxy,
)
from detection import detect_head


def run_tracking(model_path: str,
                 conf: float,
                 ema_alpha: float,
                 drift_check_interval: int,
                 iou_thresh: float,
                 downscale: float,
                 monitor_index: int,
                 dev_overlay: bool,
                 console_status: bool):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    model.to(device)

    tracker = None
    prev_box = None
    frame_count = 0
    last_status_t = 0.0

    with mss.mss() as sct:
        if monitor_index >= len(sct.monitors):
            raise RuntimeError(f"Monitor-Index {monitor_index} existiert nicht. Verfügbare: 0..{len(sct.monitors)-1}")
        monitor = sct.monitors[monitor_index]

        fps_t0 = time.time()
        fps_frames = 0

        while True:
            raw = np.array(sct.grab(monitor))  # BGRA
            frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

            state = "DETECT" if tracker is None else "TRACK"

            if tracker is None:
                det = detect_head(frame, model, conf, downscale)
                if det is not None:
                    x, y, w, h = clamp_box_xywh(*det, frame.shape[1], frame.shape[0])
                    try:
                        tracker = create_tracker()
                        tracker.init(frame, (float(x), float(y), float(w), float(h)))
                        prev_box = (float(x), float(y), float(w), float(h))
                        state = "TRACK"
                    except Exception:
                        tracker = None
                        prev_box = None
            else:
                ok, box = tracker.update(frame)
                if not ok:
                    tracker = None
                    prev_box = None
                    state = "DETECT"
                else:
                    bx, by, bw, bh = [float(v) for v in box]
                    smooth = ema(prev_box, (bx, by, bw, bh), ema_alpha)
                    smooth = clamp_box_xywh(*smooth, frame.shape[1], frame.shape[0])
                    prev_box = smooth

                    frame_count += 1
                    if drift_check_interval > 0 and frame_count % drift_check_interval == 0:
                        det = detect_head(frame, model, conf, downscale)
                        if det is not None:
                            a = xywh_to_xyxy(*prev_box)
                            b = xywh_to_xyxy(*clamp_box_xywh(*det, frame.shape[1], frame.shape[0]))
                            if iou_xyxy(a, b) < iou_thresh:
                                try:
                                    tracker = create_tracker()
                                    tracker.init(frame, (float(det[0]), float(det[1]), float(det[2]), float(det[3])))
                                    prev_box = det
                                except Exception:
                                    tracker = None
                                    prev_box = None
                                    state = "DETECT"

            # FPS
            fps_frames += 1
            now = time.time()
            dt = now - fps_t0
            fps = fps_frames / dt if dt > 0 else 0.0

            # Console-Status (bei Overlay False)
            if console_status and not dev_overlay and (now - last_status_t) > 0.25:
                if prev_box is not None:
                    x, y, w, h = [int(v) for v in prev_box]
                    print(f"[{state}] box=({x},{y},{w},{h}) fps={fps:.1f}")
                else:
                    print(f"[{state}] box=None fps={fps:.1f}")
                last_status_t = now

            # Overlay / Fenster
            if dev_overlay:
                if prev_box is not None:
                    x, y, w, h = [int(v) for v in prev_box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{state} | FPS {fps:.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.imshow("Head Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
