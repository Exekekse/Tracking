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
from detection import detect_head_body


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

    head_tracker = None
    body_tracker = None
    head_prev = None
    body_prev = None
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

            state_head = "DETECT" if head_tracker is None else "TRACK"
            state_body = "DETECT" if body_tracker is None else "TRACK"

            if head_tracker is None or body_tracker is None:
                det_head, det_body = detect_head_body(frame, model, conf, downscale)
                if head_tracker is None and det_head is not None:
                    x, y, w, h = clamp_box_xywh(*det_head, frame.shape[1], frame.shape[0])
                    try:
                        head_tracker = create_tracker()
                        head_tracker.init(frame, (float(x), float(y), float(w), float(h)))
                        head_prev = (float(x), float(y), float(w), float(h))
                        state_head = "TRACK"
                    except Exception:
                        head_tracker = None
                        head_prev = None
                if body_tracker is None and det_body is not None:
                    bx, by, bw, bh = clamp_box_xywh(*det_body, frame.shape[1], frame.shape[0])
                    try:
                        body_tracker = create_tracker()
                        body_tracker.init(frame, (float(bx), float(by), float(bw), float(bh)))
                        body_prev = (float(bx), float(by), float(bw), float(bh))
                        state_body = "TRACK"
                    except Exception:
                        body_tracker = None
                        body_prev = None
            else:
                if head_tracker is not None:
                    ok, box = head_tracker.update(frame)
                    if not ok:
                        head_tracker = None
                        head_prev = None
                        state_head = "DETECT"
                    else:
                        hx, hy, hw, hh = [float(v) for v in box]
                        smooth = ema(head_prev, (hx, hy, hw, hh), ema_alpha)
                        smooth = clamp_box_xywh(*smooth, frame.shape[1], frame.shape[0])
                        head_prev = smooth
                if body_tracker is not None:
                    ok, box = body_tracker.update(frame)
                    if not ok:
                        body_tracker = None
                        body_prev = None
                        state_body = "DETECT"
                    else:
                        bx, by, bw, bh = [float(v) for v in box]
                        smooth = ema(body_prev, (bx, by, bw, bh), ema_alpha)
                        smooth = clamp_box_xywh(*smooth, frame.shape[1], frame.shape[0])
                        body_prev = smooth

                frame_count += 1
                if drift_check_interval > 0 and frame_count % drift_check_interval == 0:
                    det_head, det_body = detect_head_body(frame, model, conf, downscale)
                    if det_head is not None and head_prev is not None:
                        a = xywh_to_xyxy(*head_prev)
                        b = xywh_to_xyxy(*clamp_box_xywh(*det_head, frame.shape[1], frame.shape[0]))
                        if iou_xyxy(a, b) < iou_thresh:
                            try:
                                head_tracker = create_tracker()
                                x, y, w, h = det_head
                                head_tracker.init(frame, (float(x), float(y), float(w), float(h)))
                                head_prev = (float(x), float(y), float(w), float(h))
                            except Exception:
                                head_tracker = None
                                head_prev = None
                                state_head = "DETECT"
                    if det_body is not None and body_prev is not None:
                        a = xywh_to_xyxy(*body_prev)
                        b = xywh_to_xyxy(*clamp_box_xywh(*det_body, frame.shape[1], frame.shape[0]))
                        if iou_xyxy(a, b) < iou_thresh:
                            try:
                                body_tracker = create_tracker()
                                x, y, w, h = det_body
                                body_tracker.init(frame, (float(x), float(y), float(w), float(h)))
                                body_prev = (float(x), float(y), float(w), float(h))
                            except Exception:
                                body_tracker = None
                                body_prev = None
                                state_body = "DETECT"

            # FPS
            fps_frames += 1
            now = time.time()
            dt = now - fps_t0
            fps = fps_frames / dt if dt > 0 else 0.0

            # Console-Status (bei Overlay False)
            if console_status and not dev_overlay and (now - last_status_t) > 0.25:
                head_txt = (
                    f"({int(head_prev[0])},{int(head_prev[1])},{int(head_prev[2])},{int(head_prev[3])})"
                    if head_prev is not None else "None"
                )
                body_txt = (
                    f"({int(body_prev[0])},{int(body_prev[1])},{int(body_prev[2])},{int(body_prev[3])})"
                    if body_prev is not None else "None"
                )
                print(f"[H:{state_head} B:{state_body}] head={head_txt} body={body_txt} fps={fps:.1f}")
                last_status_t = now

            # Overlay / Fenster
            if dev_overlay:
                if body_prev is not None:
                    bx, by, bw, bh = [int(v) for v in body_prev]
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
                if head_prev is not None:
                    hx, hy, hw, hh = [int(v) for v in head_prev]
                    cv2.rectangle(frame, (hx, hy), (hx + hw, hy + hh), (0, 255, 0), 2)
                cv2.putText(frame, f"H:{state_head} B:{state_body} | FPS {fps:.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.imshow("Head/Body Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
