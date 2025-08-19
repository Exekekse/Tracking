import time
import cv2
import numpy as np
import mss
from ultralytics import YOLO
import torch
import pathlib

from config import (
    DEFAULT_AREA_RATIO,
    DEFAULT_VIEWMODEL_Y,
    DEFAULT_HEAD_AR,
    DEFAULT_BODY_AR,
    DEFAULT_LOCK_FRAMES,
    DEFAULT_HEATMAP_GRID,
    DEFAULT_HEATMAP_INTERVAL,
    DEFAULT_STORAGE_LIMIT_MB,
    DEFAULT_TARGET_FPS,
)
from utils import (
    create_tracker,
    clamp_box_xywh,
    ema,
    xywh_to_xyxy,
    iou_xyxy,
    get_storage_paths,
    load_heatmap,
    save_heatmap,
    load_calibration,
    save_calibration,
    Heatmap,
    candidate_valid,
    head_above_body,
    enforce_size_limit,
    setup_logger,
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

    paths = get_storage_paths()
    logger = setup_logger(paths)
    defaults = {
        "conf": conf,
        "area_ratio": DEFAULT_AREA_RATIO,
        "viewmodel_y": DEFAULT_VIEWMODEL_Y,
        "head_ar": DEFAULT_HEAD_AR,
        "body_ar": DEFAULT_BODY_AR,
        "drift_check_interval": drift_check_interval,
        "iou_thresh": iou_thresh,
        "detection_downscale": downscale,
    }
    calib, calib_path = load_calibration(paths, defaults)
    conf_dyn = calib.get("conf", conf)

    grid_w, grid_h = DEFAULT_HEATMAP_GRID
    shape = (grid_h, grid_w)
    heatmap_head = Heatmap(shape)
    heatmap_body = Heatmap(shape)
    heatmap_head.grid = load_heatmap(paths["data"] / "heatmap_head.npy", shape)
    heatmap_body.grid = load_heatmap(paths["data"] / "heatmap_body.npy", shape)
    ignore_mask_path = pathlib.Path(calib.get("ignore_mask_path", paths["data"] / "ignore_mask.npy"))
    ignore_mask = load_heatmap(ignore_mask_path, shape)

    head_tracker = None
    body_tracker = None
    head_prev = None
    body_prev = None
    head_streak = 0
    body_streak = 0
    frame_count = 0
    redetects = 0
    last_status_t = 0.0

    with mss.mss() as sct:
        if monitor_index >= len(sct.monitors):
            raise RuntimeError(f"Monitor-Index {monitor_index} existiert nicht. Verfügbare: 0..{len(sct.monitors)-1}")
        monitor = sct.monitors[monitor_index]

        fps_t0 = time.time()
        fps_frames = 0
        drift_interval_cur = calib["drift_check_interval"]

        while True:
            raw = np.array(sct.grab(monitor))  # BGRA
            frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
            h, w = frame.shape[:2]

            fps_frames += 1
            now = time.time()
            dt = now - fps_t0
            fps = fps_frames / dt if dt > 0 else 0.0
            if fps < DEFAULT_TARGET_FPS * 0.8:
                drift_interval_cur = calib["drift_check_interval"] * 2
                learning = False
            else:
                drift_interval_cur = calib["drift_check_interval"]
                learning = True

            state_head = "DETECT" if head_tracker is None else "TRACK"
            state_body = "DETECT" if body_tracker is None else "TRACK"

            need_detect = (
                head_tracker is None
                or body_tracker is None
                or (drift_interval_cur > 0 and frame_count % drift_interval_cur == 0)
            )

            if need_detect:
                redetects += 1
                det_head, det_body = detect_head_body(frame, model, conf_dyn, downscale)
                if learning:
                    if det_head is not None:
                        cx = det_head[0] + det_head[2] * 0.5
                        cy = det_head[1] + det_head[3] * 0.5
                        heatmap_head.accumulate(cx, cy, w, h)
                    if det_body is not None:
                        cx = det_body[0] + det_body[2] * 0.5
                        cy = det_body[1] + det_body[3] * 0.5
                        heatmap_body.accumulate(cx, cy, w, h)
                    if heatmap_body.counter >= DEFAULT_HEATMAP_INTERVAL:
                        new_mask = heatmap_body.build_mask()
                        ignore_mask = new_mask if ignore_mask is None else np.maximum(ignore_mask, new_mask)
                        save_heatmap(ignore_mask_path, ignore_mask)
                        save_heatmap(paths["data"] / "heatmap_head.npy", heatmap_head.grid)
                        save_heatmap(paths["data"] / "heatmap_body.npy", heatmap_body.grid)
                        heatmap_head.reset()
                        heatmap_body.reset()
                        stability = max(0.0, 1.0 - redetects / float(DEFAULT_HEATMAP_INTERVAL))
                        if stability > 0.8:
                            conf_dyn = max(0.1, conf_dyn - 0.05)
                        elif stability < 0.5:
                            conf_dyn = min(0.9, conf_dyn + 0.05)
                        redetects = 0
                        save_calibration(calib_path, {**calib, "conf": conf_dyn, "ignore_mask_path": str(ignore_mask_path)})
                        enforce_size_limit(paths, int(DEFAULT_STORAGE_LIMIT_MB * 1024 * 1024))

                if not candidate_valid(det_head, w, h, calib["head_ar"], ignore_mask, calib["area_ratio"], calib["viewmodel_y"]):
                    det_head = None
                if not candidate_valid(det_body, w, h, calib["body_ar"], ignore_mask, calib["area_ratio"], calib["viewmodel_y"]):
                    det_body = None
                if det_head and det_body and not head_above_body(det_head, det_body):
                    det_head = None
                    det_body = None

                if head_tracker is None and det_head is not None:
                    head_streak += 1
                    if head_streak >= DEFAULT_LOCK_FRAMES:
                        x, y, w1, h1 = clamp_box_xywh(*det_head, w, h)
                        try:
                            head_tracker = create_tracker()
                            head_tracker.init(frame, (float(x), float(y), float(w1), float(h1)))
                            head_prev = (float(x), float(y), float(w1), float(h1))
                            state_head = "TRACK"
                        except Exception:
                            head_tracker = None
                            head_prev = None
                        head_streak = 0
                elif head_tracker is None:
                    head_streak = 0

                if body_tracker is None and det_body is not None:
                    body_streak += 1
                    if body_streak >= DEFAULT_LOCK_FRAMES:
                        bx, by, bw, bh = clamp_box_xywh(*det_body, w, h)
                        try:
                            body_tracker = create_tracker()
                            body_tracker.init(frame, (float(bx), float(by), float(bw), float(bh)))
                            body_prev = (float(bx), float(by), float(bw), float(bh))
                            state_body = "TRACK"
                        except Exception:
                            body_tracker = None
                            body_prev = None
                        body_streak = 0
                elif body_tracker is None:
                    body_streak = 0

                if head_tracker is not None and det_head is not None and head_prev is not None:
                    a = xywh_to_xyxy(*head_prev)
                    b = xywh_to_xyxy(*clamp_box_xywh(*det_head, w, h))
                    if iou_xyxy(a, b) < iou_thresh:
                        try:
                            head_tracker = create_tracker()
                            x, y, w1, h1 = det_head
                            head_tracker.init(frame, (float(x), float(y), float(w1), float(h1)))
                            head_prev = (float(x), float(y), float(w1), float(h1))
                        except Exception:
                            head_tracker = None
                            head_prev = None
                            state_head = "DETECT"
                if body_tracker is not None and det_body is not None and body_prev is not None:
                    a = xywh_to_xyxy(*body_prev)
                    b = xywh_to_xyxy(*clamp_box_xywh(*det_body, w, h))
                    if iou_xyxy(a, b) < iou_thresh:
                        try:
                            body_tracker = create_tracker()
                            bx, by, bw, bh = det_body
                            body_tracker.init(frame, (float(bx), float(by), float(bw), float(bh)))
                            body_prev = (float(bx), float(by), float(bw), float(bh))
                        except Exception:
                            body_tracker = None
                            body_prev = None
                            state_body = "DETECT"
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
                        head_prev = clamp_box_xywh(*smooth, w, h)
                if body_tracker is not None:
                    ok, box = body_tracker.update(frame)
                    if not ok:
                        body_tracker = None
                        body_prev = None
                        state_body = "DETECT"
                    else:
                        bx, by, bw, bh = [float(v) for v in box]
                        smooth = ema(body_prev, (bx, by, bw, bh), ema_alpha)
                        body_prev = clamp_box_xywh(*smooth, w, h)
                if head_prev and body_prev and not head_above_body(head_prev, body_prev):
                    head_tracker = None
                    body_tracker = None
                    head_prev = None
                    body_prev = None
                    state_head = "DETECT"
                    state_body = "DETECT"

            frame_count += 1

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
