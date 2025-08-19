"""ScreenTracker v2 – simplified object tracking demo."""

from __future__ import annotations

import os
import signal
import time
from typing import List

import cv2
import mss
import numpy as np
import torch
from ultralytics import YOLO

import config
from detection import Detection
from overlay import Drawer, KeyHelper, MenuState
from pose_parts import compute_parts_for_detection, export_parts_for_training, rescale_keypoints
from rule_engine import RuleEngine
from tracker import CentroidTracker


UI_WATCHDOG_TIMEOUT = 1.0  # seconds


def safe_mkdir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def export_for_training(frame_bgr: np.ndarray, detections: List[Detection], names_map: dict) -> None:
    """Speichert Frame + YOLO-Labeldatei (Pseudo-Labels) in ``./captures``."""

    safe_mkdir("captures")
    ts = time.strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join("captures", f"cap_{ts}.jpg")
    txt_path = os.path.join("captures", f"cap_{ts}.txt")

    cv2.imwrite(img_path, frame_bgr)

    h, w = frame_bgr.shape[:2]
    with open(txt_path, "w", encoding="utf-8") as f:
        for d in detections:
            cx = (d.x1 + d.x2) / 2.0 / w
            cy = (d.y1 + d.y2) / 2.0 / h
            bw = (d.x2 - d.x1) / w
            bh = (d.y2 - d.y1) / h
            f.write(f"{d.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"[TRAIN] Exportiert: {img_path} + {txt_path}")


def main() -> None:
    running = True

    def handle_sigint(sig, frame):
        nonlocal running
        print("\n[INFO] SIGINT erhalten. Beende…")
        running = False

    signal.signal(signal.SIGINT, handle_sigint)

    pose_enabled = False
    try:
        if config.ENABLE_POSE_PARTS:
            model = YOLO(config.POSE_MODEL_PATH)
            pose_enabled = True
        else:
            model = YOLO(config.MODEL_PATH)
    except Exception as exc:
        print(f"[WARN] Pose-Modell konnte nicht geladen werden: {exc}. Parts deaktiviert")
        pose_enabled = False
        model = YOLO(config.MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    names_raw = model.names
    names = names_raw if isinstance(names_raw, dict) else {i: n for i, n in enumerate(names_raw)}

    person_id = None
    for k, v in names.items():
        if v.lower() == "person":
            person_id = int(k)
            break

    menu_state = MenuState(
        conf_thres=config.CONF_THRES,
        person_only=config.PERSON_ONLY,
        show_trails=True,
        show_hud=True,
        show_parts=False,
    )

    key = KeyHelper()
    drawer = Drawer(names)
    engine = RuleEngine()
    engine.load_rules()
    ct = CentroidTracker(max_disappeared=20, trail_len=24, dist_thresh=80)

    with mss.mss() as sct:
        monitor = sct.monitors[config.MONITOR_INDEX]
        roi = config.CAPTURE_ROI

        win_name = "ScreenTracker v2"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.resizeWindow(win_name, 1280, 720)

        def on_mouse(event, x, y, flags, param):
            nonlocal menu_state
            if event == cv2.EVENT_LBUTTONDOWN and drawer.menu_open:
                drawer.menu_click(x, y, menu_state)

        cv2.setMouseCallback(win_name, on_mouse)

        prev = time.perf_counter()
        fps_smooth = 0.0

        try:
            while running:
                if roi:
                    frame = np.array(sct.grab(roi))
                else:
                    frame = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                orig_h, orig_w = frame.shape[:2]
                infer_img = frame
                sx = sy = 1.0
                if config.INFER_RESIZE:
                    infer_img = cv2.resize(frame, (config.INFER_RESIZE, config.INFER_RESIZE))
                    sx = orig_w / config.INFER_RESIZE
                    sy = orig_h / config.INFER_RESIZE

                results = model(infer_img, conf=menu_state.conf_thres, verbose=False)
                r = results[0]

                dets: List[Detection] = []
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    if config.INFER_RESIZE:
                        xyxy[:, [0, 2]] *= sx
                        xyxy[:, [1, 3]] *= sy
                    xyxy = xyxy.astype(int)
                    confs = r.boxes.conf.cpu().numpy()
                    clss = r.boxes.cls.cpu().numpy().astype(int)
                    kpts_xy = None
                    kpts_conf = None
                    if pose_enabled and r.keypoints is not None:
                        kpts_xy = r.keypoints.xy.cpu().numpy()
                        kpts_conf = r.keypoints.conf.cpu().numpy()
                        if config.INFER_RESIZE:
                            kpts_xy[:, :, 0] *= sx
                            kpts_xy[:, :, 1] *= sy
                    for idx, ((x1, y1, x2, y2), c, cls) in enumerate(zip(xyxy, confs, clss)):
                        if menu_state.person_only and person_id is not None and cls != person_id:
                            continue
                        kps = None
                        if pose_enabled and kpts_xy is not None:
                            kps = [(float(x), float(y), float(kpts_conf[idx][j])) for j, (x, y) in enumerate(kpts_xy[idx])]
                        dets.append(Detection(int(x1), int(y1), int(x2), int(y2), int(cls), float(c), kps))

                dets = engine.apply(dets, frame)
                if pose_enabled:
                    for d in dets:
                        d.parts = compute_parts_for_detection(d, frame.shape[:2])
                id2box = ct.update([(d.x1, d.y1, d.x2, d.y2) for d in dets])

                annotated = frame.copy()
                drawer.draw_scene(
                    annotated,
                    dets=dets,
                    id2box=id2box,
                    trails=ct.trails if menu_state.show_trails else {},
                    show_hud=menu_state.show_hud,
                    show_parts=menu_state.show_parts and pose_enabled,
                    fps_smooth=fps_smooth,
                    person_only=menu_state.person_only,
                    conf_thres=menu_state.conf_thres,
                )

                if drawer.menu_open:
                    drawer.draw_menu(annotated, menu_state)

                ui_start = time.perf_counter()
                cv2.imshow(win_name, annotated)
                k = cv2.waitKey(1) & 0xFFFFFFFF
                ui_dt = time.perf_counter() - ui_start
                if ui_dt > UI_WATCHDOG_TIMEOUT:
                    print(f"[WARN] UI blockiert für {ui_dt:.2f}s")

                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    running = False

                now = time.perf_counter()
                dt = now - prev
                prev = now
                inst = 1.0 / dt if dt > 0 else 0.0
                fps_smooth = (fps_smooth * 0.9) + (inst * 0.1) if fps_smooth > 0 else inst

                if k != -1:
                    if key.is_quit(k):
                        running = False
                    elif key.is_toggle_menu(k):
                        drawer.menu_open = not drawer.menu_open
                    elif key.is_toggle_filter(k):
                        menu_state.person_only = not menu_state.person_only
                    elif key.is_reset_tracker(k):
                        ct = CentroidTracker(max_disappeared=20, trail_len=24, dist_thresh=80)
                    elif key.is_screenshot(k):
                        drawer.save_frame(annotated)
                    elif key.is_capture(k):
                        export_for_training(frame, dets, names)
                    elif key.is_toggle_parts(k) and pose_enabled:
                        menu_state.show_parts = not menu_state.show_parts
                        print(f"[INFO] Parts {'ON' if menu_state.show_parts else 'OFF'}")
                    elif key.is_export_parts(k) and pose_enabled:
                        export_parts_for_training(frame, dets, config.EXPORT_PARTS_DIR)
                    elif key.is_reload_rules(k):
                        engine.load_rules()
                    elif drawer.menu_open:
                        if key.is_plus(k):
                            menu_state.conf_thres = min(0.99, menu_state.conf_thres + 0.02)
                        elif key.is_minus(k):
                            menu_state.conf_thres = max(0.01, menu_state.conf_thres - 0.02)
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

