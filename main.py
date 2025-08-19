import os
import time
import math
import cv2
import numpy as np
import mss
from ultralytics import YOLO
import torch

# -------------------------------------------------
# Defaults (werden im Menü geändert, nicht via ENV)
# -------------------------------------------------
DEFAULT_MODEL_PATH = "yolov8n.pt"   # Falls du ein Head-Model hast: yolov8n-head.pt
DEFAULT_CONF = 0.5                  # YOLO Konfidenz
DEFAULT_EMA_ALPHA = 0.30            # Glättung (0..1) – höher = snappier
DEFAULT_DRIFT_CHECK_INTERVAL = 60   # alle N Frames Drift prüfen
DEFAULT_IOU_THRESH = 0.50           # Re-Init wenn IoU darunter fällt
DEFAULT_DOWNSCALE = 1.0             # 1.0=volle Auflösung; >1.0 = kleinere Detektion
DEFAULT_MONITOR_INDEX = 1           # mss Monitor-Index (1 = Hauptmonitor)
DEFAULT_DEV_OVERLAY = False         # Overlay/Window
DEFAULT_CONSOLE_STATUS = True       # Konsolen-Statusausgabe bei Overlay=False


# -------------------------
# Tracker helpers
# -------------------------

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


# -------------------------
# Geometrie & Hilfsfunktionen
# -------------------------

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


# -------------------------
# YOLO-Detektion (Head-only)
# -------------------------

def _select_person_box(results):
    boxes = []
    try:
        r0 = results[0]
    except Exception:
        return None
    if getattr(r0, "boxes", None) is None or len(r0.boxes) == 0:
        return None
    cpu_boxes = r0.boxes.cpu()
    xyxy = cpu_boxes.xyxy.numpy()
    cls = cpu_boxes.cls.numpy() if hasattr(cpu_boxes, "cls") else None
    conf = cpu_boxes.conf.numpy() if hasattr(cpu_boxes, "conf") else None
    for i in range(xyxy.shape[0]):
        if cls is not None and int(cls[i]) != 0:  # 0 = person (COCO)
            continue
        x1, y1, x2, y2 = map(float, xyxy[i])
        c = float(conf[i]) if conf is not None else 0.0
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        boxes.append((c, area, (x1, y1, x2, y2)))
    if not boxes:
        return None
    boxes.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return boxes[0][2]


def _derive_head_from_person(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    head_h = max(8.0, 0.35 * h)
    head_w = max(8.0, 0.60 * w)
    cx = x1 + w * 0.5
    x = cx - head_w * 0.5
    y = y1
    return xyxy_to_xywh(x, y, x + head_w, y + head_h)


def detect_head(frame_bgr: np.ndarray, model: YOLO, conf: float, downscale: float):
    h, w = frame_bgr.shape[:2]
    det_frame = frame_bgr
    scale = 1.0
    if downscale and downscale > 1.0:
        det_frame = cv2.resize(frame_bgr, (w // int(downscale), h // int(downscale)), interpolation=cv2.INTER_LINEAR)
        scale = float(w) / float(det_frame.shape[1])

    results = model(det_frame, conf=conf, verbose=False)

    # Head-Model-Heuristik: hat das Modell exakt 1 Klasse? Dann sind es Köpfe.
    is_head_model = False
    try:
        nc = getattr(model.model, "nc", None)
        is_head_model = (nc == 1)
    except Exception:
        pass

    if is_head_model:
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return None
        x1, y1, x2, y2 = map(float, r0.boxes.xyxy[0].cpu().numpy())
        x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
        return xyxy_to_xywh(x1, y1, x2, y2)

    # Sonst: Person finden und Kopfbereich daraus ableiten
    person_xyxy = _select_person_box(results)
    if person_xyxy is None:
        return None
    x1, y1, x2, y2 = person_xyxy
    x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
    return _derive_head_from_person(x1, y1, x2, y2)


# -------------------------
# Tracking-Loop (mit optionalem Overlay & Console-Status)
# -------------------------

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


# -------------------------
# Einfaches Konsolen-Menü
# -------------------------

def list_monitors():
    with mss.mss() as sct:
        for idx, mon in enumerate(sct.monitors):
            w = mon.get('width'); h = mon.get('height')
            print(f"  [{idx}] {w}x{h} +{mon.get('left',0)},{mon.get('top',0)}")


def run_menu():
    cfg = {
        "model_path": DEFAULT_MODEL_PATH,
        "conf": DEFAULT_CONF,
        "ema_alpha": DEFAULT_EMA_ALPHA,
        "drift_check_interval": DEFAULT_DRIFT_CHECK_INTERVAL,
        "iou_thresh": DEFAULT_IOU_THRESH,
        "downscale": DEFAULT_DOWNSCALE,
        "monitor_index": DEFAULT_MONITOR_INDEX,
        "dev_overlay": DEFAULT_DEV_OVERLAY,
        "console_status": DEFAULT_CONSOLE_STATUS,
    }

    while True:
        print("\n=== Head Tracking Menü ===")
        print(f"Model: {cfg['model_path']} | conf={cfg['conf']} | ema={cfg['ema_alpha']} | driftN={cfg['drift_check_interval']} | IoU={cfg['iou_thresh']} | downscale={cfg['downscale']}")
        print(f"Monitor: {cfg['monitor_index']} | Overlay={cfg['dev_overlay']} | ConsoleStatus={cfg['console_status']}")
        print("\nAktionen:")
        print("  1) Tracking starten (ohne Overlay)")
        print("  2) Tracking starten (mit Overlay)")
        print("  3) Overlay umschalten (on/off)")
        print("  4) Monitor wechseln")
        print("  5) Model-Pfad ändern")
        print("  6) Downscale ändern (z.B. 1.0 / 2.0)")
        print("  7) Konfidenz ändern (0.1..0.9)")
        print("  8) Beenden")
        choice = input("Auswahl: ").strip()

        if choice == "1":
            cfg["dev_overlay"] = False
            run_tracking(**cfg)
        elif choice == "2":
            cfg["dev_overlay"] = True
            run_tracking(**cfg)
        elif choice == "3":
            cfg["dev_overlay"] = not cfg["dev_overlay"]
        elif choice == "4":
            print("Verfügbare Monitore:")
            list_monitors()
            try:
                idx = int(input("Monitor-Index: ").strip())
                cfg["monitor_index"] = idx
            except Exception:
                print("Ungültiger Index.")
        elif choice == "5":
            p = input("Neuer Model-Pfad: ").strip()
            if p:
                cfg["model_path"] = p
        elif choice == "6":
            try:
                ds = float(input("Downscale (>1.0 = kleiner): ").strip())
                cfg["downscale"] = max(1.0, ds)
            except Exception:
                print("Ungültiger Wert.")
        elif choice == "7":
            try:
                c = float(input("Konfidenz (0.1..0.9): ").strip())
                cfg["conf"] = min(max(0.1, c), 0.9)
            except Exception:
                print("Ungültiger Wert.")
        elif choice == "8":
            print("Tschüss!")
            break
        else:
            print("Unbekannte Auswahl.")


if __name__ == "__main__":
    # Standardmäßig Menü starten, sodass `python main.py` immer das Menü zeigt
    try:
        run_menu()
    except RuntimeError as e:
        print(f"Fehler: {e}")
        print("Hinweis: Installiere opencv-contrib-python und stelle sicher, dass NumPy/Torch kompatibel sind.")
import os
import time
import math
import cv2
import numpy as np
import mss
from ultralytics import YOLO
import torch

# -------------------------------------------------
# Defaults (werden im Menü geändert, nicht via ENV)
# -------------------------------------------------
DEFAULT_MODEL_PATH = "yolov8n.pt"   # Falls du ein Head-Model hast: yolov8n-head.pt
DEFAULT_CONF = 0.5                  # YOLO Konfidenz
DEFAULT_EMA_ALPHA = 0.30            # Glättung (0..1) – höher = snappier
DEFAULT_DRIFT_CHECK_INTERVAL = 60   # alle N Frames Drift prüfen
DEFAULT_IOU_THRESH = 0.50           # Re-Init wenn IoU darunter fällt
DEFAULT_DOWNSCALE = 1.0             # 1.0=volle Auflösung; >1.0 = kleinere Detektion
DEFAULT_MONITOR_INDEX = 1           # mss Monitor-Index (1 = Hauptmonitor)
DEFAULT_DEV_OVERLAY = False         # Overlay/Window
DEFAULT_CONSOLE_STATUS = True       # Konsolen-Statusausgabe bei Overlay=False


# -------------------------
# Tracker helpers
# -------------------------

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


# -------------------------
# Geometrie & Hilfsfunktionen
# -------------------------

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


# -------------------------
# YOLO-Detektion (Head-only)
# -------------------------

def _select_person_box(results):
    boxes = []
    try:
        r0 = results[0]
    except Exception:
        return None
    if getattr(r0, "boxes", None) is None or len(r0.boxes) == 0:
        return None
    cpu_boxes = r0.boxes.cpu()
    xyxy = cpu_boxes.xyxy.numpy()
    cls = cpu_boxes.cls.numpy() if hasattr(cpu_boxes, "cls") else None
    conf = cpu_boxes.conf.numpy() if hasattr(cpu_boxes, "conf") else None
    for i in range(xyxy.shape[0]):
        if cls is not None and int(cls[i]) != 0:  # 0 = person (COCO)
            continue
        x1, y1, x2, y2 = map(float, xyxy[i])
        c = float(conf[i]) if conf is not None else 0.0
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        boxes.append((c, area, (x1, y1, x2, y2)))
    if not boxes:
        return None
    boxes.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return boxes[0][2]


def _derive_head_from_person(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    head_h = max(8.0, 0.35 * h)
    head_w = max(8.0, 0.60 * w)
    cx = x1 + w * 0.5
    x = cx - head_w * 0.5
    y = y1
    return xyxy_to_xywh(x, y, x + head_w, y + head_h)


def detect_head(frame_bgr: np.ndarray, model: YOLO, conf: float, downscale: float):
    h, w = frame_bgr.shape[:2]
    det_frame = frame_bgr
    scale = 1.0
    if downscale and downscale > 1.0:
        det_frame = cv2.resize(frame_bgr, (w // int(downscale), h // int(downscale)), interpolation=cv2.INTER_LINEAR)
        scale = float(w) / float(det_frame.shape[1])

    results = model(det_frame, conf=conf, verbose=False)

    # Head-Model-Heuristik: hat das Modell exakt 1 Klasse? Dann sind es Köpfe.
    is_head_model = False
    try:
        nc = getattr(model.model, "nc", None)
        is_head_model = (nc == 1)
    except Exception:
        pass

    if is_head_model:
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return None
        x1, y1, x2, y2 = map(float, r0.boxes.xyxy[0].cpu().numpy())
        x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
        return xyxy_to_xywh(x1, y1, x2, y2)

    # Sonst: Person finden und Kopfbereich daraus ableiten
    person_xyxy = _select_person_box(results)
    if person_xyxy is None:
        return None
    x1, y1, x2, y2 = person_xyxy
    x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
    return _derive_head_from_person(x1, y1, x2, y2)


# -------------------------
# Tracking-Loop (mit optionalem Overlay & Console-Status)
# -------------------------

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


# -------------------------
# Einfaches Konsolen-Menü
# -------------------------

def list_monitors():
    with mss.mss() as sct:
        for idx, mon in enumerate(sct.monitors):
            w = mon.get('width'); h = mon.get('height')
            print(f"  [{idx}] {w}x{h} +{mon.get('left',0)},{mon.get('top',0)}")


def run_menu():
    cfg = {
        "model_path": DEFAULT_MODEL_PATH,
        "conf": DEFAULT_CONF,
        "ema_alpha": DEFAULT_EMA_ALPHA,
        "drift_check_interval": DEFAULT_DRIFT_CHECK_INTERVAL,
        "iou_thresh": DEFAULT_IOU_THRESH,
        "downscale": DEFAULT_DOWNSCALE,
        "monitor_index": DEFAULT_MONITOR_INDEX,
        "dev_overlay": DEFAULT_DEV_OVERLAY,
        "console_status": DEFAULT_CONSOLE_STATUS,
    }

    while True:
        print("\n=== Head Tracking Menü ===")
        print(f"Model: {cfg['model_path']} | conf={cfg['conf']} | ema={cfg['ema_alpha']} | driftN={cfg['drift_check_interval']} | IoU={cfg['iou_thresh']} | downscale={cfg['downscale']}")
        print(f"Monitor: {cfg['monitor_index']} | Overlay={cfg['dev_overlay']} | ConsoleStatus={cfg['console_status']}")
        print("\nAktionen:")
        print("  1) Tracking starten (ohne Overlay)")
        print("  2) Tracking starten (mit Overlay)")
        print("  3) Overlay umschalten (on/off)")
        print("  4) Monitor wechseln")
        print("  5) Model-Pfad ändern")
        print("  6) Downscale ändern (z.B. 1.0 / 2.0)")
        print("  7) Konfidenz ändern (0.1..0.9)")
        print("  8) Beenden")
        choice = input("Auswahl: ").strip()

        if choice == "1":
            cfg["dev_overlay"] = False
            run_tracking(**cfg)
        elif choice == "2":
            cfg["dev_overlay"] = True
            run_tracking(**cfg)
        elif choice == "3":
            cfg["dev_overlay"] = not cfg["dev_overlay"]
        elif choice == "4":
            print("Verfügbare Monitore:")
            list_monitors()
            try:
                idx = int(input("Monitor-Index: ").strip())
                cfg["monitor_index"] = idx
            except Exception:
                print("Ungültiger Index.")
        elif choice == "5":
            p = input("Neuer Model-Pfad: ").strip()
            if p:
                cfg["model_path"] = p
        elif choice == "6":
            try:
                ds = float(input("Downscale (>1.0 = kleiner): ").strip())
                cfg["downscale"] = max(1.0, ds)
            except Exception:
                print("Ungültiger Wert.")
        elif choice == "7":
            try:
                c = float(input("Konfidenz (0.1..0.9): ").strip())
                cfg["conf"] = min(max(0.1, c), 0.9)
            except Exception:
                print("Ungültiger Wert.")
        elif choice == "8":
            print("Tschüss!")
            break
        else:
            print("Unbekannte Auswahl.")


if __name__ == "__main__":
    # Standardmäßig Menü starten, sodass `python main.py` immer das Menü zeigt
    try:
        run_menu()
    except RuntimeError as e:
        print(f"Fehler: {e}")
        print("Hinweis: Installiere opencv-contrib-python und stelle sicher, dass NumPy/Torch kompatibel sind.")
