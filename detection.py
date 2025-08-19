import cv2
import numpy as np
from ultralytics import YOLO

from utils import xyxy_to_xywh


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


def detect_head_body(frame_bgr: np.ndarray, model: YOLO, conf: float, downscale: float):
    """Detect head and body boxes of the most confident person in the frame.

    Returns a tuple ``(head_xywh, body_xywh)`` where each element may be
    ``None`` if the respective region could not be determined.
    """
    h, w = frame_bgr.shape[:2]
    det_frame = frame_bgr
    scale = 1.0
    if downscale and downscale > 1.0:
        det_frame = cv2.resize(
            frame_bgr,
            (w // int(downscale), h // int(downscale)),
            interpolation=cv2.INTER_LINEAR,
        )
        scale = float(w) / float(det_frame.shape[1])

    results = model(det_frame, conf=conf, verbose=False)

    # Sonderfall: Head-only-Modelle haben nur eine Klasse. Dann haben wir
    # keine Körperbox und übernehmen nur den Kopf.
    is_head_model = False
    try:
        nc = getattr(model.model, "nc", None)
        is_head_model = (nc == 1)
    except Exception:
        pass

    if is_head_model:
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return None, None
        x1, y1, x2, y2 = map(float, r0.boxes.xyxy[0].cpu().numpy())
        x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
        return xyxy_to_xywh(x1, y1, x2, y2), None

    # Standardfall: Person finden und daraus Kopf- und Körperbox ableiten
    person_xyxy = _select_person_box(results)
    if person_xyxy is None:
        return None, None
    x1, y1, x2, y2 = person_xyxy
    x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
    body = xyxy_to_xywh(x1, y1, x2, y2)
    head = _derive_head_from_person(x1, y1, x2, y2)
    return head, body


def detect_head(frame_bgr: np.ndarray, model: YOLO, conf: float, downscale: float):
    """Backward compatible wrapper returning only the head box."""
    head, _ = detect_head_body(frame_bgr, model, conf, downscale)
    return head
