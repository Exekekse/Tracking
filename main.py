# =============================
# README (Kurz)
# =============================
# Projektstruktur (alle Dateien in einem Ordner ablegen):
#
#  main.py                -> Startpunkt (YOLO, Capture, Menü, Regeln, Tracker)
#  config.py              -> Konfiguration (Hotkeys, Klassenfilter, Regionen usw.)
#  detection.py           -> Detection-Datenklasse
#  tracker.py             -> Einfacher Centroid-Tracker mit Trails
#  overlay.py             -> Zeichnen (HUD, Labels, Menü, Klick-Handling)
#  rule_engine.py         -> Lädt/verwaltet Regeln aus dem Ordner ./rules
#  rules/__init__.py      -> leer (macht rules zum Paket)
#  rules/base_rule.py     -> Basisklasse für Regeln
#  rules/ignore_regions.py-> Beispielregel: ignoriert Detections in Maskenbereichen
#  rules/ignore_small_boxes.py -> Beispielregel: ignoriert winzige Boxen
#  rules/example_weapon_hud_color.py -> Beispiel: farbbasierte Ignorier-Regel (Demo)
#
# Start:  python main.py
#
# Steuerung:
#  q/Esc  -> Beenden
#  m      -> Menü ein/aus
#  f      -> Klassenfilter (nur Person) umschalten
#  t      -> Tracker resetten
#  p      -> Screenshot speichern (annotated_YYYYmmdd_HHMMSS.png)
#  k      -> Frame + Pseudo-Labels in ./captures/ ablegen (für Training)
#  r      -> Regeln neu laden (hot reload)
#
# Hinweise:
# - Falls Tasten in OpenCV-Fenster nicht reagieren, einmal ins Fenster klicken, oder Esc nutzen.
# - Menü ist zusätzlich per Maus klickbar.
# - "Waffe nicht tracken": Lege Regionen in config.py (IGNORE_REGIONS) an oder schreibe eine eigene Regel
#   im Ordner ./rules (siehe example_weapon_hud_color.py). Regeln werden zur Laufzeit geladen.
# - Für eigenes Training kannst du mit "k" schnell Datenschnappschüsse + Pseudolabels sammeln und später korrigieren.

# =============================
# main.py
# =============================
from ultralytics import YOLO
import mss
import numpy as np
import cv2
import torch
import time
import os
import signal
from typing import List

import config
from detection import Detection
from tracker import CentroidTracker
from overlay import Drawer, MenuState, KeyHelper
from rule_engine import RuleEngine


def safe_mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def export_for_training(frame_bgr: np.ndarray, detections: List[Detection], names_map: dict):
    """Speichert Frame + YOLO-Labeldatei (Pseudo-Labels) in ./captures/"""
    safe_mkdir("captures")
    ts = time.strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join("captures", f"cap_{ts}.jpg")
    txt_path = os.path.join("captures", f"cap_{ts}.txt")

    # Bild speichern (BGR->JPG)
    cv2.imwrite(img_path, frame_bgr)

    # YOLO-Labels (class cx cy w h) in Normalized Koordinaten
    h, w = frame_bgr.shape[:2]
    with open(txt_path, "w", encoding="utf-8") as f:
        for d in detections:
            cx = (d.x1 + d.x2) / 2.0 / w
            cy = (d.y1 + d.y2) / 2.0 / h
            bw = (d.x2 - d.x1) / w
            bh = (d.y2 - d.y1) / h
            f.write(f"{d.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"[TRAIN] Exportiert: {img_path} + {txt_path}")


def main():
    # Graceful exit via Ctrl+C
    running = True

    def handle_sigint(sig, frame):
        nonlocal running
        print("\n[INFO] SIGINT erhalten. Beende...")
        running = False

    signal.signal(signal.SIGINT, handle_sigint)

    # YOLO laden
    model = YOLO(config.MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Klassen-Namen
    names_raw = model.names
    names = names_raw if isinstance(names_raw, dict) else {i: n for i, n in enumerate(names_raw)}

    # Person-Klasse für Filter finden
    person_id = None
    for k, v in names.items():
        if v.lower() == "person":
            person_id = int(k)
            break

    # State/Helper
    menu_state = MenuState(
        conf_thres=config.CONF_THRES,
        person_only=config.PERSON_ONLY,
        show_trails=True,
        show_hud=True,
    )

    key = KeyHelper()
    drawer = Drawer(names)
    engine = RuleEngine()

    # Regeln initial laden
    engine.load_rules()

    # Tracker
    ct = CentroidTracker(max_disappeared=20, trail_len=24, dist_thresh=80)

    # Bildschirmquelle
    with mss.mss() as sct:
        monitor = sct.monitors[config.MONITOR_INDEX]
        roi = config.CAPTURE_ROI  # None oder dict mit left, top, width, height

        # Fenster initialisieren
        win_name = "ScreenTracker v2"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.resizeWindow(win_name, 1280, 720)

        # Maus für Menü
        def on_mouse(event, x, y, flags, param):
            nonlocal menu_state
            if event == cv2.EVENT_LBUTTONDOWN:
                if drawer.menu_open:
                    drawer.menu_click(x, y, menu_state)

        cv2.setMouseCallback(win_name, on_mouse)

        prev = time.perf_counter()
        fps_smooth = 0.0

        while running:
            # Screenshot BGRA
            if roi:
                frame = np.array(sct.grab(roi))
            else:
                frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Inferenz
            results = model(frame, conf=menu_state.conf_thres, verbose=False)
            r = results[0]

            # Detections extrahieren
            dets: List[Detection] = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), c, cls in zip(xyxy, confs, clss):
                    if menu_state.person_only and person_id is not None and cls != person_id:
                        continue
                    dets.append(Detection(x1, y1, x2, y2, int(cls), float(c)))

            # Regeln anwenden (Blacklist, Regionen, etc.)
            dets = engine.apply(dets, frame)

            # Tracken
            id2box = ct.update([(d.x1, d.y1, d.x2, d.y2) for d in dets])

            # Annotieren
            annotated = frame.copy()
            drawer.draw_scene(
                annotated,
                dets=dets,
                id2box=id2box,
                trails=ct.trails if menu_state.show_trails else {},
                show_hud=menu_state.show_hud,
                fps_smooth=fps_smooth,
                person_only=menu_state.person_only,
                conf_thres=menu_state.conf_thres,
            )

            # Menü ggf. zeichnen (klickbar)
            if drawer.menu_open:
                drawer.draw_menu(annotated, menu_state)

            # Anzeige
            cv2.imshow(win_name, annotated)

            # FPS glätten
            now = time.perf_counter()
            dt = now - prev
            prev = now
            inst = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = (fps_smooth * 0.9) + (inst * 0.1) if fps_smooth > 0 else inst

            # Tasten
            k = cv2.waitKey(1) & 0xFFFFFFFF
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
                elif key.is_reload_rules(k):
                    engine.load_rules()
                elif drawer.menu_open:
                    # Zusätzliche Shortcuts, wenn Menü offen (z.B. +/- für conf)
                    if key.is_plus(k):
                        menu_state.conf_thres = min(0.99, menu_state.conf_thres + 0.02)
                    elif key.is_minus(k):
                        menu_state.conf_thres = max(0.01, menu_state.conf_thres - 0.02)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# =============================
# config.py
# =============================
MODEL_PATH = "yolov8n.pt"
CONF_THRES = 0.35
PERSON_ONLY = True

# Welcher Monitor? 1 = Hauptmonitor (mss-Konvention)
MONITOR_INDEX = 1

# Optional: Nur Ausschnitt capturen, sonst None
# CAPTURE_ROI = {"left": 300, "top": 200, "width": 1280, "height": 720}
CAPTURE_ROI = None

# Regionen, in denen niemals getrackt werden soll (z. B. Waffen-HUD unten rechts)
# Typ: "rect" mit x, y, w, h ODER "poly" mit Punkten [(x,y), ...]
IGNORE_REGIONS = [
    # Beispiel: Rechteck unten rechts (anpassen!)
    # {"type": "rect", "x": 1500, "y": 800, "w": 400, "h": 280},

    # Beispiel Polygon
    # {"type": "poly", "points": [(1600,760), (1910,760), (1910,1070), (1600,1070)]}
]

# Ordner für Regel-Plugins
RULES_DIR = "rules"


# =============================
# detection.py
# =============================
from dataclasses import dataclass

@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    cls: int
    conf: float

    @property
    def cx(self):
        return (self.x1 + self.x2) // 2

    @property
    def cy(self):
        return (self.y1 + self.y2) // 2

    @property
    def wh(self):
        return self.x2 - self.x1, self.y2 - self.y1


# =============================
# tracker.py
# =============================
import numpy as np
from collections import deque
from math import hypot

class CentroidTracker:
    def __init__(self, max_disappeared=30, trail_len=24, dist_thresh=80):
        self.next_id = 0
        self.objects = {}
        self.boxes = {}
        self.disappeared = {}
        self.trails = {}
        self.max_disappeared = max_disappeared
        self.trail_len = trail_len
        self.dist_thresh = dist_thresh

    def register(self, rect):
        x1, y1, x2, y2 = rect
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        oid = self.next_id
        self.next_id += 1
        self.objects[oid] = (cx, cy)
        self.boxes[oid] = rect
        self.disappeared[oid] = 0
        self.trails[oid] = deque(maxlen=self.trail_len)
        self.trails[oid].append((cx, cy))

    def deregister(self, oid):
        self.objects.pop(oid, None)
        self.boxes.pop(oid, None)
        self.disappeared.pop(oid, None)
        self.trails.pop(oid, None)

    def update(self, rects):
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.boxes.copy()

        if len(self.objects) == 0:
            for r in rects:
                self.register(r)
            return self.boxes.copy()

        input_centroids = []
        for (x1, y1, x2, y2) in rects:
            input_centroids.append(((x1 + x2) // 2, (y1 + y2) // 2))

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = np.zeros((len(object_centroids), len(input_centroids)), dtype=float)
        for i, (ocx, ocy) in enumerate(object_centroids):
            for j, (icx, icy) in enumerate(input_centroids):
                D[i, j] = hypot(ocx - icx, ocy - icy)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.dist_thresh:
                continue

            oid = object_ids[row]
            self.objects[oid] = input_centroids[col]
            self.boxes[oid] = rects[col]
            self.disappeared[oid] = 0
            self.trails[oid].append(input_centroids[col])

            used_rows.add(row)
            used_cols.add(col)

        for row in set(range(0, D.shape[0])) - used_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        for col in set(range(0, D.shape[1])) - used_cols:
            self.register(rects[col])

        return self.boxes.copy()


# =============================
# overlay.py
# =============================
import cv2
import time
from typing import Dict, List
from detection import Detection

class KeyHelper:
    def is_quit(self, k):
        return k in (ord('q'), 27)  # 27 = Esc

    def is_toggle_menu(self, k):
        return k == ord('m')

    def is_toggle_filter(self, k):
        return k == ord('f')

    def is_reset_tracker(self, k):
        return k == ord('t')

    def is_screenshot(self, k):
        return k == ord('p')

    def is_capture(self, k):
        return k == ord('k')

    def is_reload_rules(self, k):
        return k == ord('r')

    def is_plus(self, k):
        return k in (ord('+'), ord('='))

    def is_minus(self, k):
        return k in (ord('-'), ord('_'))


def deterministic_color(seed: int):
    return (int((seed * 29) % 255), int((seed * 53) % 255), int((seed * 97) % 255))


def draw_label(img, x, y, text):
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)


def put_hud(img, fps, count, filter_active, conf_thres):
    text_lines = [
        f"FPS: {fps:.1f}",
        f"Tracked: {count}",
        f"Filter: {'person-only' if filter_active else 'all'}",
        f"conf>={conf_thres:.2f}",
        "q/Esc: Quit  m:Menu  f:Filter  t:Reset  p:Shot  k:Capture  r:Reload",
    ]
    x, y = 10, 22
    for t in text_lines:
        (w, h), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x-6, y-16), (x + w + 6, y + 6), (0,0,0), -1)
        cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        y += 28


class MenuState:
    def __init__(self, conf_thres: float, person_only: bool, show_trails: bool, show_hud: bool):
        self.conf_thres = conf_thres
        self.person_only = person_only
        self.show_trails = show_trails
        self.show_hud = show_hud


class Drawer:
    def __init__(self, names_map: dict):
        self.names = names_map
        self.menu_open = False
        self._menu_rects = []  # (x1,y1,x2,y2, key)

    def draw_scene(self, img, dets: List[Detection], id2box: Dict[int, tuple], trails: dict,
                   show_hud: bool, fps_smooth: float, person_only: bool, conf_thres: float):
        # Boxes & IDs
        # Mapping: Zuordnung ID->Detection per Center-Nähe (für Label/Conf)
        def nearest_det(box):
            if not dets:
                return None
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            best, best_d = None, 1e9
            for d in dets:
                d2 = abs(d.cx - cx) + abs(d.cy - cy)
                if d2 < best_d:
                    best_d = d2
                    best = d
            return best

        for oid, (x1, y1, x2, y2) in id2box.items():
            color = deterministic_color(oid + 7)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            d = nearest_det((x1, y1, x2, y2))
            if d is not None:
                label = self.names.get(int(d.cls), str(d.cls))
                text = f"ID {oid} | {label} {d.conf:.2f}"
            else:
                text = f"ID {oid}"
            draw_label(img, x1, y1, text)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(img, (cx, cy), 3, color, -1)
            if oid in trails:
                pts = list(trails[oid])
                for k in range(1, len(pts)):
                    cv2.line(img, pts[k-1], pts[k], color, 2)

        if show_hud:
            put_hud(img, fps_smooth, len(id2box), person_only, conf_thres)

    def draw_menu(self, img, state: MenuState):
        # Halbtransparente Fläche links
        overlay = img.copy()
        panel_w = 360
        panel_h = 240
        x1, y1 = 12, 12
        x2, y2 = x1 + panel_w, y1 + panel_h
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (40, 40, 40), -1)
        img[:] = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

        # Titel
        cv2.putText(img, "MENÜ", (x1 + 16, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

        # Einträge
        entries = [
            ("Person-only", "toggle_person"),
            ("Trails anzeigen", "toggle_trails"),
            ("HUD anzeigen", "toggle_hud"),
            (f"Conf Schwelle: {state.conf_thres:.2f} (+/-)", "noop"),
            ("Regeln neu laden", "reload_rules"),
            ("Tracker resetten", "reset_tracker"),
            ("Screenshot speichern", "save_shot"),
            ("Menü schließen", "close_menu"),
        ]

        self._menu_rects = []
        bx, by = x1 + 16, y1 + 60
        bw, bh = panel_w - 32, 28
        gap = 8
        for text, key in entries:
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (60,60,60), -1)
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (110,110,110), 1)
            cv2.putText(img, text, (bx + 10, by + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            self._menu_rects.append((bx, by, bx + bw, by + bh, key))
            by += bh + gap

        # Statuspunkte
        def dot(val, yoff):
            clr = (0, 200, 0) if val else (0, 0, 200)
            cv2.circle(img, (x2 - 20, y1 + yoff), 7, clr, -1)
        dot(state.person_only, 78)
        dot(state.show_trails, 114)
        dot(state.show_hud, 150)

    def menu_click(self, x, y, state: MenuState):
        for (x1, y1, x2, y2, key) in self._menu_rects:
            if x1 <= x <= x2 and y1 <= y <= y2:
                if key == "toggle_person":
                    state.person_only = not state.person_only
                elif key == "toggle_trails":
                    state.show_trails = not state.show_trails
                elif key == "toggle_hud":
                    state.show_hud = not state.show_hud
                elif key == "reload_rules":
                    import rule_engine
                    rule_engine.RuleEngine().load_rules()  # Nur Trigger; eigentl. handled in main bei Taste 'r'
                elif key == "reset_tracker":
                    pass  # Wird in main über Taste 't' sauber gesetzt
                elif key == "save_shot":
                    # In main umgesetzt (Taste 'p'), hier kein Frame verfügbar
                    pass
                elif key == "close_menu":
                    self.menu_open = False
                break

    def save_frame(self, img):
        ts = time.strftime("%Y%m%d_%H%M%S")
        fn = f"annotated_{ts}.png"
        cv2.imwrite(fn, img)
        print(f"[INFO] Frame gespeichert: {fn}")


# =============================
# rule_engine.py
# =============================
import os
import sys
import importlib
import traceback
from typing import List

import config
from detection import Detection

class RuleEngine:
    def __init__(self):
        self.rules = []  # Instanzen mit .accept(detection, frame)->bool

    def load_rules(self):
        self.rules.clear()
        rules_dir = config.RULES_DIR
        pkg_name = os.path.basename(rules_dir)

        # sicherstellen, dass rules importierbar sind
        if rules_dir not in sys.path:
            sys.path.insert(0, os.path.abspath("."))
        # Paket-Import ("rules")
        try:
            importlib.import_module(pkg_name)
        except Exception:
            print("[RULES] Hinweis: Konnte Paket 'rules' nicht initial importieren (evtl. fehlt __init__.py).")

        # Alle .py-Module in rules laden, außer base_rule und __init__
        if not os.path.isdir(rules_dir):
            print(f"[RULES] Ordner '{rules_dir}' nicht gefunden – keine Regeln aktiv.")
            return

        for fname in os.listdir(rules_dir):
            if not fname.endswith('.py'):
                continue
            modname = fname[:-3]
            if modname in ('__init__', 'base_rule'):
                continue
            fqmn = f"{pkg_name}.{modname}"
            try:
                if fqmn in sys.modules:
                    importlib.reload(sys.modules[fqmn])
                    mod = sys.modules[fqmn]
                else:
                    mod = importlib.import_module(fqmn)
                # Erwartet: build() -> Rule-Instanz
                if hasattr(mod, 'build'):
                    rule = mod.build()
                    self.rules.append(rule)
                    print(f"[RULES] Geladen: {fqmn} ({getattr(rule, 'name', 'unnamed')})")
                else:
                    print(f"[RULES] Übersprungen (keine build()): {fqmn}")
            except Exception:
                print(f"[RULES] Fehler beim Laden von {fqmn}:\n{traceback.format_exc()}")

    def apply(self, dets: List[Detection], frame) -> List[Detection]:
        if not self.rules:
            return dets
        kept = []
        for d in dets:
            ok = True
            for rule in self.rules:
                try:
                    if not rule.accept(d, frame):
                        ok = False
                        break
                except Exception:
                    # Einzelne Regel darf nicht alles stoppen
                    print(f"[RULES] Fehler in Regel {getattr(rule, 'name', 'unknown')} (Detection übersprungen)")
                    ok = False
                    break
            if ok:
                kept.append(d)
        return kept


# =============================
# rules/__init__.py
# =============================
# leer – erforderlich, damit 'rules' als Paket importierbar ist


# =============================
# rules/base_rule.py
# =============================
from detection import Detection

class Rule:
    name = "BaseRule"

    def accept(self, det: Detection, frame) -> bool:
        """True = behalten, False = verwerfen"""
        return True


# =============================
# rules/ignore_regions.py
# =============================
import cv2
import numpy as np
import config
from detection import Detection
from .base_rule import Rule

class IgnoreRegionsRule(Rule):
    name = "IgnoreRegions"

    def __init__(self, regions):
        self.regions = regions

    def _in_rect(self, d: Detection, r):
        cx, cy = d.cx, d.cy
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        return (x <= cx <= x + w) and (y <= cy <= y + h)

    def _in_poly(self, d: Detection, pts):
        cx, cy = d.cx, d.cy
        cnt = np.array(pts, dtype=np.int32)
        # pointPolygonTest: >0 innen, =0 auf Kante, <0 außen
        res = cv2.pointPolygonTest(cnt, (float(cx), float(cy)), False)
        return res >= 0

    def accept(self, det: Detection, frame) -> bool:
        for reg in self.regions:
            if reg.get("type") == "rect":
                if self._in_rect(det, reg):
                    return False
            elif reg.get("type") == "poly":
                if self._in_poly(det, reg.get("points", [])):
                    return False
        return True


def build():
    return IgnoreRegionsRule(config.IGNORE_REGIONS)


# =============================
# rules/ignore_small_boxes.py
# =============================
from detection import Detection
from .base_rule import Rule

class IgnoreSmallBoxes(Rule):
    name = "IgnoreSmallBoxes"

    def __init__(self, min_w=24, min_h=24):
        self.min_w = min_w
        self.min_h = min_h

    def accept(self, det: Detection, frame) -> bool:
        w, h = det.wh
        return (w >= self.min_w) and (h >= self.min_h)


def build():
    # Standardwerte, bei Bedarf hier anpassen
    return IgnoreSmallBoxes(min_w=24, min_h=24)


# =============================
# rules/example_weapon_hud_color.py
# =============================
# Demo-Regel: Ignoriere Detections, wenn der durchschnittliche HSV-Farbton im Boxbereich
# in einem bestimmten Bereich liegt (z. B. typische HUD-Farbe einer Waffe/Overlay).
# Hinweis: Nur als Beispiel – auf deine Spiele/HUD-Farben anpassen!

import cv2
import numpy as np
from detection import Detection
from .base_rule import Rule

class IgnoreByHue(Rule):
    name = "IgnoreByHue"

    def __init__(self, hue_min=20, hue_max=40, sat_min=60, val_min=60):
        self.hue_min = hue_min
        self.hue_max = hue_max
        self.sat_min = sat_min
        self.val_min = val_min

    def accept(self, det: Detection, frame) -> bool:
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        crop = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
        if crop.size == 0:
            return True
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h_mean = int(np.mean(h))
        s_mean = int(np.mean(s))
        v_mean = int(np.mean(v))
        # Wenn sehr farbig (s/v hoch) UND Farbton im Bereich -> ignorieren
        if s_mean >= self.sat_min and v_mean >= self.val_min and self.hue_min <= h_mean <= self.hue_max:
            return False
        return True


def build():
    # Werte an deine HUD-Farbe anpassen
    return IgnoreByHue(hue_min=20, hue_max=40, sat_min=80, val_min=80)
