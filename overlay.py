"""Drawing utilities and interactive HUD/menu implementation."""

from __future__ import annotations

import time
from typing import Dict, List

import cv2

from detection import Detection


class KeyHelper:
    """Utility to check pressed keys returned by :func:`cv2.waitKey`."""

    def is_quit(self, k: int) -> bool:
        return k in (ord("q"), 27)  # 27 = ESC

    def is_toggle_menu(self, k: int) -> bool:
        return k == ord("m")

    def is_toggle_filter(self, k: int) -> bool:
        return k == ord("f")

    def is_reset_tracker(self, k: int) -> bool:
        return k == ord("t")

    def is_screenshot(self, k: int) -> bool:
        return k == ord("p")

    def is_capture(self, k: int) -> bool:
        return k == ord("k")

    def is_reload_rules(self, k: int) -> bool:
        return k == ord("r")

    def is_plus(self, k: int) -> bool:
        return k in (ord("+"), ord("="))

    def is_minus(self, k: int) -> bool:
        return k in (ord("-"), ord("_"))


def deterministic_color(seed: int) -> tuple[int, int, int]:
    return (
        int((seed * 29) % 255),
        int((seed * 53) % 255),
        int((seed * 97) % 255),
    )


def draw_label(img, x: int, y: int, text: str):
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def put_hud(img, fps: float, count: int, filter_active: bool, conf_thres: float):
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
        cv2.rectangle(img, (x - 6, y - 16), (x + w + 6, y + 6), (0, 0, 0), -1)
        cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28


class MenuState:
    def __init__(self, conf_thres: float, person_only: bool, show_trails: bool, show_hud: bool):
        self.conf_thres = conf_thres
        self.person_only = person_only
        self.show_trails = show_trails
        self.show_hud = show_hud


class Drawer:
    """Responsible for rendering detections, HUD and the menu."""

    def __init__(self, names_map: dict):
        self.names = names_map
        self.menu_open = False
        self._menu_rects = []  # type: List[tuple[int, int, int, int, str]]

    def draw_scene(
        self,
        img,
        dets: List[Detection],
        id2box: Dict[int, tuple],
        trails: dict,
        show_hud: bool,
        fps_smooth: float,
        person_only: bool,
        conf_thres: float,
    ):
        # Boxes & IDs
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
                    cv2.line(img, pts[k - 1], pts[k], color, 2)

        if show_hud:
            put_hud(img, fps_smooth, len(id2box), person_only, conf_thres)

    def draw_menu(self, img, state: MenuState):
        overlay = img.copy()
        panel_w = 360
        panel_h = 240
        x1, y1 = 12, 12
        x2, y2 = x1 + panel_w, y1 + panel_h
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (40, 40, 40), -1)
        img[:] = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

        cv2.putText(
            img,
            "MENÜ",
            (x1 + 16, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

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
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (110, 110, 110), 1)
            cv2.putText(img, text, (bx + 10, by + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            self._menu_rects.append((bx, by, bx + bw, by + bh, key))
            by += bh + gap

        def dot(val, yoff):
            clr = (0, 200, 0) if val else (0, 0, 200)
            cv2.circle(img, (x2 - 20, y1 + yoff), 7, clr, -1)

        dot(state.person_only, 78)
        dot(state.show_trails, 114)
        dot(state.show_hud, 150)

    def menu_click(self, x: int, y: int, state: MenuState):
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

                    rule_engine.RuleEngine().load_rules()
                elif key == "reset_tracker":
                    pass
                elif key == "save_shot":
                    pass
                elif key == "close_menu":
                    self.menu_open = False
                break

    def save_frame(self, img):
        ts = time.strftime("%Y%m%d_%H%M%S")
        fn = f"annotated_{ts}.png"
        cv2.imwrite(fn, img)
        print(f"[INFO] Frame gespeichert: {fn}")

