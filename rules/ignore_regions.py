"""Rule that discards detections within predefined regions."""

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

