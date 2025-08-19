"""Example rule that ignores detections based on average hue."""

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
        crop = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
        if crop.size == 0:
            return True
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h_mean = int(np.mean(h))
        s_mean = int(np.mean(s))
        v_mean = int(np.mean(v))
        if s_mean >= self.sat_min and v_mean >= self.val_min and self.hue_min <= h_mean <= self.hue_max:
            return False
        return True


def build():
    return IgnoreByHue(hue_min=20, hue_max=40, sat_min=80, val_min=80)

