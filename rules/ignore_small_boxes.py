"""Rule that ignores detections that are too small."""

from detection import Detection
from .base_rule import Rule


class IgnoreSmallBoxes(Rule):
    name = "IgnoreSmallBoxes"

    def __init__(self, min_w: int = 24, min_h: int = 24):
        self.min_w = min_w
        self.min_h = min_h

    def accept(self, det: Detection, frame) -> bool:
        w, h = det.wh
        return (w >= self.min_w) and (h >= self.min_h)


def build():
    return IgnoreSmallBoxes(min_w=24, min_h=24)

