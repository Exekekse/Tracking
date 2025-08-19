"""Application configuration for ScreenTracker v2.

This module collects tunable parameters such as model paths, thresholds
and region definitions.  The initial values mirror the defaults used in
the prototype contained in :mod:`main`.
"""

# Path to the YOLO model weights
MODEL_PATH = "yolov8n.pt"

# Default confidence threshold for detections
CONF_THRES = 0.35

# If True only the class "person" will be tracked
PERSON_ONLY = True

# Monitor to capture (mss numbering, 1 = primary monitor)
MONITOR_INDEX = 1

# Optional capture region.  When ``None`` the full monitor is grabbed.
# Example: {"left": 300, "top": 200, "width": 1280, "height": 720}
CAPTURE_ROI = None

# Regions that should never be tracked.  Each entry is either a rectangle
# ({"type": "rect", "x": .., "y": .., "w": .., "h": ..}) or a polygon
# ({"type": "poly", "points": [(x, y), ...]}).
IGNORE_REGIONS = [
    # Example entries – adapt to your game HUD if required
    # {"type": "rect", "x": 1500, "y": 800, "w": 400, "h": 280},
    # {"type": "poly", "points": [(1600,760), (1910,760), (1910,1070), (1600,1070)]},
]

# Directory that contains rule plug‑ins
RULES_DIR = "rules"

