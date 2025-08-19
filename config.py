DEFAULT_MODEL_PATH = "yolov8n.pt"   # Falls du ein Head-Model hast: yolov8n-head.pt
DEFAULT_CONF = 0.5                  # YOLO Konfidenz
DEFAULT_EMA_ALPHA = 0.30            # Glättung (0..1) – höher = snappier
DEFAULT_DRIFT_CHECK_INTERVAL = 60   # alle N Frames Drift prüfen
DEFAULT_IOU_THRESH = 0.50           # Re-Init wenn IoU darunter fällt
DEFAULT_DOWNSCALE = 1.0             # 1.0=volle Auflösung; >1.0 = kleinere Detektion
DEFAULT_MONITOR_INDEX = 1           # mss Monitor-Index (1 = Hauptmonitor)
DEFAULT_DEV_OVERLAY = False         # Overlay/Window
DEFAULT_CONSOLE_STATUS = True       # Konsolen-Statusausgabe bei Overlay=False
