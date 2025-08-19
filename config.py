DEFAULT_MODEL_PATH = "yolov8n.pt"   # Falls du ein Head-Model hast: yolov8n-head.pt
DEFAULT_CONF = 0.5                  # YOLO Konfidenz
DEFAULT_EMA_ALPHA = 0.30            # Glättung (0..1) – höher = snappier
DEFAULT_DRIFT_CHECK_INTERVAL = 60   # alle N Frames Drift prüfen
DEFAULT_IOU_THRESH = 0.50           # Re-Init wenn IoU darunter fällt
DEFAULT_DOWNSCALE = 1.0             # 1.0=volle Auflösung; >1.0 = kleinere Detektion
DEFAULT_MONITOR_INDEX = 1           # mss Monitor-Index (1 = Hauptmonitor)
DEFAULT_DEV_OVERLAY = False         # Overlay/Window
DEFAULT_CONSOLE_STATUS = True       # Konsolen-Statusausgabe bei Overlay=False

# Zusatz-Parameter für verbessertes Tracking
DEFAULT_AREA_RATIO = 0.18           # max. zulässige Box-Fläche relativ zum Frame
DEFAULT_VIEWMODEL_Y = 0.80          # Unterer Bereich, in dem Viewmodel erwartet wird
DEFAULT_HEAD_AR = (0.7, 1.6)        # Aspect-Ratio-Gate Kopf (w/h)
DEFAULT_BODY_AR = (0.4, 0.9)        # Aspect-Ratio-Gate Körper (w/h)
DEFAULT_LOCK_FRAMES = 3             # Frames bis Lock aktiviert wird
DEFAULT_HEATMAP_GRID = (64, 36)     # Auflösung der Ignore-Heatmap
DEFAULT_HEATMAP_INTERVAL = 500      # N Frames bis Heatmap-Auswertung
DEFAULT_STORAGE_LIMIT_MB = 50       # Max. Speicher für Daten+Logs
DEFAULT_TARGET_FPS = 60             # gewünschte Mindest-FPS
