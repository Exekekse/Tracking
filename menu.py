import mss

from config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_CONF,
    DEFAULT_EMA_ALPHA,
    DEFAULT_DRIFT_CHECK_INTERVAL,
    DEFAULT_IOU_THRESH,
    DEFAULT_DOWNSCALE,
    DEFAULT_MONITOR_INDEX,
    DEFAULT_DEV_OVERLAY,
    DEFAULT_CONSOLE_STATUS,
)
from tracking import run_tracking


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
