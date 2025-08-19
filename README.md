# ScreenTracker v2

Dieses Projekt demonstriert einfaches Bildschirm‑Tracking mit YOLOv8. 
Es enthält einen interaktiven HUD, Menü mit Hotkeys sowie eine 
regelbasierte Filterung und Exportfunktionen.

## Körperteile über Pose‑Schätzung

Wenn `ENABLE_POSE_PARTS=True` (Standard) wird automatisch ein
YOLOv8‑Pose‑Modell (`POSE_MODEL_PATH`) geladen und für Personen eine
robuste Aufteilung in **head**, **upper_body** und **body** berechnet.
Die farbigen Overlays lassen sich mit der Taste `b` oder über das Menü
("Körperteile anzeigen") ein‑ und ausschalten.

Die aktuelle Aktivierung wird im HUD als `Parts: ON/OFF` angezeigt.

### Export für Training

Mit `o` wird ein Screenshot samt Labeldatei im Verzeichnis
`captures_parts/` gespeichert. Die Labeldatei folgt dem YOLO‑Format:

```
<class> <cx> <cy> <w> <h>
```

Die Klassenbelegung lautet: `0=head`, `1=upper_body`, `2=body`.
Alle Werte sind normalisiert auf [0,1] bezogen auf das Originalbild.
Fehlen Keypoints, wird der Export mit einer Warnung abgebrochen.

### Einschränkungen

- Das Pose‑Modell erkennt ausschließlich Personen.
- Bei starken Okklusionen oder extremen Posen können die Heuristiken
  ungenaue Boxen liefern.
- Wird das Pose‑Modell nicht gefunden, bleibt die Anwendung nutzbar,
  jedoch ohne Teile‑Overlay.

Weitere Hotkeys und Optionen sind in `overlay.put_hud` dokumentiert.
