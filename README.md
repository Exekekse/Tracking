# ScreenTracker v2

This demo captures a screen region, runs YOLO detection and optional pose
segmentation and renders tracked objects.

## Pose based body parts

* Activate by setting `ENABLE_POSE_PARTS=True` in `config.py`.
* Toggle overlay with **b** or via menu entry "Körperteile anzeigen".
* Export part boxes with **o** into `captures_parts/` using class ids:
  * `0` – head
  * `1` – upper_body
  * `2` – body
* Images and labels are saved in YOLO format (`<class> <cx> <cy> <w> <h>` normalised).

### Hotkeys

`b` parts overlay, `o` export parts labels, `r` reload rules, `q`/Esc quit,
`m` menu.
