# Tracking

This project captures the screen and tracks a player's head and body using [Ultralytics YOLO](https://github.com/ultralytics/ultralytics). It is aimed at games such as Valorant where lightweight tracking of a single target is required.

## Features
- Screen capture via [mss](https://github.com/BoboTiG/python-mss)
- Detection of persons and head boxes using YOLO models
- OpenCV based tracking with optional development overlay
- Automatic calibration and heatmap based ignore mask
- Configurable defaults in `config.py`

## Requirements
Python 3.9+ and the following packages:
```bash
pip install ultralytics mss opencv-contrib-python torch numpy
```

## Usage
Run the interactive menu:
```bash
python main.py
```
From the menu you can start tracking with or without an overlay and adjust parameters such as monitor selection, model path, confidence threshold and more.

## Notes
The application stores calibration data, logs and heatmaps in the user's local data directory (see `utils.get_storage_paths`).
