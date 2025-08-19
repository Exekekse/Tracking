import sys, os
sys.path.append(os.path.dirname(__file__) + "/..")

from detection import Detection
from pose_parts import compute_parts_for_detection, rescale_keypoints, export_parts_for_training


def _blank_keypoints():
    return [(0.0, 0.0, 0.0) for _ in range(17)]


def test_full_person():
    kps = _blank_keypoints()
    kps[0] = (100.0, 10.0, 1.0)  # nose
    kps[5] = (90.0, 40.0, 1.0)
    kps[6] = (110.0, 40.0, 1.0)
    kps[11] = (95.0, 80.0, 1.0)
    kps[12] = (105.0, 80.0, 1.0)
    det = Detection(80, 0, 120, 100, 0, 0.9, kps)
    parts = compute_parts_for_detection(det, (100, 200))
    assert parts is not None
    assert parts["head"][3] <= parts["upper_body"][1] <= parts["upper_body"][3]


def test_missing_shoulder():
    kps = _blank_keypoints()
    kps[0] = (100.0, 10.0, 1.0)
    kps[5] = (90.0, 40.0, 1.0)
    kps[11] = (95.0, 80.0, 1.0)
    kps[12] = (105.0, 80.0, 1.0)
    det = Detection(80, 0, 120, 100, 0, 0.9, kps)
    parts = compute_parts_for_detection(det, (100, 200))
    assert parts is not None
    assert parts["head"][0] < parts["head"][2]


def test_missing_hip():
    kps = _blank_keypoints()
    kps[0] = (100.0, 10.0, 1.0)
    kps[5] = (90.0, 40.0, 1.0)
    kps[6] = (110.0, 40.0, 1.0)
    det = Detection(80, 0, 120, 100, 0, 0.9, kps)
    parts = compute_parts_for_detection(det, (100, 200))
    assert parts is not None
    # fallback bottom around 60
    assert 55 <= parts["upper_body"][3] <= 65


def test_head_only():
    kps = _blank_keypoints()
    kps[0] = (100.0, 10.0, 1.0)
    det = Detection(80, 0, 120, 100, 0, 0.9, kps)
    parts = compute_parts_for_detection(det, (100, 200))
    assert parts is not None
    assert parts["head"][1] == 10


def test_rescale_keypoints():
    kps = [(10.0, 20.0, 0.9)]
    scaled = rescale_keypoints(kps, 2.0, 3.0)
    assert scaled[0][0] == 20.0 and scaled[0][1] == 60.0


def test_export_smoke(tmp_path):
    import pytest
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    kps = _blank_keypoints()
    kps[0] = (100.0, 10.0, 1.0)
    kps[5] = (90.0, 40.0, 1.0)
    kps[6] = (110.0, 40.0, 1.0)
    kps[11] = (95.0, 80.0, 1.0)
    kps[12] = (105.0, 80.0, 1.0)
    det = Detection(80, 0, 120, 100, 0, 0.9, kps)
    det.parts = compute_parts_for_detection(det, (100, 200))
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    export_parts_for_training(frame, [det], tmp_path)
    files = list(tmp_path.iterdir())
    assert any(f.suffix == ".jpg" for f in files)
    txts = [f for f in files if f.suffix == ".txt"]
    assert txts
    content = txts[0].read_text().strip().split()
    for v in content[1:]:
        assert 0.0 <= float(v) <= 1.0
