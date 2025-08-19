import numpy as np

from detection import Detection
from pose_parts import compute_parts_for_detection, export_parts_labels, rescale_keypoints


def _base_keypoints(conf: float = 1.0):
    k = np.zeros((17, 3), dtype=float)
    k[:, 2] = conf
    # Simple upright pose
    k[0, :2] = (50, 10)   # nose
    k[1, :2] = (45, 9)
    k[2, :2] = (55, 9)
    k[3, :2] = (40, 10)
    k[4, :2] = (60, 10)
    k[5, :2] = (40, 40)  # left shoulder
    k[6, :2] = (60, 40)  # right shoulder
    k[11, :2] = (45, 120)  # left hip
    k[12, :2] = (55, 120)  # right hip
    # remaining points roughly along limbs
    k[13, :2] = (45, 170)
    k[14, :2] = (55, 170)
    k[15, :2] = (45, 200)
    k[16, :2] = (55, 200)
    return k


def make_det(keypoints):
    return Detection(0, 0, 100, 200, 0, 0.9, keypoints.tolist())


def test_compute_parts_full_person():
    k = _base_keypoints()
    det = make_det(k)
    parts = compute_parts_for_detection(det, (200, 100))
    assert parts is not None
    assert parts["head"][3] <= parts["upper_body"][1] + 1
    assert parts["upper_body"][3] <= parts["body"][3]


def test_compute_parts_only_head():
    k = _base_keypoints()
    k[5:, 2] = 0.0  # remove shoulders and below
    det = make_det(k)
    parts = compute_parts_for_detection(det, (200, 100))
    assert parts is not None
    assert "head" in parts and "upper_body" in parts and "body" in parts


def test_compute_parts_missing_shoulder():
    k = _base_keypoints()
    k[5, 2] = 0.0  # left shoulder missing
    det = make_det(k)
    parts = compute_parts_for_detection(det, (200, 100))
    assert parts is not None
    hx1, _, hx2, _ = parts["head"]
    assert hx2 - hx1 > 0


def test_compute_parts_missing_hip():
    k = _base_keypoints()
    k[11, 2] = 0.0
    k[12, 2] = 0.0
    det = make_det(k)
    parts = compute_parts_for_detection(det, (200, 100))
    assert parts is not None
    assert parts["upper_body"][3] > parts["upper_body"][1]


def test_compute_parts_occluded():
    k = _base_keypoints()
    k[0, 2] = 0.0  # nose not visible
    k[6, 2] = 0.0  # right shoulder missing
    k[13:, 2] = 0.0  # legs occluded
    det = make_det(k)
    parts = compute_parts_for_detection(det, (200, 100))
    assert parts is not None
    assert "body" in parts


def test_rescale_keypoints():
    k = np.array([[10.0, 20.0, 1.0]])
    scaled = rescale_keypoints(k, 2.0, 3.0)
    assert np.allclose(scaled[0, :2], [20.0, 60.0])


def test_export_parts_smoke(tmp_path):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    parts = [{"head": (10, 10, 30, 30), "upper_body": (10, 30, 30, 70), "body": (5, 5, 40, 95)}]
    export_parts_labels(frame, parts, tmp_path)
    files = list(tmp_path.iterdir())
    assert any(p.suffix == ".jpg" for p in files)
    txt = next(p for p in files if p.suffix == ".txt")
    data = np.loadtxt(txt)
    assert np.all((0.0 <= data[:, 1:]) & (data[:, 1:] <= 1.0))
