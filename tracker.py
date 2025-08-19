"""A simple centroid tracker used for associating detections over time."""

from collections import deque
from math import hypot
from typing import Dict, List, Tuple

import numpy as np


class CentroidTracker:
    """Tracks objects based on the distance between bounding box centroids."""

    def __init__(self, max_disappeared: int = 30, trail_len: int = 24, dist_thresh: float = 80):
        self.next_id = 0
        self.objects: Dict[int, Tuple[int, int]] = {}
        self.boxes: Dict[int, Tuple[int, int, int, int]] = {}
        self.disappeared: Dict[int, int] = {}
        self.trails: Dict[int, deque] = {}
        self.max_disappeared = max_disappeared
        self.trail_len = trail_len
        self.dist_thresh = dist_thresh

    def register(self, rect: Tuple[int, int, int, int]):
        x1, y1, x2, y2 = rect
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        oid = self.next_id
        self.next_id += 1
        self.objects[oid] = (cx, cy)
        self.boxes[oid] = rect
        self.disappeared[oid] = 0
        self.trails[oid] = deque(maxlen=self.trail_len)
        self.trails[oid].append((cx, cy))

    def deregister(self, oid: int):
        self.objects.pop(oid, None)
        self.boxes.pop(oid, None)
        self.disappeared.pop(oid, None)
        self.trails.pop(oid, None)

    def update(self, rects: List[Tuple[int, int, int, int]]):
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.boxes.copy()

        if len(self.objects) == 0:
            for r in rects:
                self.register(r)
            return self.boxes.copy()

        input_centroids = []
        for (x1, y1, x2, y2) in rects:
            input_centroids.append(((x1 + x2) // 2, (y1 + y2) // 2))

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = np.zeros((len(object_centroids), len(input_centroids)), dtype=float)
        for i, (ocx, ocy) in enumerate(object_centroids):
            for j, (icx, icy) in enumerate(input_centroids):
                D[i, j] = hypot(ocx - icx, ocy - icy)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.dist_thresh:
                continue
            oid = object_ids[row]
            self.objects[oid] = input_centroids[col]
            self.boxes[oid] = rects[col]
            self.disappeared[oid] = 0
            self.trails[oid].append(input_centroids[col])
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(D.shape[0])) - used_rows
        unused_cols = set(range(D.shape[1])) - used_cols

        for row in unused_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        for col in unused_cols:
            self.register(rects[col])

        return self.boxes.copy()

