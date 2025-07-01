import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


class Tracker:
    def __init__(self, max_age: int = 30, nn_budget: int = 100):
        self.tracker = DeepSort(max_age=max_age, nn_budget=nn_budget)

    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        return self.tracker.update(detections, frame)

    def draw_tracks(self, frame: np.ndarray, tracks: np.ndarray, classes: list) -> np.ndarray:
        for track in tracks:
            x1, y1, x2, y2, track_id, cls = track
            label = f"ID: {track_id} {classes[int(cls)]}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

