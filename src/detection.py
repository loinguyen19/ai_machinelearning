import torch
import cv2
import numpy as np
from pathlib import Path
import os
import yaml
import certifi
import ssl


class YOLODetector:
    def __init__(self, weights: str = "yolov5s.pt", data_yaml: str = None):
        project_root = Path(__file__).parent.parent if os.path.basename(os.getcwd()) == 'src' else Path(os.getcwd())
        self.data_yaml = os.getenv("DATASET_YAML", data_yaml or str(project_root / "dataset" / "detection" / "dataset.yaml"))
        if not os.path.exists(self.data_yaml):
            raise FileNotFoundError(f"YOLO dataset YAML not found at {self.data_yaml}")
        # Create SSL context with certifi certificates
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        # Load YOLOv5 model with custom SSL context
        self.model = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path=weights,
            force_reload=False,  # Avoid re-downloading unless necessary
            trust_repo=True,    # Trust the repository
            _ssl=ssl_context    # Use certifi SSL context
        )
        self.model.eval()
        self.classes = self._load_classes(self.data_yaml)

    def _load_classes(self, yaml_path: str) -> list:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data['names']

    def detect(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(frame)
        return results.xyxy[0].cpu().numpy()

    def draw_detections(self, frame: np.ndarray, detections: np.ndarray) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            label = f"{self.classes[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

