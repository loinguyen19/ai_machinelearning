import cv2
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv
from detection import YOLODetector
from tracking import Tracker
from feedback import Feedback, FeedbackLog


class DispatchMonitoringSystem:
    def __init__(self, video_path: str, feedback_log_path: str, dataset_yaml: str):
        self.video_path = video_path
        self.feedback_log_path = feedback_log_path
        self.dataset_yaml = dataset_yaml
        if not os.path.exists(self.dataset_yaml):
            raise FileNotFoundError(f"Dataset YAML not found at {self.dataset_yaml}")
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found at {self.video_path}")
        self.feedback_log = FeedbackLog(self.feedback_log_path)
        self.detector = YOLODetector(data_yaml=self.dataset_yaml)
        self.tracker = Tracker()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        self.output_dir = Path(self.feedback_log_path).parent / "retraining_frames"
        self.output_dir.mkdir(exist_ok=True)

    def process_frame(self, frame: np.ndarray, frame_id: int) -> tuple:
        # Detect objects
        detections = self.detector.detect(frame)
        # Update tracks
        tracks = self.tracker.update(detections, frame)
        # Draw detections and tracks
        frame = self.detector.draw_detections(frame, detections)
        frame = self.tracker.draw_tracks(frame, tracks, self.detector.classes)
        return frame, tracks

    def save_frame_for_retraining(self, frame: np.ndarray, frame_id: int):
        output_path = self.output_dir / f"frame_{frame_id}.jpg"
        cv2.imwrite(str(output_path), frame)

    def run(self):
        frame_id = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process frame
            frame, tracks = self.process_frame(frame, frame_id)

            # Simulate user feedback (replace with UI in production)
            if frame_id % 100 == 0 and len(tracks) > 0:
                track = tracks[0]
                feedback = Feedback(
                    frame_id=frame_id,
                    item_id=int(track[4]),
                    user_label=self.detector.classes[int(track[5])],
                    classification="not_empty",  # Example; adjust based on UI input
                    correct=False,
                    comments="Misclassified as empty"
                )
                self.feedback_log.add_feedback(feedback)
                self.save_frame_for_retraining(frame, frame_id)

            # Save output video (instead of cv2.imshow for Docker compatibility)
            if not hasattr(self, 'out'):
                height, width = frame.shape[:2]
                self.out = cv2.VideoWriter('/app/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
            self.out.write(frame)
            frame_id += 1
        self.cap.release()
        self.out.release()

    def improve_model(self):
        misclassified = self.feedback_log.get_misclassified_samples()
        if not misclassified:
            print("No misclassified samples to improve model.")
            return

        print(f"Found {len(misclassified)} misclassified samples for retraining.")
        for fb in misclassified:
            print(f"Frame {fb.frame_id}: Item {fb.item_id} misclassified as {fb.user_label}/{fb.classification}")
            # In production: Update dataset/detection/labels/ with new annotations

        # Placeholder for retraining
        # os.system(f"python yolov5/train.py --data {self.dataset_yaml} --weights yolov5s.pt --epochs 10")


if __name__ == "__main__":
    # Environment variables
    # Debug paths
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {Path(__file__).parent}")

    # Environment variables with robust fallback
    project_root = Path(__file__).parent.parent if os.path.basename(os.getcwd()) == 'src' else Path(os.getcwd())
    print(f'Project root: {project_root}')

    VIDEO_PATH = os.getenv("VIDEO_FILE", str(project_root / "1473_CH05_20250501133703_154216.mp4"))
    FEEDBACK_LOG_PATH = os.getenv("FEEDBACK_LOG_PATH", str(project_root / "feedback_log.json"))
    DATASET_YAML = os.getenv("DATASET_YAML", str(project_root / "dataset" / "detection" / "dataset.yaml"))
    print(f"VIDEO_PATH: {VIDEO_PATH}")
    print(f"FEEDBACK_LOG_PATH: {FEEDBACK_LOG_PATH}")
    print(f"DATASET_YAML: {DATASET_YAML}")

    # Initialize and run system
    system = DispatchMonitoringSystem(VIDEO_PATH, FEEDBACK_LOG_PATH, DATASET_YAML)
    system.run()
    system.improve_model()

    cap = cv2.VideoCapture(0)
    if cap:
        cap = cv2.VideoCapture(VIDEO_PATH)


