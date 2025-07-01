### Dispatch Monitoring System
This project implements an intelligent monitoring system for a commercial kitchen's dispatch area, using YOLOv5 for object detection, DeepSORT for tracking, and Pydantic for user feedback validation. The system is deployed using Docker Compose for consistent and reproducible environments.

### Features
+ Detects and tracks items (e.g., trays, plates) in video feeds from the dispatch area.
+ Allows user feedback to improve model accuracy. 
+ Stores feedback in a JSON log for model retraining.
+ Deployed via Docker Compose with support for CPU and GPU.
+ Prerequisites
+ Docker: Install Docker and Docker Compose on your system.
+ Docker Installation
+ Docker Compose Installation
+ Dataset: Download the dataset from the provided Google Drive link and place it in a dataset folder.
+ GPU (Optional): NVIDIA GPU with CUDA support for faster inference (requires NVIDIA Container Toolkit).

#2
Detects and tracks items (dishes, trays) in a video feed from the dispatch area.
Supports classification (empty, kakigori, not_empty) and detection (dish, tray).
Collects user feedback to improve model accuracy.
Stores feedback and misclassified frames for retraining.
Deployed with Docker Compose for CPU/GPU environments.

### Project Structure
```
AU_Dispatch_Monitoring_System/
├── src/
│   ├── __init__.py
│   ├── detection.py       # YOLOv5 detection logic
│   ├── tracking.py        # DeepSORT tracking logic
│   ├── feedback.py        # Feedback handling with Pydantic
│   ├── main.py            # Main application script
├── dataset/               # Dataset folder (populated by user)
├── video.mp4             # Input video (populated by user)
├── feedback_log.json     # Feedback log
├── retraining_frames/    # Frames saved for retraining
├── Dockerfile            # Docker image configuration
├── docker-compose.yml    # Docker Compose configuration
├── README.md             # This file
```

#### Dataset structure:
```
AU_Dispatch_Monitoring_System/
├── video.mp4  # Video of restaurant activities
├── dataset/
│   ├── classification/
│   │   ├── dish/
│   │   │   ├── empty/           # Images of empty dishes
│   │   │   ├── kakigori/        # Images of dishes with kakigori
│   │   │   ├── not_empty/       # Images of dishes with other food/objects
│   │   ├── tray/
│   │   │   ├── empty/           # Images of empty trays
│   │   │   ├── kakigori/        # Images of trays with kakigori
│   │   │   ├── not_empty/       # Images of trays with other food/objects
│   ├── detection/
│   │   ├── train/
│   │   │   ├── images/          # Training images of dishes/trays
│   │   │   ├── labels/          # YOLO-format labels (e.g., "0 0.12265625 0.440625 0.2109375 0.440625")
│   │   ├── val/
│   │   │   ├── images/          # Validation images of dishes/trays
│   │   │   ├── labels/          # YOLO-format labels (e.g., "1 0.1765625 0.46171875 0.18359375 0.759375")
│   │   ├── dataset.yaml         # YOLO config (classes: 0: dish, 1: tray)
```

### Prerequisites
+ Docker: Install Docker and Docker Compose. (from website: ../web)
+ Docker Installation
+ Docker Compose Installation
+ Dataset: Download the dataset and video.mp4 from the provided Google Drive link.
+ GPU (Optional): NVIDIA GPU with CUDA and NVIDIA Container Toolkit for faster inference.

### Setup Instructions
Setup Instructions

### NOTE: Please download/extract the AU Dispatch Monitoring System files, such as: video, dataset folder and put it in the repo as the Project structure above (in the root level)

Note: Install Makefile plugin if wanted

Add environment variables:
```
DATASET_PATH = /dataset
FEEDBACK_LOG_PATH = feedback_log.json
DATASET_YAML = /dataset/detection/dataset.yaml
VIDEO_FILE = /1473_CH05_20250501133703_154216.mp4
```

1. Build and Run with Docker Compose

Build the Docker image: run this command
```yaml
docker-compose build --no-cache
```
Run with Docker Compose
```yaml
docker-compose up -d
docker-compose logs dispatch_monitor
```

2. Display Output (Optional)
   To view the video with bounding boxes:

On Linux, enable X11 forwarding:
```
xhost +local:docker
docker-compose up
```
On Windows/Mac, modify src/main.py to save output to a video file using cv2.VideoWriter:
```
out = cv2.VideoWriter('/app/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
out.write(frame)
out.release()
```

3. Provide User Feedback
   + The system logs simulated feedback every 100 frames to feedback_log.json.
   + For real feedback, implement a UI to collect:
   + Frame ID
   + Item ID
   + Label (dish, tray)
   + Classification (empty, kakigori, not_empty)
   + Correctness (True/False)
   + Comments
   + Feedback is stored in feedback_log.json and frames are saved in retraining_frames/.

4. Improve the Model
   Run the system to collect feedback:
```
docker-compose up
```
Misclassified frames are saved in retraining_frames/.
Manually or programmatically update dataset/detection/train/labels/ with new annotations based on feedback.

Fine-tune YOLOv5:
```
docker-compose run dispatch_monitor python yolov5/train.py --data /app/dataset/detection/dataset.yaml --weights yolov5s.pt --epochs 10
```
Update weights to use the fine-tuned model in future runs.

### Notes
GPU Support: Remove the deploy section in docker-compose.yml for CPU-only environments.
DeepSORT: Uses a placeholder repository. Replace with the correct DeepSORT package if needed.
Output: If cv2.imshow fails, save output to /app/output.mp4 (see above).
Dataset: Ensure video.mp4 and dataset/ are in the project root before running.
