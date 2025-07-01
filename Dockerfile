FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    git \
    && rm -rf /var/lib/apt/lists/*

# Update pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --user --no-cache-dir \
    torch==1.13.1 \
    torchvision==0.14.1 \
    opencv-python==4.8.1.78 \
    pydantic==1.10.14 \
    numpy==1.26.4 \
    pyyaml==6.0.2

# Clone the github of YOLOv5 and install dependencies
RUN git clone https://github.com/ultralytics/yolov5.git /yolov5 \
      && cd /yolov5 \
      && git checkout v7.0 \
      && pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Install DeepSORT (from pypi)
RUN git clone https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch.git /deepsort
RUN pip install --no-cache-dir deep-sort-realtime

# Create directory retraining_frames
RUN mkdir "retraining_frames"

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY 1473_CH05_20250501133703_154216.mp4 .
COPY dataset/ ./dataset/
COPY /yolov5 /app/yolov5

# Command to run the application (will be overridden in docker-compose)
CMD ["python", "src/main.py"]