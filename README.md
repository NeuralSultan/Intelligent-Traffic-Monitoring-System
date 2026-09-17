# Intelligent Traffic Monitoring & Violation Detection System

An AI-powered traffic monitoring system for detecting, tracking, and analyzing vehicles in road traffic videos.

The system combines object detection, multi-object tracking, vehicle counting, road calibration, direction estimation, speed estimation, traffic-light detection, and traffic violation analysis into a unified video processing pipeline.

## Features

* Vehicle detection using YOLO
* Multi-object tracking using ByteTrack
* Vehicle counting
* Vehicle direction estimation
* Road calibration
* Wrong-way driving detection
* Vehicle speed estimation
* Speed-limit violation detection
* Traffic-light detection
* Red-light violation detection
* Automatic violation evidence collection
* Centralized violation management
* **Streamlit web interface — currently under development**

## System Pipeline

The system processes traffic videos through a unified pipeline:

```text
Input Video
    │
    ▼
Vehicle Detection
    │
    ▼
ByteTrack Tracking
    │
    ├── Vehicle Counting
    ├── Direction Estimation
    ├── Speed Estimation
    │
    ▼
Violation Detection
    ├── Wrong-Way Detection
    ├── Speed Violation
    └── Red-Light Violation
    │
    ▼
Evidence Collection
    │
    ▼
Output Video / Violation Evidence
```

## Project Structure

```text
Intelligent Traffic Monitoring & Violation Detection System/
│
├── app/
│
├── configs/
│   ├── calibrations/
│   ├── road_config.json
│   ├── road_config_loader.py
│   └── traffic_config.py
│
├── data/
│   └── raw/
│
├── models/
│   ├── detection/
│   └── traffic_sign/
│
├── outputs/
│   ├── evidence/
│   ├── results/
│   └── training/
│
├── src/
│   ├── analytics/
│   ├── core/
│   ├── detection/
│   ├── evidence/
│   ├── pipeline/
│   ├── tracking/
│   └── violations/
│
├── tests/
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── yolo11n.pt
└── yolo26n.pt
```

> The Streamlit interface is currently under development and will be added to the `app/` directory.

## Technologies

* Python
* PyTorch
* Ultralytics YOLO
* OpenCV
* NumPy
* ByteTrack

## Models

The project uses YOLO-based models for vehicle and traffic-sign detection.

The final vehicle detection model is located at:

```text
models/detection/final_yolo11n.pt
```

The repository also contains the base YOLO models used during training.

## Installation

Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd "Intelligent Traffic Monitoring & Violation Detection System"
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The datasets used during development and testing are **not included in this repository**.

They are kept locally and excluded through `.gitignore` because of their size and licensing/distribution considerations.

Place the required datasets and test videos inside the corresponding local `data/` directories before running the project.

## Configuration

Traffic and road-related settings are stored in:

```text
configs/
```

The main traffic configuration is:

```text
configs/traffic_config.py
```

Road-specific calibration settings are stored under:

```text
configs/calibrations/
```

These configurations control parameters such as:

* Detection confidence
* Image size
* Vehicle classes
* Speed measurement lines
* Reference distance
* Speed limit
* Road type

## Running the Pipeline

The main traffic processing pipeline is implemented in:

```text
src/pipeline/traffic_pipeline.py
```

The pipeline can be imported and used from a Python script:

```python
from src.pipeline.traffic_pipeline import TrafficPipeline

pipeline = TrafficPipeline(
    video_path="path/to/video.mp4",
    road_type="two_way",
    enabled_violations=[
        "wrong_way",
        "speeding",
        "red_light",
    ],
)

pipeline.run_video(display=True)
```

Replace the video path with the path to your local traffic video.

## Output

The system can generate:

* Processed traffic videos
* Vehicle tracking information
* Violation records
* Violation evidence images
* Training and evaluation outputs

Generated files are kept locally and are excluded from Git where appropriate.

## Limitations

The accuracy of direction, speed, and traffic-violation detection depends on factors such as:

* Camera position
* Video resolution
* Road configuration
* Vehicle visibility
* Lighting conditions
* Calibration accuracy

Speed estimation in particular depends on the configured road reference distance and measurement lines.

## License

This project is licensed under the MIT License.

See the `LICENSE` file for more information.
