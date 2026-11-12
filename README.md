# VideoSense AI — Video Intelligence & Analytics Dashboard

## Overview

VideoSense AI is a video intelligence and analytics dashboard built with **Streamlit, YOLO11, and OpenCV**. It analyzes uploaded videos, detects objects across sampled frames, and converts detection results into interactive visual reports.

The application provides an interactive dashboard for exploring object distributions, activity over time, annotated video frames, and AI-assisted questions about video analysis results.

## Features

- **Video Upload & Processing:** Upload a video and analyze it using the YOLO11 object detection model.
- **Object Detection:** Identify objects such as people, cars, bicycles, buses, trucks, and other supported COCO classes.
- **Overview Dashboard:** View key metrics, including total detections, processed frames, object classes, video duration, and processing time.
- **Detection Analysis:** Explore object counts, class distributions, percentages, and downloadable CSV reports.
- **Timeline Analytics:** Examine detection activity over time, inspect individual seconds, and compare two object classes using interactive charts.
- **Annotated Frame Gallery:** View selected video frames with detected objects and bounding-box annotations.
- **AI Assistant:** Ask questions about detection results, object counts, activity peaks, and trends. The assistant uses direct data-based responses and can use the Groq API for open-ended analysis.
- **CSV Export:** Download detection reports and timeline data for further analysis.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web dashboard |
| YOLO11n | Object detection |
| OpenCV | Video processing and frame handling |
| Pandas | Data processing and analysis |
| Plotly Express | Interactive visualizations |
| Plotly Graph Objects | Timeline and comparison charts |
| Groq API | AI-assisted responses |
| NumPy | Numerical operations |

## How It Works

1. **Upload a Video:** The user uploads a video through the dashboard.
2. **Video Processing:** OpenCV reads the video and extracts its properties, including frame rate, frame count, and duration.
3. **Frame Sampling:** The application samples approximately one frame per second.
4. **Object Detection:** YOLO11n processes the sampled frames and identifies objects.
5. **Detection Aggregation:** Detected object classes and their counts are collected and organized into structured data.
6. **Timeline Generation:** Detection information is recorded against video timestamps for temporal analysis.
7. **Visualization:** Streamlit and Plotly display the results through interactive dashboards, charts, tables, and an annotated frame gallery.
8. **AI-Assisted Interaction:** Users can ask questions about the analyzed video through the AI Assistant.

## Dashboard Sections

### 1. Overview
Displays the overall video analysis, including total detections, processed frames, detected object classes, video duration, processing time, and object distribution.

### 2. Detection Analysis
Provides detailed object-class statistics, filtering options, percentage distributions, and downloadable CSV reports.

### 3. Timeline
Provides temporal analysis of detected objects, including:
- Detection activity at a selected second.
- Comparison between two object classes.
- Interactive line charts.
- Combined detection activity over time.
- Downloadable timeline data.

### 4. Gallery
Displays selected video frames with object detection annotations and timestamps.

### 5. AI Assistant
Allows users to ask questions about the analyzed video. It first uses available detection statistics and timeline data for direct answers. For open-ended interpretation, it can use the Groq API when configured.

## Installation and Setup

### Prerequisites

- Python 3.10 or later
- pip
- VS Code or another Python IDE
- An internet connection for installing dependencies and downloading model weights when needed

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd VideoSense-AI
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you do not have a requirements file yet, install the packages with:

```bash
pip install streamlit ultralytics opencv-python pandas plotly groq
```

### 4. Configure the Groq API (Optional)

The AI Assistant can use the Groq API for open-ended questions.

Set your API key as an environment variable in PowerShell:

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
```

Alternatively, configure `GROQ_API_KEY` in Streamlit secrets.

The dashboard's direct data-based question handling can still work without Groq.

### 5. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the dashboard.

## Project Structure

```text
VideoSense-AI/
│
├── app.py
├── requirements.txt
├── detection_data.json       # Optional sample detection data
├── yolo11n.pt                # YOLO weights; downloaded when needed
├── README.md
└── venv/                     # Virtual environment (optional)
```

## Model and Processing Details

- **Detection Model:** YOLO11n
- **Object Categories:** COCO-supported object classes
- **Frame Sampling:** Approximately one frame per second
- **Keyframe Gallery:** Up to 12 annotated frames
- **Video Processing:** OpenCV
- **Visualization:** Plotly interactive charts

The application counts detections across sampled frames. These counts represent model detections, not necessarily unique physical objects, because object tracking and identity association are not implemented.

## Limitations

- Detection accuracy depends on video quality, lighting, camera angle, and object visibility.
- The application samples approximately one frame per second rather than processing every frame through the detection model.
- Detection counts may include repeated detections of the same physical object across different sampled frames.
- AI-generated interpretations may require verification against the underlying detection data.
- Groq-powered responses require a valid API key and an available API connection.

## Future Enhancements

- Real-time video stream analysis.
- Object tracking and unique-object counting.
- Speed and movement estimation.
- Traffic density and congestion analysis.
- Advanced detection and reporting features.
- Exportable PDF analytics reports.
- Deployment as a publicly accessible web application.



---

*VideoSense AI — Turning video data into meaningful insights.*
