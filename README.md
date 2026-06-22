# SpatioScribe

## Overview
SpatioScribe is a computer vision pipeline that detects, tracks, and narrates objects across video frames.

It combines:
- YOLOv8 for convolutional object detection
- DeepSORT for persistent multi-object tracking
- BLIP for vision-language scene captioning
- Motion analysis using per-object spatiotemporal history

**Track:** Computer Vision

## Key Idea
The system maintains object identity across frames and uses motion history to describe how objects move over time.

## Features
- Detects common objects such as people, vehicles, and bicycles
- Tracks objects across frames with consistent IDs
- Generates natural language descriptions of scenes
- Detects movement direction (left, right, up, down)

## Pipeline

**Video → YOLO Detection → DeepSORT Tracking → Motion Analysis → NLP → Output**

The system processes the input video frame-by-frame using the following pipeline:

This pipeline ensures that object identity and motion are preserved across frames, enabling more meaningful scene understanding.

1. **Input (main.py)**
   - Reads video frames using OpenCV
   - Controls the overall pipeline execution

2. **Object Detection (detector.py)**
   - Uses YOLOv8 to perform object detection on each frame
   - Outputs bounding boxes, class labels, and confidence scores

3. **Object Tracking (tracker.py)**
   - Uses DeepSORT to assign consistent IDs to detected objects
   - Maintains object identity across frames
   - Stores position history for motion analysis

4. **Motion Analysis (tracker.py + nlp.py)**
   - Uses stored object history to determine movement direction
   - Example: moving left, right, up, or down

5. **Natural Language Processing (nlp.py)**
   - Uses BLIP to generate a caption for selected frames
   - Combines caption with detected objects and motion data
   - Produces human-readable descriptions

6. **Visualization (drawer.py)**
   - Draws bounding boxes and object IDs on frames
   - Displays captions on the video

7. **Output (utils.py + main.py)**
   - Saves annotated video
   - Saves structured JSON data
   - Saves text descriptions

## How to Run

```bash
python main.py --video input.mp4
```


## Demo

### Video 1
- [Processed Output](demos/output_video_for_input.mp4)
- [JSON Output](demos/results_in_json_for_input.json)
- [Description](demos/description_for_input.txt)
- [Original Input](https://www.youtube.com/watch?v=BB35ocRGazk)

### Video 2
- [Processed Output](demos/output_video_for_input1.mp4)
- [JSON Output](demos/results_in_json_for_input1.json)
- [Description](demos/description_for_input1.txt)
- [Original Input](https://www.youtube.com/watch?v=uNN115UfPRY)

The videos are used for educational purposes only.

---

## Example Outputs

The system produces both structured data and natural language descriptions.

### JSON Output (sample)

```json
{
  "frame": 120,
  "timestamp": 4.0,
  "objects": [
    {"id": 1, "label": "car", "bbox": [100, 200, 300, 400]},
    {"id": 2, "label": "person", "bbox": [500, 250, 550, 400]}
  ]
}
```
### Description Output (sample)

At 4.0s: person moving right. (2 cars, 1 person)


---

## All Options

| Flag | Default | Description |
|---|---|---|
| `--video` | `input.mp4` | Path to input video file |
| `--weights` | `yolov8n.pt` | YOLO model weights |
| `--caption-every` | `30` | Generate caption every N frames |
| `--no-nlp` | off | Disable BLIP captioning (faster) |
| `--output-dir` | `output/` | Directory for output files |

---

## Files
```
├── main.py          # Pipeline orchestrator and entry point
├── detector.py      # YOLOv8 object detection
├── tracker.py       # DeepSORT multi-object tracking and motion history
├── nlp.py           # BLIP captioning and scene description generation
├── drawer.py        # Frame annotation and visualization
├── utils.py         # Logging, timestamp conversion, JSON utilities
├── requirements.txt # Python dependencies
└── demos/           # Sample outputs from two test videos
```

---
## Requirements
- Python 3.8+
- CUDA-capable GPU strongly recommended (BLIP captioning is very slow on CPU)
Models are downloaded automatically on first run:
- YOLOv8n (~6 MB, from Ultralytics)
- BLIP image captioning base (~990 MB, from Hugging Face)
---
## Limitations
- Detection is limited to 8 object classes: person, car, bicycle, airplane, train, truck, motorcycle, bus
- BLIP captioning sometimes misidentifies objects (e.g., labels a truck as a car)
- DeepSORT can lose track of objects during fast movement or occlusion
- No real-time processing — designed for offline video analysis