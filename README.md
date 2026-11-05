# SpatioScribe

## Overview
SpatioScribe is a computer vision pipeline that detects, tracks, and narrates objects across video frames.

It combines:
- YOLO (YOLO26 by default; YOLO11 and YOLOv8 also supported) for convolutional object detection
- DeepSORT for persistent multi-object tracking
- BLIP for vision-language scene captioning
- Motion analysis using per-object spatiotemporal history

**Track:** Computer Vision

## Key Idea
The system maintains object identity across frames and uses motion history to describe how objects move over time.

## Features
- Detects common objects such as people, vehicles, and bicycles (or all 80 COCO classes with `--all-classes` in live mode)
- Tracks objects across frames with consistent IDs
- Generates natural language descriptions of scenes
- Detects movement direction (left, right, up, down)
- Runs on recorded videos or live cameras (laptop webcam, USB camera, Raspberry Pi camera)

## Pipeline

**Video → YOLO Detection → DeepSORT Tracking → Motion Analysis → NLP → Output**

The system processes the input video frame-by-frame using the following pipeline:

This pipeline ensures that object identity and motion are preserved across frames, enabling more meaningful scene understanding.

1. **Input (main.py / live.py)**
   - Reads frames from a video file (`main.py`) or a live camera (`live.py`) using OpenCV
   - Controls the overall pipeline execution

2. **Object Detection (detector.py)**
   - Uses YOLO26 (or any Ultralytics YOLO weights) to perform object detection on each frame
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
pip install -r requirements.txt
```

### Recorded video

```bash
python main.py --video input.mp4
```

### Live camera

```bash
python live.py --source 0 --all-classes                  # laptop / USB webcam, full pipeline
python live.py --source 0 --no-nlp --duration 10         # video + JSON, stop after 10 seconds
python live.py --source picam --lite --imgsz 320 --tracker bytetrack   # Raspberry Pi camera, JSON only
```

Press `q` in the preview window (or `Ctrl+C`) to stop. Results go to `output/`:

Live cameras and streams are read in a background thread, so frame capture overlaps with detection and the newest frame is always processed; video files are read frame by frame.

### Tracking options

| `--tracker` | How it matches objects | Speed | Best for |
|---|---|---|---|
| `deepsort` (default) | Motion + appearance (extra neural network per object) | Slowest, ~13 s warm-up on a Pi | Laptop, scenes where similar objects cross paths |
| `bytetrack` | Motion only | Fastest | Raspberry Pi |
| `ocsort` | Motion only, better with erratic motion and occlusion | Fast | Raspberry Pi, fast-moving objects |
| `botsort` | Motion + camera-motion compensation | Fast, extra CPU for motion compensation | Moving drone camera |
| `none` | No IDs | Fastest | Only "what is in view" |

`bytetrack`, `ocsort` and `botsort` are built into Ultralytics (8.4.63+) and need the small `lap` package.

| Mode | Captions | `live_output.mp4` | `live_results.json` | `live_description.txt` |
|---|---|---|---|---|
| default | yes | yes | yes | yes |
| `--no-nlp` | no | yes | yes | no |
| `--lite` | no | no | yes | no |

### Sending results to a remote server

`live.py` can also upload its JSON records to a server that stores them in a database. Records are sent in batches from a background thread, so a slow or lost connection does not slow down detection; the local JSON is always saved as a backup.

Start the server (on your laptop or a remote machine):

```bash
pip install -r server/requirements.txt
uvicorn server.server:app --host 0.0.0.0 --port 8000
```

Then point `live.py` at it (use the server's IP address; both devices must be able to reach each other):

```bash
python live.py --source picam --lite --server http://<server-ip>:8000/detections
```

The server writes one row per detected object to `detections.db` (SQLite) with `session_id`, `device`, `frame`, `video_time`, `wall_time`, `track_id`, `label`, and box coordinates `x1, y1, x2, y2`. View the latest rows in a browser at `http://<server-ip>:8000/detections`, or interactive API docs at `http://<server-ip>:8000/docs`.

### Switching YOLO versions

Any Ultralytics detection weights work; they are downloaded automatically on first use:

```bash
python live.py --source 0 --weights yolo26n.pt   # default, fastest on CPU
python live.py --source 0 --weights yolo11n.pt
python live.py --source 0 --weights yolov8n.pt
```

### Raspberry Pi

Use Raspberry Pi OS (64-bit). Use the nano model, a small inference size, and `--lite` (BLIP does not fit in a Pi's memory).

```bash
sudo apt install -y python3-picamera2 python3-venv git
git clone https://github.com/bartoszbryg/SpatioScribe.git && cd SpatioScribe
python3 -m venv --system-site-packages venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU-only build (the default one pulls GBs of CUDA packages)
pip install ultralytics deep-sort-realtime "lap>=0.5.12" "setuptools<81"

# recommended: export to NCNN for ~2.5x faster inference on ARM CPUs
yolo export model=yolo26n.pt format=ncnn imgsz=320
python live.py --source picam --weights yolo26n_ncnn_model --imgsz 320 --lite --tracker bytetrack --no-show
```

If `pip` fails with "No space left on device", prefix the commands with `TMPDIR=~/piptmp` (Raspberry Pi OS keeps `/tmp` in RAM).

**Benchmark** (Raspberry Pi 4 Model B Rev 1.5, 2 GB RAM, Raspberry Pi OS 64-bit, Python 3.13, PyTorch 2.15 CPU, Ultralytics 8.4.202; YOLO26n at 320 px on `demos/output_video_for_input.mp4`, `--lite`, steady-state FPS after warm-up):

| Model | With DeepSORT | Without tracking |
|---|---|---|
| PyTorch | 3.6 FPS | 3.7 FPS |
| NCNN | 9.4 FPS | 9.4 FPS |

`--system-site-packages` lets the virtual environment use the system `picamera2` package. Without a monitor, use `--no-show` and read the JSON results, or run with `DISPLAY=:0` to show the preview on a screen attached to the Pi.


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

### main.py (recorded video)

| Flag | Default | Description |
|---|---|---|
| `--video` | `input.mp4` | Path to input video file |
| `--weights` | `yolo26n.pt` | YOLO model weights |
| `--caption-every` | `30` | Generate caption every N frames |
| `--no-nlp` | off | Disable BLIP captioning (faster) |
| `--output-dir` | `output/` | Directory for output files |

### live.py (live camera)

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | `picam` for the Pi camera, a number for a webcam, or a video file |
| `--weights` | `yolo26n.pt` | YOLO weights (`.pt` or an exported `_ncnn_model` folder) |
| `--imgsz` | `640` | YOLO inference size (use `320` on a Raspberry Pi) |
| `--conf` | `0.55` | Minimum detection confidence |
| `--all-classes` | off | Detect all 80 COCO classes |
| `--caption-every` | `30` | Generate a caption every N frames |
| `--no-nlp` | off | Skip captions; save video + JSON |
| `--lite` | off | Skip captions and video; save JSON only (Raspberry Pi) |
| `--tracker` | `deepsort` | `deepsort`, `bytetrack`, `ocsort`, `botsort`, or `none` (see Tracking options) |
| `--no-show` | off | Run without a preview window |
| `--duration` | `0` | Stop after N seconds (0 = run until stopped) |
| `--output-dir` | `output/` | Directory for results |
| `--server` | off | Also upload results to this URL (e.g. `http://<server-ip>:8000/detections`) |
| `--device` | computer name | Device name stored with uploaded results |

---

## Files
```
├── main.py          # Pipeline orchestrator for recorded videos
├── live.py          # Real-time pipeline for webcams and the Raspberry Pi camera
├── uploader.py      # Background upload of results to a remote server
├── server/          # FastAPI server that stores uploaded results in SQLite
├── detector.py      # YOLO object detection
├── tracker.py       # DeepSORT tracking and motion history (ByteTrack/OC-SORT/BoT-SORT run through detector.py)
├── nlp.py           # BLIP captioning and scene description generation
├── drawer.py        # Frame annotation and visualization
├── utils.py         # Logging, timestamp conversion, JSON utilities
├── requirements.txt # Python dependencies
└── demos/           # Sample outputs from two test videos
```

---
## Requirements
- Python 3.10+
- CUDA-capable GPU strongly recommended (BLIP captioning is very slow on CPU)
Models are downloaded automatically on first run:
- YOLO26n (a few MB, from Ultralytics)
- BLIP image captioning base (~990 MB, from Hugging Face)
---
## Limitations
- By default, detection is limited to 8 object classes (person, car, bicycle, airplane, train, truck, motorcycle, bus); use `--all-classes` in live mode for all 80 COCO classes
- BLIP captioning sometimes misidentifies objects (e.g., labels a truck as a car)
- Trackers can lose or swap IDs during fast movement or occlusion, especially at the low frame rates of a Raspberry Pi
- Real-time speed depends on hardware: smooth on a laptop, about 9 FPS on a Raspberry Pi 4 with NCNN
- BLIP captioning is too heavy for a Raspberry Pi