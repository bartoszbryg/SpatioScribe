from ultralytics import YOLO

# Labels we care about
TARGET_LABELS = {"person", "car", "bicycle", "airplane", "train", "truck", "motorcycle", "bus"}

model = None


# Any Ultralytics detection weights work here (yolov8n.pt, yolo11n.pt, yolo26n.pt, or an exported *_ncnn_model folder)
DEFAULT_YOLO_WEIGHTS = "yolo26n.pt"

def load_model(weights = DEFAULT_YOLO_WEIGHTS):
    global model
    model = YOLO(weights, task="detect")


# Detect objects
# Returns list of dicts: {label, bbox: [x1,y1,x2,y2], confidence}
def detect_objects(frame, imgsz=640, conf=0.55, all_classes=False):
    if model is None:
        raise RuntimeError("Call load_model() before detect_objects()")

    results = model(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    detections = []

    # Go over all detections from a frame and store them as dictionaries in a list
    for box in results.boxes:
        class_id = int(box.cls[0])
        label = model.names[class_id]
        if not all_classes and label not in TARGET_LABELS:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence_score = float(box.conf[0])
        detections.append({"label": label, "bbox": [x1, y1, x2, y2], "confidence": confidence_score})

    return detections


# Detect and track objects in one step with a tracker built into Ultralytics.
# bytetrack and ocsort match boxes by position and motion only (no extra neural network), so they are much faster
# than DeepSORT; botsort also compensates for camera movement, which helps with a moving drone camera.
# Returns list of dicts: {id, label, bbox: [x1,y1,x2,y2]}
def track_objects(frame, imgsz=640, conf=0.55, all_classes=False, tracker="bytetrack"):
    if model is None:
        raise RuntimeError("Call load_model() before track_objects()")

    classes = None if all_classes else [i for i, name in model.names.items() if name in TARGET_LABELS]
    results = model.track(frame, imgsz=imgsz, conf=conf, classes=classes,
                          persist=True, tracker=f"{tracker}.yaml", verbose=False)[0]
    tracks = []
    if results.boxes.id is None:
        return tracks

    for box, track_id in zip(results.boxes, results.boxes.id.int().tolist()):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        tracks.append({"id": track_id, "label": model.names[int(box.cls[0])], "bbox": [x1, y1, x2, y2]})

    return tracks

