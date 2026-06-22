from ultralytics import YOLO

# Labels we care about
TARGET_LABELS = {"person", "car", "bicycle", "airplane", "train", "truck", "motorcycle", "bus"}

model = None


DEFAULT_YOLO_WEIGHTS = "yolov8n.pt"

def load_model(weights = DEFAULT_YOLO_WEIGHTS):
    global model
    model = YOLO(weights)


# Detect objects
# Returns list of dicts: {label, bbox: [x1,y1,x2,y2], confidence}
def detect_objects(frame):
    if model is None:
        print("Call load_model() before detect_objects()")

    results = model(frame, verbose=False)[0]
    detections = []

    # Go over all detections from a frame and store them as dictionaries in a list
    for box in results.boxes:
        class_id = int(box.cls[0])
        label = model.names[class_id]
        if label not in TARGET_LABELS:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence_score = float(box.conf[0])
        if confidence_score < 0.4:
            continue
        detections.append({"label": label, "bbox": [x1, y1, x2, y2], "confidence": confidence_score})

    return detections