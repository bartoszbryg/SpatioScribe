from ultralytics import YOLO

# Labels we care about
TARGET_LABELS = {"person", "car", "bicycle", "airplane", "train", "truck", "motorcycle", "bus"}

model = None


DEFAULT_YOLO_WEIGHTS = "yolo26n.pt"

def load_model(weights = DEFAULT_YOLO_WEIGHTS):
    global model
    model = YOLO(weights, task="detect")


# Detect objects
# Returns list of dicts: {label, bbox: [x1,y1,x2,y2], confidence}
def detect_objects(frame, imgsz=640, conf=0.4, all_classes=False):
    if model is None:
        print("Call load_model() before detect_objects()")

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