from collections import deque

from deep_sort_realtime.deepsort_tracker import DeepSort


# How many frames a track is kept "alive" after it is no longer detected.
# Example: if a car disappears behind another object for a few frames,
# it will still keep the same ID instead of creating a new one.
# Too LOW - IDs disappear quickly (flickering)
# Too HIGH - "ghost boxes" stay on screen
MAX_AGE = 10


# How many consecutive frames are required before a detection becomes a real tracked object.
# This helps filter out noise (false detections).
# Example: a random false detection appearing for 1 frame won't get an ID.
# Too LOW - noisy detections become tracks
# Too HIGH - real objects appear late
N_INIT = 3


# How similar two bounding boxes must be (based on position) to be considered the same object.
# Internally uses IoU-like distance.
# Lower value - stricter matching (less ID switching, but may lose track)
# Higher value - more flexible (can match even if object moves more)
MAX_IOU_DISTANCE = 0.7


# How similar objects must look (appearance/visual features).
# This is what makes DeepSORT powerful (not just position-based).
# Lower value - stricter identity (better for crowded scenes)
# Higher value - more forgiving (but may confuse similar objects)
MAX_COSINE_DISTANCE = 0.3


# Controls removal of duplicate overlapping detections.
# YOLO already performs NMS, so we usually keep this high.
# Lower it only if you see multiple boxes for the same object.
NMS_MAX_OVERLAP = 1.0


# How many past positions we store for each object.
# Used for motion analysis (direction, speed, etc.).
# Example: detecting "moving left" vs "standing still"
# Larger value - smoother motion estimation
# Smaller value - more responsive but noisier
HISTORY_LENGTH = 8


tracker = None
history = {}


# Initialize / reset tracker
def reset_tracker():
    global tracker, history

    tracker = DeepSort(
        max_age=MAX_AGE,
        n_init=N_INIT,
        max_iou_distance=MAX_IOU_DISTANCE,
        max_cosine_distance=MAX_COSINE_DISTANCE,
        nms_max_overlap=NMS_MAX_OVERLAP
    )

    history = {}


# Update tracks using DeepSORT
# detections: list of {label, bbox, confidence}
# frame: picture that is required for embeddings
def update_tracks(detections, frame):
    ds_detections = []

    # Convert detections to DeepSORT format
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        w = x2 - x1
        h = y2 - y1

        ds_detections.append((
            [x1, y1, w, h],  # bbox format for DeepSORT
            det["confidence"],
            det["label"]
        ))

    tracks_ds = tracker.update_tracks(ds_detections, frame=frame)

    results = []

    for t in tracks_ds:
        if not t.is_confirmed():
            continue

        x1, y1, x2, y2 = map(int, t.to_ltrb())
        results.append({
            "id": t.track_id,
            "label": t.get_det_class(),
            "bbox": [x1, y1, x2, y2]
        })

    record_history(results)
    return results


# Store the latest center of each tracked object for motion analysis (used by every tracker).
# Only objects that are still tracked are kept, so motion of objects that left the scene
# is not described and memory does not grow during long live runs
def record_history(tracks):
    global history

    new_history = {}
    for obj in tracks:
        x1, y1, x2, y2 = obj["bbox"]
        positions = history.get(obj["id"], deque(maxlen=HISTORY_LENGTH))
        positions.append(((x1 + x2) // 2, (y1 + y2) // 2))
        new_history[obj["id"]] = positions

    history = new_history


# Return history (used in NLP)
def get_history():
    return history

