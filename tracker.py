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


# Minimum confidence for detections from YOLO.
# Removes weak/noisy detections before tracking.
# Too LOW - noisy boxes, unstable tracking
# Too HIGH - missing real objects
CONFIDENCE_THRESHOLD = 0.4


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
    global history

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

        track_id = t.track_id
        x1, y1, x2, y2 = map(int, t.to_ltrb())
        label = t.get_det_class()

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if track_id not in history:
            history[track_id] = deque(maxlen=HISTORY_LENGTH)

        history[track_id].append((center_x, center_y))

        results.append({
            "id": track_id,
            "label": label,
            "bbox": [x1, y1, x2, y2]
        })

    return results


# Return history (used in NLP)
def get_history():
    return history