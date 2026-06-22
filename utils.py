import json
import logging
import os

def setup_logger(name: str = "video_pipeline", level=logging.INFO,
                 log_file: str | None = None):
    logger = logging.getLogger(name)
    fmt = logging.Formatter("%(message)s")
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)
    if log_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    logger.setLevel(level)
    return logger


def frame_to_seconds(frame_idx, fps):
    return round(frame_idx / fps, 3) if fps > 0 else 0.0


def save_json(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as file:
        json.dump(data, file, indent=2)


def normalize_caption(caption):
    replacements = {
        "man": "person", 
        "woman": "person", 
        "boy": "kid", 
        "girl": "kid"
    }
    return " ".join(replacements.get(w.lower(), w) for w in caption.split())


def make_frame_entry(frame_idx, timestamp, tracks):
    objects = []
    for object in tracks:
        objects.append({
            "id": object["id"],
            "label": object["label"],
            "bbox": object["bbox"]
        })
    return {
        "frame": frame_idx,
        "timestamp": timestamp,
        "objects": objects
    }