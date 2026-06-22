import cv2
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from utils import normalize_caption

processor = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"


# Load Blip Processor and train it with already pretrained data using Salesforce
def load_captioner(model_name = "Salesforce/blip-image-captioning-base"):
    global processor, model
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()


# Return caption for the given frame with RGB rectangles
def generate_caption(frame):
    if model is None:
        return ""
    # Get each pixel from the frame and same it in array in rgb format
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        # max_new_tokens indicates what is the maximum length for the caption
        out = model.generate(**inputs, max_new_tokens=100)
    return processor.decode(out[0], skip_special_tokens=True)


# Combine BLIP caption and build the structured, readable description for the frame
def build_description(timestamp, tracks, caption, history):
    counts = {}
    for obj in tracks:
        counts[obj["label"]] = counts.get(obj["label"], 0) + 1

    detail = ", ".join(f"{n} {lbl}" for lbl, n in counts.items())
    
    movements = describe_movement(history)
    movement_text = ""
    if movements:
        movement_text = "Objects are " + ", ".join(set(movements))
    
    caption = normalize_caption(caption)
    base = caption.capitalize().rstrip(".")
    if movement_text and detail:
        return f"At {timestamp:.1f}s: {base}. {movement_text}. ({detail})"
    elif detail:
        return f"At {timestamp:.1f}s: {base}. ({detail})"
    else:
        return f"At {timestamp:.1f}s: {base}."


# Describe movement for objects
def describe_movement(history):
    movements = []

    for obj_id, positions in history.items():
        if len(positions) < 2:
            continue

        x1, y1 = positions[0]
        x2, y2 = positions[-1]

        dx = x2 - x1
        dy = y2 - y1

        if abs(dx) > abs(dy):
            if dx > 10:
                movements.append("moving right")
            elif dx < -10:
                movements.append("moving left")
        else:
            if dy > 10:
                movements.append("moving down")
            elif dy < -10:
                movements.append("moving up")

    return movements