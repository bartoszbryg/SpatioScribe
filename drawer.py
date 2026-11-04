import cv2

from utils import normalize_caption

# Caption text rendered at bottom of frame
pending_caption = ""


# Based on object id, pick a fixed color in RGB so the same object always has the same color
def color_for_id(obj_id):
    obj_id = int(obj_id)

    return (
        (obj_id * 20) % 256,
        (obj_id * 40) % 256,
        (obj_id * 60) % 256
    )


# Set caption for a frame
def set_caption(text):
    global pending_caption
    pending_caption = text


# Draw each tracked object on the frame based on its position saved in tracks
# My tracks are saved in list of {id, label, bbox: [x1,y1,x2,y2]}, frame is the picture for a specific frame
# Then, return the frame which is the picture
def draw_tracks(frame, tracks):
    for obj in tracks:
        x1, y1, x2, y2 = obj["bbox"]
        # Get the fixed color for this object id
        color = color_for_id(obj["id"])
        # Draw rectangle for this object which will show its boundaries
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{obj['label']} #{obj['id']}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.55, 1)
        # Label background
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_COMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Overlay caption at bottom
    if pending_caption:
        h, w = frame.shape[:2]
        margin = 8
        caption_text = normalize_caption(pending_caption)
        (cw, ch), _ = cv2.getTextSize(caption_text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
        cv2.rectangle(frame, (margin, h - ch - margin * 2), (margin + cw + 4, h - margin), (0, 0, 0), -1)
        cv2.putText(frame, caption_text, (margin + 2, h - margin - 2),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 255, 200), 1, cv2.LINE_AA)

    return frame
