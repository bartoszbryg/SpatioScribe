import argparse
import os
import sys

import cv2

import detector
import nlp
import drawer
import tracker
from utils import frame_to_seconds, make_frame_entry, save_json, setup_logger



DEFAULT_VIDEO = "input.mp4"
DEFAULT_CAPTION_EVERY = 30   # generate a BLIP caption every N frames
DEFAULT_YOLO_WEIGHTS = "yolov8n.pt"


def parse_args():
    p = argparse.ArgumentParser(description="Smart Video Understanding System")
    p.add_argument("--video", default=DEFAULT_VIDEO, help="Path to input video")
    p.add_argument("--weights", default=DEFAULT_YOLO_WEIGHTS, help="YOLO weights file")
    p.add_argument("--caption-every", type=int, default=DEFAULT_CAPTION_EVERY,
                   help="Generate NLP caption every N frames")
    p.add_argument("--no-nlp", action="store_true", help="Disable BLIP captioning")
    p.add_argument("--output-dir", default="output", help="Directory for results")
    p.add_argument("--output-description-file", default=None,
                   help="Path to output file with description (default: output/<log_TIMESTAMP>.txt)")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.video):
        print("Video file not found:", args.video)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    out_video_path = os.path.join(args.output_dir, f"output_video_for_{video_name}.mp4")
    out_json_path  = os.path.join(args.output_dir, f"results_in_json_for_{video_name}.json")
    log_file_path  = args.output_description_file or os.path.join(args.output_dir, f"description_for_{video_name}.txt")

    log = setup_logger(log_file=log_file_path)
    log.info("Description file: %s", log_file_path)


    log.info("Loading YOLO %s", args.weights)
    detector.load_model(args.weights)

    if not args.no_nlp:
        log.info("Loading BLIP captioner")
        nlp.load_captioner()

    tracker.reset_tracker()

    # Open video and process it
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Cannot open video: ", args.video)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info("Video: %dx%d, %.1f fps, %d frames", width, height, fps, total)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))


    all_frames = []
    frame_idx = 0
    caption = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_to_seconds(frame_idx, fps)

        # 1. Detect all objects in a frame
        # This will give you something like
        # [
        #   {"label": "person", "bbox": [...], confidence: 0.91},
        #   {"label": "car", "bbox": [...], confidence: 0.85},
        #   ...
        # ]
        detections = detector.detect_objects(frame)

        # 2. Pass detections to update_tracks function in tracker
        # This will keep track of the same objects in frames and return the same list of detections without confidence score
        tracks = tracker.update_tracks(detections, frame)


        # 3. Every N frames write caption
        if not args.no_nlp and frame_idx % args.caption_every == 0:
            caption = nlp.generate_caption(frame)
            history = tracker.get_history()
            description = nlp.build_description(timestamp, tracks, caption, history)
            log.info(description)
            drawer.set_caption(caption)

        # 4. Draw bounding boxes for objects in frame - create visualization
        annotated = drawer.draw_tracks(frame, tracks)
        writer.write(annotated)

        # 5. Save JSON record
        all_frames.append(make_frame_entry(frame_idx, timestamp, tracks))

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} / {total} frames.")


    cap.release()
    writer.release()

    save_json(all_frames, out_json_path)
    print("Done. Output video:", out_video_path)
    print("JSON results:", out_json_path)
    print("Description file:", log_file_path)
    return 0


if __name__ == "__main__":
    main()