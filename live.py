"""
live.py - real-time SpatioScribe on a live camera.

Output modes (all saved to output/):
  default    detection + tracking + BLIP captions -> annotated video, JSON, description file
  --no-nlp   skip captions                         -> annotated video, JSON
  --lite     skip captions and video (Raspberry Pi) -> JSON only
Add --server URL to also send the JSON records to a remote server (see server/server.py).

Examples
  python live.py --source 0                          # laptop / USB webcam, full pipeline
  python live.py --source 0 --no-nlp --duration 10   # video + JSON, stop after 10 seconds
  python live.py --source picam --lite --no-track    # Raspberry Pi camera, fastest settings
  python live.py --source 0 --lite --server http://192.168.1.10:8000/detections

Press q in the preview window (or Ctrl+C) to stop.
"""
import argparse
import os
import socket
import time

import cv2

import detector
import drawer
import tracker
from uploader import Uploader
from utils import make_frame_entry, save_json, setup_logger


def parse_args():
    p = argparse.ArgumentParser(description="SpatioScribe live camera mode")
    p.add_argument("--source", default="0", help="'picam' for the Pi camera, a number for a webcam, or a video file")
    p.add_argument("--weights", default=detector.DEFAULT_YOLO_WEIGHTS, help="YOLO weights file")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO inference size (use 320 on a Raspberry Pi)")
    p.add_argument("--conf", type=float, default=0.55, help="Minimum detection confidence")
    p.add_argument("--all-classes", action="store_true", help="Detect all 80 COCO classes")
    p.add_argument("--caption-every", type=int, default=30, help="Generate a BLIP caption every N frames")
    p.add_argument("--no-nlp", action="store_true", help="Skip BLIP captions, save video + JSON")
    p.add_argument("--lite", action="store_true", help="Raspberry Pi mode: skip captions and video, save only JSON")
    p.add_argument("--no-track", action="store_true", help="Skip DeepSORT tracking (faster)")
    p.add_argument("--no-show", action="store_true", help="Do not open a preview window")
    p.add_argument("--duration", type=float, default=0, help="Stop after this many seconds (0 = no limit)")
    p.add_argument("--output-dir", default="output", help="Directory for results")
    p.add_argument("--server", help="Also upload results to this URL, e.g. http://<server-ip>:8000/detections")
    p.add_argument("--device", default=socket.gethostname(), help="Device name stored with uploaded results")
    return p.parse_args()


class PiCamera:
    def __init__(self):
        # Imported here so the script still runs on computers without picamera2
        from picamera2 import Picamera2
        self.cam = Picamera2()
        self.cam.configure(self.cam.create_video_configuration(main={"size": (640, 480), "format": "RGB888"}))
        self.cam.start()

    def read(self):
        return True, self.cam.capture_array()

    def release(self):
        self.cam.stop()


def open_source(source):
    if source == "picam":
        return PiCamera()
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {source}")
    return cap


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    use_nlp = not (args.lite or args.no_nlp)
    log = setup_logger(log_file=os.path.join(args.output_dir, "live_description.txt") if use_nlp else None)

    detector.load_model(args.weights)
    if not args.no_track:
        tracker.reset_tracker()
    nlp = None
    if use_nlp:
        import nlp
        nlp.load_captioner()

    uploader = Uploader(args.server, args.device) if args.server else None
    cap = open_source(args.source)
    writer = None
    first_frames = []
    all_frames = []
    frame_idx = 0
    t_start = time.time()
    t_prev = t_start
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.time()
            timestamp = round(now - t_start, 2)
            # Smoothed frames per second, so the number on screen does not jump every frame
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6) if frame_idx else 0.0
            t_prev = now

            detections = detector.detect_objects(frame, args.imgsz, args.conf, args.all_classes)
            if args.no_track:
                tracks = [{"id": i + 1, "label": d["label"], "bbox": d["bbox"]} for i, d in enumerate(detections)]
            else:
                tracks = tracker.update_tracks(detections, frame)
            entry = make_frame_entry(frame_idx, timestamp, tracks)
            all_frames.append(entry)
            if uploader:
                uploader.add(entry)

            if nlp and frame_idx % args.caption_every == 0:
                caption = nlp.generate_caption(frame)
                log.info(nlp.build_description(timestamp, tracks, caption, tracker.get_history()))
                drawer.set_caption(caption)
            elif frame_idx % 30 == 0:
                log.info("%.1fs  %.1f FPS  %s", timestamp, fps, ", ".join(t["label"] for t in tracks) or "nothing detected")

            if not args.lite or not args.no_show:
                annotated = drawer.draw_tracks(frame, tracks)
                cv2.putText(annotated, f"{fps:.1f} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            if not args.lite:
                # Live FPS depends on the device, so measure it over the first 20 frames before creating the video
                if writer is None:
                    first_frames.append(annotated.copy())
                    if len(first_frames) == 20:
                        writer = open_writer(args.output_dir, first_frames, t_start)
                else:
                    writer.write(annotated)

            if not args.no_show:
                cv2.imshow("SpatioScribe live", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if args.duration and time.time() - t_start >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if writer is None and first_frames:
            writer = open_writer(args.output_dir, first_frames, t_start)
        if writer:
            writer.release()
        save_json(all_frames, os.path.join(args.output_dir, "live_results.json"))
        if uploader:
            uploader.close()
        log.info("Saved %d frames to %s", len(all_frames), args.output_dir)


def open_writer(output_dir, frames, t_start):
    fps = max(1.0, min(30.0, len(frames) / max(time.time() - t_start, 1e-6)))
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(os.path.join(output_dir, "live_output.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    return writer


if __name__ == "__main__":
    main()

