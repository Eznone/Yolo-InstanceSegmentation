from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile # 👈 Add this import

def track_video(video_path):
    print(f"[DEBUG] Starting track_video with: {video_path}")
    try:
        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video file: {video_path}")
            return None

        track_history = defaultdict(lambda: [])
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps == 0 or frame_width == 0 or frame_height == 0:
            print(f"[ERROR] Invalid video properties.")
            cap.release()
            return None

        # Save output video to the output folder with a unique name
        import os
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{base_name}_tracked.mp4")
        print(f"[DEBUG] Output video will be saved to: {output_path}")

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height)
        )
        
        frame_count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            if frame_count % 30 == 0: # Print less frequently to keep console clean
                print(f"[DEBUG] Processing frame {frame_count}")

            results = model.track(frame, persist=True)
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else None
            
            annotated_frame = results[0].plot()

            if track_ids:
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 30:
                        track.pop(0)

                    points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 0), thickness=2)

            out.write(annotated_frame)

        print(f"[DEBUG] Finished processing {frame_count} frames.")
        cap.release()
        out.release()
        
        print(f"[DEBUG] Returning final output path: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"[ERROR] Exception in track_video: {e}")
        # Clean up resources if an error occurs
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals() and out.isOpened():
            out.release()
        return None