
# defaultdict is used to store tracking history for each object
from collections import defaultdict
# OpenCV for video processing
import cv2
# NumPy for numerical operations
import numpy as np
# YOLO from ultralytics for object detection and tracking
from ultralytics import YOLO


# Function to track objects in a video using YOLO

def track_video(video_path):
    print(f"[DEBUG] Starting track_video with: {video_path}")
    try:
        # Load YOLO model (pretrained weights)
        model = YOLO("yolov8n.pt")

        # Open the video file for reading
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video file: {video_path}")
            return None

        print(f"[DEBUG] Video file opened successfully.")

        # Dictionary to store tracking history for each object
        track_history = defaultdict(lambda: [])

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of video frames
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of video frames
        print(f"[DEBUG] Video properties - FPS: {fps}, Width: {frame_width}, Height: {frame_height}")

        # Check for valid FPS and frame size
        if fps == 0 or frame_width == 0 or frame_height == 0:
            print(f"[ERROR] Invalid video properties. FPS: {fps}, Width: {frame_width}, Height: {frame_height}")
            cap.release()
            return None

        # Define codec and create VideoWriter object to save output video
        output_path = "output_tracked_video.mp4"  # Output video file name
        out = cv2.VideoWriter(
            output_path,  # Output file path
            cv2.VideoWriter_fourcc(*"mp4v"),  # Codec for MP4 format
            fps,  # Frames per second
            (frame_width, frame_height)  # Frame size (width, height)
        )
        print(f"[DEBUG] VideoWriter initialized.")

        frame_count = 0
        # Loop through each frame in the video
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            if not success:
                print(f"[DEBUG] No more frames to read or error reading frame.")
                break

            frame_count += 1
            if frame_count % 10 == 0:
                print(f"[DEBUG] Processing frame {frame_count}")

            # Run YOLO tracking on the current frame
            results = model.track(frame, persist=True)
            # Get bounding box coordinates (center x, center y, width, height)
            boxes = results[0].boxes.xywh.cpu()
            # Get track IDs for detected objects (if available)
            track_ids = (
                results[0].boxes.id.int().cpu().tolist()
                if results[0].boxes.id is not None else None
            )

            # Annotate the frame with detection and tracking info
            annotated_frame = results[0].plot()

            # Plot tracking lines for each tracked object
            if track_ids:
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box  # Unpack box coordinates
                    track = track_history[track_id]  # Get track history for this object
                    track.append((float(x), float(y)))  # Add current position to history
                    if len(track) > 30:
                        track.pop(0)  # Limit history to last 30 positions

                    # Draw the tracking lines on the annotated frame
                    points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        annotated_frame, [points], isClosed=False, color=(0, 255, 0), thickness=2
                    )

            # Write the annotated frame to the output video
            out.write(annotated_frame)
            # Optional: break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print(f"[DEBUG] 'q' pressed, breaking loop.")
                break

        print(f"[DEBUG] Finished processing {frame_count} frames.")
        # releasing resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"[DEBUG] Output video saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"[ERROR] Exception in track_video: {e}")
        return None