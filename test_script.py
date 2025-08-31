# test_script.py
import argparse
import time
import os
from pyimagesearch.yolo_tracking import track_video

# 1. Set up argument parser
parser = argparse.ArgumentParser(description="Test the YOLO video tracking function.")
parser.add_argument("-v", "--video", type=str, required=True, help="Path to the input video file.")
args = vars(parser.parse_args())

# 2. Run the tracking function
print(f"[INFO] Processing video: {args['video']}")
start_time = time.time()

# This is the same function you were testing in Gradio
output_video_path = track_video(args['video'])

end_time = time.time()
print(f"[INFO] Processing finished in {end_time - start_time:.2f} seconds.")

# 3. Print the result and open the file
if output_video_path:
    print(f"[INFO] Output video saved to: {output_video_path}")
    # Optional: Automatically open the video file after processing (for Windows)
    try:
        os.startfile(output_video_path)
    except AttributeError:
        print("[INFO] 'os.startfile' is not available on this system. Please open the file manually.")
else:
    print("[ERROR] Video processing failed.")