import gradio as gr
from pyimagesearch.yolo_tracking import track_video

# Gradio function to process video and return output path


# Gradio passes a dictionary with 'name' (file path) and other metadata for video input.
def process_video(video):
    print(f"Received video input: {video}")  # Debug: print input info
    # If video is a dict (Gradio >= 3.x), get the file path
    if isinstance(video, dict) and 'name' in video:
        video_path = video['name']
    elif isinstance(video, str):
        video_path = video
    else:
        raise ValueError("Unsupported video input format")
    print(f"Processing video at path: {video_path}")  # Debug: print file path
    output_path = track_video(video_path)
    print(f"Output video saved to: {output_path}")  # Debug: print output path
    return output_path

# Gradio UI
iface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload a video for YOLO tracking"),
    outputs=gr.Video(label="Tracked Output Video"),
    title="YOLO Instance Segmentation & Tracking",
    description="Upload a video to run YOLO tracking and get the output video with tracked objects."
)

if __name__ == "__main__":
    iface.launch()
