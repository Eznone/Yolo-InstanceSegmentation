from pyimagesearch.yolo_tracking import track_video

if __name__ == "__main__":
    # get input video path
    video_path = "./videos/input_video.mp4"
    output_path = track_video(video_path)
    print(f"Tracked video saved at: {output_path}")