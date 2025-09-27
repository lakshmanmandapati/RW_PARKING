import cv2
from ultralytics import YOLO

# --- IMPORTANT ---
# Replace 'path/to/your/model.pt' with the actual path to your YOLO model file.
# Example: 'best (2).pt' or 'last (1).pt'
model_path = '/Users/lakshmanmandapati/Downloads/AI SPARK/best(1).pt'

# Replace 'path/to/your/input_video.mp4' with the path to your video clip.
# You can also use other video formats like .avi, .mov, etc.
input_video_path = '/Users/lakshmanmandapati/Downloads/AI SPARK/WhatsApp Video 2025-08-30 at 01.32.26.mp4'

# Optional: Define the path for the output video
output_video_path = 'output_with_detections.mp4'

# Load the YOLO model
try:
    model = YOLO(model_path)
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

# Open the video clip
cap = cv2.VideoCapture(input_video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file at '{input_video_path}'.")
    exit()

# Get video properties to set up the output video writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
# The 'mp4v' codec is widely supported for .mp4 files
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

print(f"Video file opened. Processing frames... Press 'q' to exit.")

# Loop to process the video frame by frame
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    # If a frame was not read, the video is finished
    if not ret:
        print("End of video file. Processing complete.")
        break
    
    # Perform object detection on the frame
    # The `stream=True` parameter makes it more efficient for video streams
    results = model.predict(source=frame, stream=True, verbose=False)
    
    # Iterate through the results and draw bounding boxes on the frame
    for r in results:
        annotated_frame = r.plot()  # This returns the frame with predictions drawn
        
        # Display the resulting frame in a window
        cv2.imshow('Object Detection', annotated_frame)
        
        # Write the annotated frame to the output video file
        out.write(annotated_frame)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User requested exit. Stopping process.")
        break

# Release the video capture and video writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print(f"Script finished. The output video with detections is saved at '{output_video_path}'.")