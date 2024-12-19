import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO('runs/yolo_training/weights/best.pt')  # Replace with your trained weights path

# Paths for the input and output videos
input_video_path = r'D:\YOLO_CNN_PROJECT\Dataset\test\Test_1.mp4'  # Path to the input video
output_video_path = 'D:/YOLO_CNN_PROJECT/outputs/annotated_video.mp4'  # Path to save the output video

# Open the input video
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Unable to open input video file!")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Input Video - Resolution: {width}x{height}, FPS: {fps}")

# Define the codec and create the VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can try 'XVID' or 'mp4v' depending on your system
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

if not out.isOpened():
    print("Error: Unable to initialize VideoWriter!")
    cap.release()
    exit()

# Process each frame from the input video
while True:
    ret, frame = cap.read()  # Read the next frame
    if not ret:
        print("Finished processing all frames.")
        break

    # Perform detection on the frame
    results = model(frame)

    # Annotate the frame with detection results
    annotated_frame = results[0].plot()  # Get the frame with bounding boxes

    # Write the annotated frame to the output video
    out.write(annotated_frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Annotated video saved to {output_video_path}")
