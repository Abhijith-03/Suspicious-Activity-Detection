from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Use yolov8n.pt (nano version) or another YOLOv8 model

# Train the model
model.train(
    data='data.yaml',   # Path to the data configuration file
    epochs=50,          # Number of training epochs
    imgsz=640,          # Image size for training
    batch=16,           # Batch size
    project='runs',     # Folder to save training runs
    name='yolo_training'  # Sub-folder name for this specific training
)

print("Training completed! Check the 'runs/yolo_training' folder for results.")
