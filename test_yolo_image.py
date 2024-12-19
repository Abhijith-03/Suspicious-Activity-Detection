import os
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load the trained model
model = YOLO('runs/yolo_training/weights/best.pt')  # Replace with your trained weights path

# Path to the folder containing test images
folder_path = r"D:\YOLO_CNN_PROJECT\Dataset\test\images"  # Ensure path consistency
output_folder = r"D:\YOLO_CNN_PROJECT\outputs"  # Custom output folder

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate over all images in the folder
for filename in os.listdir(folder_path):
    if filename.endswith((".jpg", ".png", ".jpeg")):  # Check for valid image file extensions
        image_path = os.path.join(folder_path, filename)  # Construct the full path
        results = model(image_path)  # Perform detection
        
        # Process each result object in results list
        for i, result in enumerate(results):
            # Get the annotated image (NumPy array)
            annotated_image = result.plot()  # Annotate the image

            # Convert NumPy array to PIL Image
            pil_image = Image.fromarray(annotated_image)
            
            # Save the annotated image
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_{i}.jpg")
            pil_image.save(output_path)  # Save the image

print(f"Annotated images saved in {output_folder}")



