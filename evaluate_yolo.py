from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO('runs/yolo_training/weights/best.pt')  # Replace with the correct path to the best weights

# Evaluate the model on the validation set
metrics = model.val()  # Uses the validation data defined in data.yaml

# Display evaluation metrics
print("\n=== Evaluation Metrics ===")
print(f"mAP50: {metrics.box.map50:.4f}")       # Mean Average Precision at IoU=0.50
print(f"mAP50-95: {metrics.box.map:.4f}")     # Mean Average Precision across IoU thresholds from 0.50 to 0.95

# Access additional metrics
precision = metrics.box.mp  # Mean precision
recall = metrics.box.mr     # Mean recall
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.4f}")         # Mean precision
print(f"Recall: {recall:.4f}")               # Mean recall
print(f"F1 Score: {f1_score:.4f}")           # F1 Score
