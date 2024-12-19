import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2

# Flask App Initialization
app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = r'D:\YOLO_CNN_PROJECT\Uploads'  # Uploaded files directory
OUTPUT_FOLDER = r'D:\YOLO_CNN_PROJECT\Output'   # Processed files directory
MODEL_PATH = 'runs/yolo_training/weights/best.pt'    # YOLOv8 Model path
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)
print("YOLOv8 model loaded successfully!")

# Helper function to check allowed file types
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Upload Image
@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return render_template('error.html', error="No file part in the request")

    file = request.files['image']

    if file.filename == '':
        return render_template('error.html', error="No file selected")

    if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        # Save uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_filename = f"processed_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        file.save(input_path)
        print(f"Uploaded image saved to: {input_path}")

        # Process the image
        process_image(input_path, output_path)
        print(f"Processed image saved to: {output_path}")

        # Redirect to success page with filename and is_video flag set to False
        return redirect(url_for('upload_success', processed_filename=output_filename, is_video=False))

    return render_template('error.html', error="Invalid file type. Allowed types: jpg, jpeg, png, bmp")

# Upload Video
@app.route('/upload-video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return render_template('error.html', error="No file part in the request")

    file = request.files['video']

    if file.filename == '':
        return render_template('error.html', error="No file selected")

    if file and allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        # Save uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_filename = f"processed_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        file.save(input_path)
        print(f"Uploaded video saved to: {input_path}")

        # Process the video
        process_video(input_path, output_path)
        print(f"Processed video saved to: {output_path}")

        # Redirect to success page with filename and is_video flag set to True
        return redirect(url_for('upload_success', processed_filename=output_filename, is_video=True))

    return render_template('error.html', error="Invalid file type. Allowed types: mp4, avi, mov, mkv")

# Upload Success Page
@app.route('/upload_success/<processed_filename>')
def upload_success(processed_filename):
    # Check if the processed file is a video or image
    is_video = processed_filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS
    return render_template('upload_success.html', processed_filename=processed_filename, is_video=is_video)

# Route to Serve Processed Files
@app.route('/output/<processed_filename>')
def get_output_file(processed_filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], processed_filename)

# Image Processing Function
def process_image(input_path, output_path):
    frame = cv2.imread(input_path)

    # Perform YOLOv8 inference
    results = model(frame)  # Detect objects
    annotated_frame = results[0].plot()  # Annotate frame

    # Save annotated image
    cv2.imwrite(output_path, annotated_frame)
    print("Image processing completed.")

# Video Processing Function
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    # Validate video capture
    if not cap.isOpened():
        print("Error: Cannot open the input video file.")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Initialize video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Perform YOLOv8 inference
        results = model(frame)  # Detect objects
        annotated_frame = results[0].plot()  # Annotate frame

        # Write annotated frame to output video
        out.write(annotated_frame)

    # Release resources
    cap.release()
    out.release()
    print("Video processing completed.")

# Run the Flask Application
if __name__ == '__main__':
    app.run(debug=True)
