Suspicious Activity Detection using YOLOv8 and Flask

Overview:
This project focuses on detecting suspicious activities in videos using the YOLOv8 (You Only Look Once, version 8) deep learning model. The application is built with a Flask-based web interface, allowing users to upload video files for analysis and receive processed videos with annotated detections.

Features:
- **Object Detection**: Leverages YOLOv8 for real-time object detection in video frames.
- **Video Uploads**: Supports video upload through a user-friendly web interface.
- **Processed Outputs**: Annotated video files are saved and made available for download.
- **File Type Support**: Accepts video formats such as MP4, AVI, MOV, and MKV.
- **Dynamic Web Application**: Built using Flask for seamless interaction.

Tech Stack:
- **Backend**: Flask, Python
- **Model**: YOLOv8 (Ultralytics)
- **Frontend**: HTML, CSS
- **Video Processing**: OpenCV
- **Deployment**: Locally hosted (can be extended to cloud platforms)

Usage:
- Navigate to the Home Page.
- Upload a video or image file.
- Wait for the YOLOv8 model to process the video.
- View or download the processed video with detected suspicious activities.

Results
- Videos uploaded to the application are processed frame by frame using the YOLOv8 model.
- Detected objects and annotations are overlaid on the frames to highlight potential suspicious activities.

Example Use Cases
- **Surveillance Monitoring**: Detecting unusual behavior in CCTV footage.
- **Traffic Surveillance**: Identifying accidents or illegal activities on roads.
- **Public Security**: Monitoring crowded areas for threats.
