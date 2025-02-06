# Active Face Mask Detection 

## Overview
The Active Face Mask Detection project is a real-time system that monitors individuals in CCTV footage to determine if they are wearing face masks. This project was designed to introduce fundamental machine learning concepts through hands-on experience, covering image processing, computer vision, neural networks, and real-time detection.

### Features

- Preprocessing image data using OpenCV and NumPy
- Building a Convolutional Neural Network (CNN) for mask detection
- Training and evaluating the model using Keras2
- Implementing real-time mask detection with OpenCV
- Optional deployment as a web application


## Project Breakdown

### Week 1: Image Processing & Data Handling

- Manipulated image data using NumPy and OpenCV.
- Converted images to grayscale, resized, and applied transformations.
- Plotted and analyzed data using Matplotlib.

### Week 2: Basics of Neural Networks & CNN
- Learned CNN concepts and implemented a binary classification model.
- Built a CNN using Keras with layers like Conv2D, MaxPooling2D, and Dense.
- Trained the model to classify images as "with mask" or "without mask."

### Week 3: Model Evaluation & Real-Time Detection
- Evaluated model performance using precision, recall, and F1-score.
- Integrated OpenCV’s CascadeClassifier to detect faces in video streams.
- Implemented real-time classification on webcam footage.

### Week 4: Integration & Testing
- Combined all functionalities into a single application.
- Tested the model on various real-world images and video feeds.
- Created project documentation and an optional web-based UI.

### Results & Insights

- Achieved high accuracy in detecting face masks in different lighting conditions and angles.
- Successfully integrated real-time detection with minimal lag.
- Faced challenges in handling occlusions and low-resolution images.
- Explored potential deployment options, including a web app using Flask.

Future Improvements
- Enhance model performance with a larger dataset.
- Improve real-time processing speed for high-resolution footage.
- Deploy as a mobile application for broader accessibility.
