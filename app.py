from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model
model = tf.keras.models.load_model('mask_detection_model.h5')  # Path to your trained model

# Class labels
labels = {0: 'No Mask', 1: 'Mask'}

# Function to preprocess frames before prediction
def preprocess_frame(frame):
    img = cv2.resize(frame, (128, 128))  # Resize to match model input size
    img = np.array(img, dtype='float32') / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Video stream generator
def generate_frames():
    cap = cv2.VideoCapture(0)  # Use the default camera (replace 0 with a video file path for CCTV footage)

    while True:
        success, frame = cap.read()  # Read a frame
        if not success:
            break

        # Preprocess frame and predict
        preprocessed_frame = preprocess_frame(frame)
        prediction = model.predict(preprocessed_frame)
        label = labels[int(prediction[0][0] > 0.5)]  # Choose label based on model output

        # Add label to the frame
        color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)
        cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(frame, (10, 10), (200, 60), color, 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')  # Render HTML template

# Route for video stream
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
