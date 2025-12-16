import cv2
import numpy as np
import face_recognition
import os
import pickle
from flask import Flask, render_template, Response, jsonify
from datetime import datetime

app = Flask(__name__)

# Load known face encodings
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "face_encodings.pkl"
ATTENDANCE_FILE = "attendance.csv"

def load_known_faces():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            known_faces = pickle.load(f)
            return known_faces["encodings"], known_faces["names"]
    return [], []

known_face_encodings, known_face_names = load_known_faces()

# Initialize video capture (global)
video_capture = None

def generate_frames():
    global video_capture
    video_capture = cv2.VideoCapture(0)  # Open webcam
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            name = "Unknown"

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            # Draw rectangle and name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    video_capture.release()

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_webcam')
def stop_webcam():
    global video_capture
    if video_capture:
        video_capture.release()
    return "Webcam Stopped"

if __name__ == "__main__":
    app.run(debug=True)
