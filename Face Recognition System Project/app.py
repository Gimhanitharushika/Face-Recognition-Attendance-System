import cv2
import numpy as np
import face_recognition
import os
import base64
from PIL import Image
from io import BytesIO
from flask import Flask, render_template, Response, jsonify, request
from datetime import datetime

app = Flask(__name__)

# Path for known faces
KNOWN_FACES_DIR = "known_faces"
attendance_data = []

# Ensure the directory exists
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Load known faces from the directory
def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])  # Name from file name

    print(f"Loaded {len(known_face_names)} known faces.")
    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_known_faces()

# Webcam Initialization
video_capture = None

def start_webcam():
    """Start the webcam only when needed."""
    global video_capture
    if video_capture is None or not video_capture.isOpened():
        video_capture = cv2.VideoCapture(0)

def stop_webcam():
    """Release the webcam properly."""
    global video_capture
    if video_capture is not None:
        video_capture.release()
        video_capture = None

def generate_frames():
    global attendance_data
    recognized_names = set()

    start_webcam()  # Ensure webcam starts

    while True:
        if video_capture is None or not video_capture.isOpened():
            break  

        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            name = "Unknown"

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            if name != "Unknown" and name not in recognized_names:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                attendance_data.append({"name": name, "date": current_time.split()[0], "time": current_time.split()[1]})
                recognized_names.add(name)

            # Draw green box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    stop_webcam()  # Stop webcam before loading the main page
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    start_webcam()  # Ensure webcam starts
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam_route():
    stop_webcam()
    return jsonify({"message": "Webcam stopped successfully."})

@app.route('/get_attendance')
def get_attendance():
    return jsonify(attendance_data)

@app.route('/registration')
def registration():
    stop_webcam()  # Stop the recognition webcam before starting the new one
    return render_template('registration.html')

@app.route('/register', methods=["POST"])
def register():
    global known_face_encodings, known_face_names  

    try:
        data = request.json
        print("Received Data:", data)  # Debugging: Print the received data
        
        name = data.get("name")
        image_data = data.get("image")

        if not name or not image_data:
            return jsonify({"message": "Invalid data"}), 400

        # Remove base64 prefix
        image_data = image_data.replace("data:image/png;base64,", "")

        # Decode and save image
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.png")
        image.save(image_path)

        # Reload known faces
        known_face_encodings, known_face_names = load_known_faces()

        return jsonify({"message": "Registration successful! Face saved."}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)