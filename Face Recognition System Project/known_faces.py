import face_recognition
import os

known_face_encodings = []
known_face_names = []

# Path to the folder containing images of known faces
known_faces_dir = "known_faces"

# Loop through all images in the "known_faces" folder
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        print(f"🔍 Processing file: {filename}")
        
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        
        # Get face encoding
        encodings = face_recognition.face_encodings(image)
        
        if encodings:
            encoding = encodings[0]  # Only take the first face encoding
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])  # Use the filename (without extension) as the name
            print(f"✅ Encoding added for: {filename}")
        else:
            print(f"❌ No face detected in: {filename}")

print("✅ Known faces loaded successfully!")

