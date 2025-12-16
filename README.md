# Face Recognition Attendance System

## Description
This project is a Face Recognition–based Attendance System developed as a university project.
The system automates the traditional attendance process by identifying individuals using
facial recognition techniques and recording attendance automatically, reducing manual effort
and human errors.

The system captures facial data through a live camera feed, detects faces, compares them with
registered face data, and marks attendance for recognized individuals. Attendance records are
stored for future reference.

## Problem Statement
Manual attendance systems are time-consuming and prone to errors. This project aims to provide
an automated and efficient solution using computer vision and facial recognition technology.

## Project Scope
- Face detection and recognition
- Automated attendance marking
- Real-time face recognition
- Storage of attendance records

## Technologies Used
- Python
- OpenCV
- NumPy
- Face Recognition Library
- Flask (if web interface is used)

## Features
- Detects faces from live camera feed
- Recognizes registered individuals
- Automatically marks attendance
- Stores attendance records in CSV format
- Simple and user-friendly interface

## Project Structure
- `app.py` – Main application file
- `face_recognition_live.py` – Live face recognition logic
- `known_faces.py` – Handling registered face data
- `known_faces/` – Sample images of registered users
- `templates/` – HTML files for web interface
- `static/` – CSS and frontend assets
- `attendance.csv` – Sample attendance output

## How to Run
1. Install Python on your system
2. Create and activate a virtual environment
3. Install required dependencies using:

## Output
The system recognizes faces in real time and automatically records attendance.
Attendance data is saved in a CSV file for later review and analysis.

## Limitations
- Accuracy depends on lighting conditions
- Requires clear and frontal face images
- Performance may reduce with a large dataset
- Designed mainly for academic demonstration purposes

## Future Improvements
- Integration with a database system
- Improved accuracy using deep learning models
- Support for multiple cameras
- Cloud-based attendance management

## Author
Gimhani Tharushika
