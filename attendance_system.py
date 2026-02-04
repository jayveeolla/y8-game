from flask import Flask, render_template, request, jsonify, Response
import cv2
import face_recognition
import numpy as np
import os
import json
from datetime import datetime
import base64

app = Flask(__name__)

# Directories
KNOWN_FACES_DIR = 'known_faces'
ATTENDANCE_FILE = 'attendance_records.json'

# Create directories if they don't exist
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Load known faces
known_face_encodings = []
known_face_names = []

def load_known_faces():
    """Load all known faces from the known_faces directory"""
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    
    if os.path.exists(KNOWN_FACES_DIR):
        for filename in os.listdir(KNOWN_FACES_DIR):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(KNOWN_FACES_DIR, filename)
                
                try:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    
                    if encodings:
                        known_face_encodings.append(encodings[0])
                        known_face_names.append(name)
                        print(f"Loaded face: {name}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

def load_attendance_records():
    """Load attendance records from JSON file"""
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'r') as f:
            return json.load(f)
    return []

def save_attendance_record(name):
    """Save attendance record to JSON file"""
    records = load_attendance_records()
    
    # Check if already marked today
    today = datetime.now().strftime('%Y-%m-%d')
    for record in records:
        if record['name'] == name and record['date'] == today:
            return False  # Already marked today
    
    # Add new record
    new_record = {
        'name': name,
        'date': today,
        'time': datetime.now().strftime('%H:%M:%S'),
        'timestamp': datetime.now().isoformat()
    }
    records.append(new_record)
    
    with open(ATTENDANCE_FILE, 'w') as f:
        json.dump(records, f, indent=4)
    
    return True

# Load known faces at startup
load_known_faces()

@app.route('/')
def index():
    return render_template('attendance.html')

@app.route('/register')
def register_page():
    return render_template('register_face.html')

@app.route('/records')
def records_page():
    return render_template('attendance_records.html')

@app.route('/api/register_face', methods=['POST'])
def register_face():
    """Register a new face"""
    try:
        data = request.get_json()
        name = data.get('name')
        image_data = data.get('image')
        
        if not name or not image_data:
            return jsonify({'success': False, 'message': 'Name and image required'})
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Find face encodings
        face_locations = face_recognition.face_locations(rgb_image)
        
        if not face_locations:
            return jsonify({'success': False, 'message': 'No face detected in image'})
        
        if len(face_locations) > 1:
            return jsonify({'success': False, 'message': 'Multiple faces detected. Please ensure only one face is visible'})
        
        # Save image
        filename = f"{name}.jpg"
        filepath = os.path.join(KNOWN_FACES_DIR, filename)
        cv2.imwrite(filepath, image)
        
        # Reload known faces
        load_known_faces()
        
        return jsonify({'success': True, 'message': f'{name} registered successfully!'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/recognize_face', methods=['POST'])
def recognize_face():
    """Recognize face and mark attendance"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'Image required'})
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Find faces
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        if not face_encodings:
            return jsonify({'success': False, 'message': 'No face detected'})
        
        # Check each face
        recognized_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            
            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    
                    # Mark attendance
                    if save_attendance_record(name):
                        recognized_names.append({'name': name, 'status': 'Attendance marked'})
                    else:
                        recognized_names.append({'name': name, 'status': 'Already marked today'})
        
        if recognized_names:
            return jsonify({'success': True, 'recognized': recognized_names})
        else:
            return jsonify({'success': False, 'message': 'Face not recognized'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/attendance_records')
def get_attendance_records():
    """Get all attendance records"""
    records = load_attendance_records()
    return jsonify(records)

@app.route('/api/registered_faces')
def get_registered_faces():
    """Get list of registered faces"""
    faces = []
    if os.path.exists(KNOWN_FACES_DIR):
        for filename in os.listdir(KNOWN_FACES_DIR):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(filename)[0]
                faces.append(name)
    return jsonify(faces)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
