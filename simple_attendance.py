from flask import Flask, render_template, request, jsonify
import os
import json
from datetime import datetime
import base64

app = Flask(__name__)

# Directories
STUDENT_PHOTOS_DIR = 'student_photos'
ATTENDANCE_FILE = 'attendance_records.json'

# Create directories if they don't exist
if not os.path.exists(STUDENT_PHOTOS_DIR):
    os.makedirs(STUDENT_PHOTOS_DIR)

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

def get_registered_students():
    """Get list of registered students"""
    students = []
    if os.path.exists(STUDENT_PHOTOS_DIR):
        for filename in os.listdir(STUDENT_PHOTOS_DIR):
            if filename.endswith('.jpg'):
                name = filename.replace('.jpg', '')
                students.append(name)
    return sorted(students)

@app.route('/')
def index():
    return render_template('simple_attendance.html')

@app.route('/register')
def register_page():
    return render_template('simple_register.html')

@app.route('/records')
def records_page():
    return render_template('simple_records.html')

@app.route('/api/register_student', methods=['POST'])
def register_student():
    """Register a new student"""
    try:
        data = request.get_json()
        name = data.get('name')
        image_data = data.get('image')
        
        if not name or not image_data:
            return jsonify({'success': False, 'message': 'Name and image required'})
        
        # Decode and save image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        filename = f"{name}.jpg"
        filepath = os.path.join(STUDENT_PHOTOS_DIR, filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        return jsonify({'success': True, 'message': f'{name} registered successfully!'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/mark_attendance', methods=['POST'])
def mark_attendance():
    """Mark attendance for a student"""
    try:
        data = request.get_json()
        name = data.get('name')
        
        if not name:
            return jsonify({'success': False, 'message': 'Name required'})
        
        # Check if student is registered
        students = get_registered_students()
        if name not in students:
            return jsonify({'success': False, 'message': 'Student not registered'})
        
        # Mark attendance
        if save_attendance_record(name):
            return jsonify({'success': True, 'message': f'Attendance marked for {name}'})
        else:
            return jsonify({'success': False, 'message': f'{name} already marked today'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/attendance_records')
def get_attendance_records():
    """Get all attendance records"""
    records = load_attendance_records()
    return jsonify(records)

@app.route('/api/registered_students')
def get_students():
    """Get list of registered students"""
    students = get_registered_students()
    return jsonify(students)

if __name__ == '__main__':
    print("=" * 60)
    print("Simple Attendance System Started")
    print("=" * 60)
    print("Open your browser to: http://localhost:5001")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5001, debug=True)
