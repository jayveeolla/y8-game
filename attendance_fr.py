from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import json
import os
from datetime import datetime
import pandas as pd
from io import BytesIO
import base64
import cv2
import numpy as np
import pickle

app = Flask(__name__, static_folder='.', static_url_path='')

# Directories
EMPLOYEES_FILE = 'employees.json'
ATTENDANCE_FILE = 'attendance.json'
PHOTOS_DIR = 'employee_photos'
FACE_ENCODINGS_FILE = 'face_encodings.pkl'

# Create directories
if not os.path.exists(PHOTOS_DIR):
    os.makedirs(PHOTOS_DIR)

# Try to use dlib for better face detection, fallback to OpenCV
try:
    import dlib
    USE_DLIB = True
    # Load dlib's face detector (HOG + SVM based)
    dlib_detector = dlib.get_frontal_face_detector()
    # Try to load shape predictor for facial landmarks (68 points)
    try:
        shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        HAS_LANDMARKS = True
    except:
        HAS_LANDMARKS = False
        print("⚠ Shape predictor not found. Facial landmarks disabled.")
    
    # Try to load dlib's face recognition model (ResNet-based, 128D embeddings like FaceNet)
    try:
        face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
        USE_DLIB_RECOGNITION = True
        print("✓ Using Dlib ResNet Face Recognition (FaceNet-style 128D embeddings)")
    except:
        USE_DLIB_RECOGNITION = False
        print("⚠ Dlib face recognition model not found. Using custom embeddings.")
except ImportError:
    USE_DLIB = False
    USE_DLIB_RECOGNITION = False
    HAS_LANDMARKS = False
    print("⚠ Dlib not installed. Using OpenCV + custom algorithms.")

# OpenCV Haar Cascade as fallback
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def load_employees():
    if os.path.exists(EMPLOYEES_FILE):
        with open(EMPLOYEES_FILE, 'r') as f:
            return json.load(f)
    return []

def save_employees(employees):
    with open(EMPLOYEES_FILE, 'w') as f:
        json.dump(employees, f, indent=4)

def load_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'r') as f:
            return json.load(f)
    return []

def save_attendance(attendance):
    with open(ATTENDANCE_FILE, 'w') as f:
        json.dump(attendance, f, indent=4)

def load_face_encodings():
    if os.path.exists(FACE_ENCODINGS_FILE):
        with open(FACE_ENCODINGS_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_face_encodings(encodings):
    with open(FACE_ENCODINGS_FILE, 'wb') as f:
        pickle.dump(encodings, f)

def compute_lbp(image, radius=1, points=8):
    """Compute Local Binary Pattern (LBP) features - Real face recognition algorithm"""
    lbp = np.zeros_like(image)
    for i in range(radius, image.shape[0] - radius):
        for j in range(radius, image.shape[1] - radius):
            center = image[i, j]
            binary = 0
            for p in range(points):
                angle = 2 * np.pi * p / points
                x = i + int(radius * np.sin(angle))
                y = j + int(radius * np.cos(angle))
                if x >= 0 and x < image.shape[0] and y >= 0 and y < image.shape[1]:
                    if image[x, y] >= center:
                        binary += (1 << p)
            lbp[i, j] = binary
    return lbp

def compute_hog_features(image):
    """Compute Histogram of Oriented Gradients (HOG) - Advanced feature extraction"""
    # Calculate gradients
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
    
    # Calculate magnitude and angle
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * (180 / np.pi) % 180
    
    # Divide into cells (8x8 pixels)
    cell_size = 8
    h, w = image.shape
    num_cells_h = h // cell_size
    num_cells_w = w // cell_size
    
    # 9 orientation bins (0-180 degrees)
    bins = 9
    histograms = []
    
    for i in range(num_cells_h):
        for j in range(num_cells_w):
            cell_mag = magnitude[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell_ang = angle[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            
            hist, _ = np.histogram(cell_ang, bins=bins, range=(0, 180), weights=cell_mag)
            histograms.append(hist)
    
    # Normalize in blocks of 2x2 cells
    block_hists = []
    for i in range(num_cells_h - 1):
        for j in range(num_cells_w - 1):
            block = []
            for bi in range(2):
                for bj in range(2):
                    idx = (i + bi) * num_cells_w + (j + bj)
                    if idx < len(histograms):
                        block.extend(histograms[idx])
            
            # L2 normalization
            block = np.array(block)
            norm = np.linalg.norm(block)
            if norm > 0:
                block = block / norm
            block_hists.extend(block)
    
    return np.array(block_hists)

def align_face_dlib(image, face_rect, shape_predictor):
    """Align face using dlib's 68 facial landmarks for better recognition"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Get facial landmarks
    shape = shape_predictor(gray, face_rect)
    
    # Extract eye coordinates for alignment
    left_eye = []
    right_eye = []
    
    # Left eye landmarks (36-41)
    for i in range(36, 42):
        left_eye.append((shape.part(i).x, shape.part(i).y))
    
    # Right eye landmarks (42-47)
    for i in range(42, 48):
        right_eye.append((shape.part(i).x, shape.part(i).y))
    
    # Calculate eye centers
    left_eye_center = np.mean(left_eye, axis=0).astype(int)
    right_eye_center = np.mean(right_eye, axis=0).astype(int)
    
    # Calculate angle for rotation
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Calculate center point for rotation
    eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                   (left_eye_center[1] + right_eye_center[1]) // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    
    # Apply rotation
    aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    return aligned, shape

def extract_dlib_embedding(image, face_rect, shape_predictor, face_rec_model):
    """Extract 128D FaceNet-style embedding using dlib's ResNet model"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Get facial landmarks
    shape = shape_predictor(rgb, face_rect)
    
    # Compute 128D face descriptor (FaceNet-style embedding)
    face_descriptor = face_rec_model.compute_face_descriptor(rgb, shape)
    
    return np.array(face_descriptor)

def detect_face_landmarks(face):
    """Detect key facial landmarks for better feature extraction"""
    eyes = eye_cascade.detectMultiScale(face, scaleFactor=1.1, minNeighbors=5)
    
    landmarks = {
        'has_eyes': len(eyes) >= 2,
        'eye_positions': eyes.tolist() if len(eyes) > 0 else []
    }
    
    return landmarks

def extract_face_features(image):
    """Extract face embeddings using best available method:
    1. Dlib ResNet (128D FaceNet-style) - Most accurate
    2. Dlib HOG + Custom features - Good accuracy
    3. OpenCV Haar + LBP/HOG - Fallback
    """
    
    # Method 1: Use Dlib with ResNet face recognition (FaceNet-style 128D embeddings)
    if USE_DLIB and USE_DLIB_RECOGNITION and HAS_LANDMARKS:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect face with dlib (HOG-based detector)
            faces = dlib_detector(gray, 1)
            
            if len(faces) == 0:
                return None
            
            # Get the largest face
            face_rect = max(faces, key=lambda rect: rect.width() * rect.height())
            
            # Align face using landmarks
            aligned, shape = align_face_dlib(rgb, face_rect, shape_predictor)
            
            # Extract 128D embedding using ResNet model
            embedding = extract_dlib_embedding(aligned, face_rect, shape_predictor, face_rec_model)
            
            # Add metadata flag
            embedding_with_meta = np.concatenate([embedding, [1.0]])  # Flag for dlib embedding
            
            return embedding_with_meta
            
        except Exception as e:
            print(f"Dlib recognition failed: {e}. Falling back...")
    
    # Method 2: Use Dlib detector + Custom features
    if USE_DLIB:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect face with dlib
            faces = dlib_detector(gray, 1)
            
            if len(faces) > 0:
                # Get the largest face
                face_rect = max(faces, key=lambda rect: rect.width() * rect.height())
                
                # Extract face region
                x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
                
                # Add padding
                padding = int(w * 0.15)
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(gray.shape[1] - x, w + 2 * padding)
                h = min(gray.shape[0] - y, h + 2 * padding)
                
                face = gray[y:y+h, x:x+w]
                
                # Apply custom feature extraction
                return extract_custom_features(face, use_advanced=True)
        except Exception as e:
            print(f"Dlib detection failed: {e}. Falling back to OpenCV...")
    
    # Method 3: Fallback to OpenCV Haar Cascade
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Multi-scale face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(80, 80))
    
    if len(faces) == 0:
        # Try with relaxed parameters
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        
    if len(faces) == 0:
        return None
    
    # Get the largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    
    # Add padding
    padding = int(w * 0.15)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(gray.shape[1] - x, w + 2 * padding)
    h = min(gray.shape[0] - y, h + 2 * padding)
    
    face = gray[y:y+h, x:x+w]
    
    return extract_custom_features(face, use_advanced=True)

def extract_custom_features(face, use_advanced=True):
    """Extract custom features using LBP, HOG, Gabor filters"""
    # Resize to standard size
    face = cv2.resize(face, (128, 128))
    
    # Preprocessing: CLAHE for lighting normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    face_clahe = clahe.apply(face)
    
    # Noise reduction
    face_blur = cv2.GaussianBlur(face_clahe, (5, 5), 0)
    
    # Detect facial landmarks
    landmarks = detect_face_landmarks(face)
    
    # Feature 1: Local Binary Patterns (LBP)
    lbp = compute_lbp(face_blur, radius=2, points=8)
    
    # Divide LBP into 8x8 grid
    h, w = lbp.shape
    lbp_features = []
    for i in range(8):
        for j in range(8):
            region = lbp[i*h//8:(i+1)*h//8, j*w//8:(j+1)*w//8]
            hist = cv2.calcHist([region.astype(np.uint8)], [0], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            lbp_features.append(hist)
    
    lbp_features = np.concatenate(lbp_features)
    
    # Feature 2: HOG
    hog_features = compute_hog_features(face_blur)
    
    # Feature 3: Regional intensity histograms
    hist_features = []
    for i in range(6):
        for j in range(6):
            region = face_blur[i*h//6:(i+1)*h//6, j*w//6:(j+1)*w//6]
            hist = cv2.calcHist([region], [0], None, [16], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.append(hist)
    
    hist_features = np.concatenate(hist_features)
    
    # Feature 4: Gabor filters
    gabor_features = []
    for theta in [0, 45, 90, 135]:
        kernel = cv2.getGaborKernel((21, 21), 5.0, np.pi * theta / 180, 10.0, 0.5, 0)
        filtered = cv2.filter2D(face_blur, cv2.CV_32F, kernel)
        gabor_features.append(filtered.mean())
        gabor_features.append(filtered.std())
    
    gabor_features = np.array(gabor_features)
    
    # Combine all features
    combined_features = np.concatenate([
        lbp_features * 0.40,
        hog_features * 0.30,
        hist_features * 0.20,
        gabor_features * 0.10
    ])
    
    # Add landmark confidence score
    landmark_score = 1.0 if landmarks['has_eyes'] else 0.7
    combined_features = combined_features * landmark_score
    
    # Add metadata flag
    combined_features = np.concatenate([combined_features, [0.0]])  # Flag for custom embedding
    
    return combined_features

def compare_faces(encoding1, encoding2):
    """Compare two face encodings using appropriate method:
    - For dlib embeddings (128D): Euclidean distance (FaceNet/ArcFace style)
    - For custom features: Multiple similarity metrics
    """
    if encoding1 is None or encoding2 is None:
        return 0
    
    # Check if encodings have different sizes
    if len(encoding1) != len(encoding2):
        return 0
    
    # Ensure numpy arrays
    enc1 = np.array(encoding1, dtype=np.float64)
    enc2 = np.array(encoding2, dtype=np.float64)
    
    # Check metadata flag (last element)
    is_dlib_embedding = enc1[-1] > 0.5 and enc2[-1] > 0.5
    
    # Remove metadata flag
    enc1 = enc1[:-1]
    enc2 = enc2[:-1]
    
    # Method 1: Dlib ResNet embeddings (FaceNet-style)
    # Use Euclidean distance like FaceNet/ArcFace
    if is_dlib_embedding:
        # Calculate Euclidean distance
        distance = np.linalg.norm(enc1 - enc2)
        
        # FaceNet threshold: distance < 0.6 is same person
        # ArcFace threshold: distance < 1.0 is same person
        # Convert to similarity score (0-1)
        # Typical same-person distances: 0.3-0.6
        # Different person distances: 0.8-1.5
        
        if distance < 0.4:
            similarity = 1.0
        elif distance < 0.6:
            similarity = 1.0 - (distance - 0.4) / 0.2 * 0.2  # 0.8-1.0
        elif distance < 1.0:
            similarity = 0.8 - (distance - 0.6) / 0.4 * 0.3  # 0.5-0.8
        else:
            similarity = max(0, 0.5 - (distance - 1.0) / 0.5 * 0.5)  # 0-0.5
        
        # Also use cosine similarity for additional validation
        dot_product = np.dot(enc1, enc2)
        norm_product = np.linalg.norm(enc1) * np.linalg.norm(enc2)
        cosine_sim = dot_product / norm_product if norm_product > 0 else 0
        
        # Combine both metrics (weighted toward Euclidean as per FaceNet)
        final_similarity = (similarity * 0.8) + (cosine_sim * 0.2)
        
        return float(np.clip(final_similarity, 0, 1))
    
    # Method 2: Custom features - use multiple metrics
    # 1. Euclidean Distance
    euclidean_dist = np.linalg.norm(enc1 - enc2)
    max_dist = np.sqrt(len(enc1))
    euclidean_sim = max(0, 1 - (euclidean_dist / max_dist))
    
    # 2. Cosine Similarity
    dot_product = np.dot(enc1, enc2)
    norm_product = np.linalg.norm(enc1) * np.linalg.norm(enc2)
    cosine_sim = dot_product / norm_product if norm_product > 0 else 0
    cosine_sim = (cosine_sim + 1) / 2
    
    # 3. Pearson Correlation
    if np.std(enc1) > 0 and np.std(enc2) > 0:
        correlation = np.corrcoef(enc1, enc2)[0, 1]
        correlation = (correlation + 1) / 2
    else:
        correlation = 0
    
    # 4. Chi-Square Distance
    eps = 1e-10
    chi_square = np.sum((enc1 - enc2) ** 2 / (enc1 + enc2 + eps))
    chi_square_sim = max(0, 1 - (chi_square / len(enc1)))
    
    # 5. Manhattan Distance
    manhattan_dist = np.sum(np.abs(enc1 - enc2))
    manhattan_sim = max(0, 1 - (manhattan_dist / (len(enc1) * 255)))
    
    # 6. Histogram Intersection
    hist_intersection = np.sum(np.minimum(enc1, enc2)) / np.sum(enc2) if np.sum(enc2) > 0 else 0
    
    # Weighted combination
    similarity = (
        euclidean_sim * 0.25 +
        cosine_sim * 0.25 +
        correlation * 0.20 +
        chi_square_sim * 0.15 +
        manhattan_sim * 0.10 +
        hist_intersection * 0.05
    )
    
    # Non-linear transformation
    similarity = similarity ** 1.2
    
    return float(np.clip(similarity, 0, 1))

@app.route('/')
def index():
    return render_template('attendance_home.html')

@app.route('/register')
def register():
    return render_template('register_employee.html')

@app.route('/scan')
def scan():
    return render_template('scan_attendance.html')

@app.route('/management')
def management():
    return render_template('attendance_management.html')

@app.route('/manage_employees')
def manage_employees():
    return render_template('manage_employees.html')

@app.route('/api/register_employee', methods=['POST'])
def register_employee():
    try:
        data = request.get_json()
        emp_id = data.get('emp_id')
        name = data.get('name')
        location = data.get('location')
        shift = data.get('shift')
        photos = data.get('photos')
        
        if not all([emp_id, name, location, shift]):
            return jsonify({'success': False, 'message': 'All fields required'})
        
        employees = load_employees()
        
        # Check if employee ID exists
        if any(e['emp_id'] == emp_id for e in employees):
            return jsonify({'success': False, 'message': 'Employee ID already exists'})
        
        # Save photos and extract face encodings
        face_encodings_list = []
        photo_filenames = []
        
        if photos:
            # Create folder in labels directory based on employee ID
            emp_folder = os.path.join('labels', emp_id)
            os.makedirs(emp_folder, exist_ok=True)
            
            # Process each photo
            for idx, photo_data_url in enumerate(photos, start=1):
                # Decode image
                image_data = photo_data_url.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Extract face features
                face_encoding = extract_face_features(image)
                
                if face_encoding is not None:
                    face_encodings_list.append(face_encoding.tolist())
                
                # Save photo in labels folder
                photo_filename = f"{idx}.jpg"
                photo_path = os.path.join(emp_folder, photo_filename)
                cv2.imwrite(photo_path, image)
                photo_filenames.append(photo_filename)
            
            if not face_encodings_list:
                return jsonify({'success': False, 'message': 'No face detected in photos. Please try again.'})
            
            # Also save first photo in employee_photos for backward compatibility
            image_data = photos[0].split(',')[1]
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            photo_path = os.path.join(PHOTOS_DIR, f'{emp_id}.jpg')
            cv2.imwrite(photo_path, image)
            
            # Save face encodings (use average or first one)
            encodings = load_face_encodings()
            encodings[emp_id] = face_encodings_list[0]  # Use first encoding
            save_face_encodings(encodings)
        
        # Add employee
        employees.append({
            'emp_id': emp_id,
            'name': name,
            'location': location,
            'shift': shift,
            'photo': bool(photos),
            'photos_count': len(photo_filenames),
            'registered_date': datetime.now().isoformat()
        })
        
        save_employees(employees)
        
        return jsonify({'success': True, 'message': f'{name} registered successfully!'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/recognize_face', methods=['POST'])
def recognize_face():
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'No image provided'})
        
        # Decode image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Extract face features
        captured_encoding = extract_face_features(image)
        
        if captured_encoding is None:
            return jsonify({'success': False, 'message': 'No face detected. Please position your face clearly in the camera.'})
        
        # Load all face encodings
        encodings = load_face_encodings()
        
        if not encodings:
            return jsonify({'success': False, 'message': 'No registered faces found. Please register first.'})
        
        # Find best match with advanced scoring system
        best_match = None
        best_score = 0
        second_best_score = 0
        match_scores = []
        
        for emp_id, stored_encoding in encodings.items():
            score = compare_faces(captured_encoding, np.array(stored_encoding))
            match_scores.append((emp_id, score))
            
            if score > best_score:
                second_best_score = best_score
                best_score = score
                best_match = emp_id
            elif score > second_best_score:
                second_best_score = score
        
        # Check if using dlib embeddings (128D + 1 flag)
        is_dlib = len(captured_encoding) == 129 and captured_encoding[-1] > .3
        
        # Adaptive threshold based on embedding type
        if is_dlib:
            # Dlib ResNet (FaceNet-style) has higher accuracy, lower threshold
            min_threshold = 0.70  # Dlib is more reliable
            min_margin = 0.08
        else:
            # Custom features require higher threshold
            min_threshold = 0.75
            min_margin = 0.05
        
        # Calculate confidence margin (difference between best and second-best)
        confidence_margin = best_score 
        
        # Require both high score AND clear distinction from other faces
        embedding_type = "Dlib ResNet (FaceNet)" if is_dlib else "LBP+HOG+Gabor"
        
        if best_score < min_threshold:
            return jsonify({
                'success': False, 
                'message': f'Face similarity too low ({best_score*100:.1f}%). Please position your face clearly and try again.',
                'debug': f'Using: {embedding_type}, Threshold: {min_threshold}'
            })
        
        if confidence_margin < min_margin and len(match_scores) > 1:
            return jsonify({
                'success': False,
                'message': 'Multiple similar faces detected. Please ensure better lighting and try again.',
                'debug': f'Margin: {confidence_margin:.3f}, Required: {min_margin}'
            })
        
        # Additional validation: check if the match is distinctly better
        if best_match is None:
            return jsonify({'success': False, 'message': 'Face not recognized. Please try again or register first.'})
        
        # Get employee info
        employees = load_employees()
        employee = next((e for e in employees if e['emp_id'] == best_match), None)
        
        if not employee:
            return jsonify({'success': False, 'message': 'Employee data not found.'})
        
        # Mark attendance
        attendance = load_attendance()
        today = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H:%M:%S')
        
        # Get attendance type from request or auto-detect
        attendance_type = data.get('attendance_type')
        
        if not attendance_type:
            # Auto-detect based on last record
            today_records = [r for r in attendance if r['emp_id'] == best_match and r['date'] == today]
            
            if not today_records:
                attendance_type = 'IN'
            else:
                last_record = today_records[-1]
                # Check if last record is already OUT
                if last_record['type'] == 'OUT':
                    return jsonify({
                        'success': False,
                        'message': f'{employee["name"]} already marked OUT today at {last_record["time"]}. Cannot mark attendance again for today.'
                    })
                # Last record is IN, so next should be OUT
                attendance_type = 'OUT'
        else:
            # Validate requested attendance type
            today_records = [r for r in attendance if r['emp_id'] == best_match and r['date'] == today]
            
            if attendance_type == 'OUT':
                # Check if there's an IN record today
                has_in_today = any(r['type'] == 'IN' for r in today_records)
                if not has_in_today:
                    return jsonify({
                        'success': False,
                        'message': f'{employee["name"]} must mark IN first before marking OUT!'
                    })
                # Check if already marked OUT
                has_out_today = any(r['type'] == 'OUT' for r in today_records)
                if has_out_today:
                    return jsonify({
                        'success': False,
                        'message': f'{employee["name"]} already marked OUT today. Cannot mark OUT again.'
                    })
            elif attendance_type == 'IN':
                # Check if already marked IN without OUT
                if today_records and today_records[-1]['type'] == 'IN':
                    return jsonify({
                        'success': False,
                        'message': f'{employee["name"]} already marked IN today. Please mark OUT first.'
                    })
        
        # Generate message
        if attendance_type == 'IN':
            message = f'Welcome {employee["name"]}! Attendance marked IN at {current_time}'
        else:
            message = f'Goodbye {employee["name"]}! Attendance marked OUT at {current_time}'
        
        # Add attendance record
        attendance.append({
            'emp_id': best_match,
            'name': employee['name'],
            'date': today,
            'time': current_time,
            'type': attendance_type,
            'timestamp': datetime.now().isoformat(),
            'confidence': float(best_score)
        })
        
        save_attendance(attendance)
        
        # Determine recognition method for display
        if is_dlib and USE_DLIB_RECOGNITION:
            recognition_method = "Dlib ResNet (FaceNet-style 128D)"
        elif is_dlib:
            recognition_method = "Dlib HOG + Custom Features"
        else:
            recognition_method = "OpenCV + LBP/HOG/Gabor"
        
        return jsonify({
            'success': True,
            'employee': {
                'emp_id': employee['emp_id'],
                'name': employee['name'],
                'location': employee['location'],
                'shift': employee['shift']
            },
            'attendance_type': attendance_type,
            'time': current_time,
            'message': message,
            'confidence': round(best_score * 100, 2),
            'recognition_method': recognition_method
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/employees')
def get_employees():
    employees = load_employees()
    return jsonify(employees)

@app.route('/api/employee_status')
def get_employee_status():
    """Get current IN/OUT status of all employees"""
    try:
        employees = load_employees()
        attendance = load_attendance()
        today = datetime.now().strftime('%Y-%m-%d')
        
        employee_status = []
        
        for employee in employees:
            emp_id = employee['emp_id']
            
            # Get today's records for this employee
            today_records = [r for r in attendance if r['emp_id'] == emp_id and r['date'] == today]
            
            if today_records:
                # Check if they have both IN and OUT
                has_in = any(r['type'] == 'IN' for r in today_records)
                has_out = any(r['type'] == 'OUT' for r in today_records)
                
                # If they have both IN and OUT, mark as completed (don't show)
                if has_in and has_out:
                    continue  # Skip this employee
                
                # Sort by timestamp to get the latest
                today_records.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                last_record = today_records[0]
                
                status = last_record['type']  # 'IN' or 'OUT'
                time = last_record['time']
            else:
                # No attendance today
                status = 'OUT'
                time = None
            
            employee_status.append({
                'emp_id': emp_id,
                'name': employee['name'],
                'location': employee.get('location', ''),
                'status': status,
                'time': time
            })
        
        return jsonify({
            'success': True,
            'employees': employee_status
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/attendance')
def get_attendance():
    from_date = request.args.get('from_date')
    to_date = request.args.get('to_date')
    
    attendance = load_attendance()
    employees = load_employees()
    emp_dict = {e['emp_id']: e for e in employees}
    
    if from_date and to_date:
        attendance = [r for r in attendance if from_date <= r['date'] <= to_date]
    
    # Group by employee and date
    grouped = {}
    for record in attendance:
        key = f"{record['emp_id']}_{record['date']}"
        if key not in grouped:
            emp = emp_dict.get(record['emp_id'], {})
            grouped[key] = {
                'emp_id': record['emp_id'],
                'name': record['name'],
                'date': record['date'],
                'location': emp.get('location', 'N/A'),
                'shift': emp.get('shift', 'N/A'),
                'time_in': '-',
                'time_out': '-',
                'status': 'Incomplete',
                'timestamp': record.get('timestamp', record['date'])
            }
        
        if record['type'] == 'IN':
            grouped[key]['time_in'] = record['time']
        else:
            grouped[key]['time_out'] = record['time']
        
        # Update status
        if grouped[key]['time_in'] != '-' and grouped[key]['time_out'] != '-':
            grouped[key]['status'] = 'Complete'
    
    result = list(grouped.values())
    return jsonify(result)

@app.route('/api/employee/<emp_id>', methods=['GET', 'PUT', 'DELETE'])
def manage_employee(emp_id):
    employees = load_employees()
    
    if request.method == 'GET':
        employee = next((e for e in employees if e['emp_id'] == emp_id), None)
        if employee:
            return jsonify({'success': True, 'employee': employee})
        return jsonify({'success': False, 'message': 'Employee not found'})
    
    elif request.method == 'PUT':
        try:
            data = request.get_json()
            employee = next((e for e in employees if e['emp_id'] == emp_id), None)
            
            if not employee:
                return jsonify({'success': False, 'message': 'Employee not found'})
            
            # Update fields
            employee['name'] = data.get('name', employee['name'])
            employee['location'] = data.get('location', employee['location'])
            employee['shift'] = data.get('shift', employee['shift'])
            
            # Update photo if provided
            if data.get('photo'):
                image_data = data['photo'].split(',')[1]
                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Extract and save new face encoding
                face_encoding = extract_face_features(image)
                if face_encoding is not None:
                    encodings = load_face_encodings()
                    encodings[emp_id] = face_encoding.tolist()
                    save_face_encodings(encodings)
                    
                    photo_path = os.path.join(PHOTOS_DIR, f'{emp_id}.jpg')
                    cv2.imwrite(photo_path, image)
                    employee['photo'] = True
            
            save_employees(employees)
            return jsonify({'success': True, 'message': 'Employee updated successfully'})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
    
    elif request.method == 'DELETE':
        try:
            employees = [e for e in employees if e['emp_id'] != emp_id]
            save_employees(employees)
            
            # Delete photo
            photo_path = os.path.join(PHOTOS_DIR, f'{emp_id}.jpg')
            if os.path.exists(photo_path):
                os.remove(photo_path)
            
            # Delete face encoding
            encodings = load_face_encodings()
            if emp_id in encodings:
                del encodings[emp_id]
                save_face_encodings(encodings)
            
            return jsonify({'success': True, 'message': 'Employee deleted successfully'})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})

@app.route('/api/employee_photo/<emp_id>')
def get_employee_photo(emp_id):
    photo_path = os.path.join(PHOTOS_DIR, f'{emp_id}.jpg')
    if os.path.exists(photo_path):
        return send_file(photo_path, mimetype='image/jpeg')
    return '', 404

@app.route('/api/employee_attendance/<emp_id>')
def get_employee_attendance(emp_id):
    try:
        attendance = load_attendance()
        employee_records = [record for record in attendance if record['emp_id'] == emp_id]
        return jsonify({'success': True, 'attendance': employee_records})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/export/csv', methods=['POST'])
def export_csv():
    """Export attendance to CSV"""
    try:
        data = request.get_json()
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        attendance = load_attendance()
        
        # Filter by date range
        if start_date and end_date:
            filtered = [r for r in attendance if start_date <= r['date'] <= end_date]
        else:
            filtered = attendance
        
        if not filtered:
            return jsonify({'success': False, 'message': 'No records found for the selected date range'})
        
        # Group by employee and date
        grouped = {}
        for record in filtered:
            key = f"{record['emp_id']}_{record['date']}"
            if key not in grouped:
                grouped[key] = {
                    'emp_id': record['emp_id'],
                    'name': record['name'],
                    'date': record['date'],
                    'in': '-',
                    'out': '-',
                    'location': record.get('location', '-'),
                    'shift': record.get('shift', '-')
                }
            if record['type'] == 'IN':
                grouped[key]['in'] = record['time']
            else:
                grouped[key]['out'] = record['time']
        
        # Create DataFrame
        df_data = []
        for key, record in grouped.items():
            df_data.append({
                'Date': record['date'],
                'Employee ID': record['emp_id'],
                'Name': record['name'],
                'Location': record['location'],
                'Shift': record['shift'],
                'Time In': record['in'],
                'Time Out': record['out'],
                'Status': 'Complete' if record['in'] != '-' and record['out'] != '-' else 'Incomplete'
            })
        
        df = pd.DataFrame(df_data)
        df = df.sort_values(['Date', 'Employee ID'], ascending=[False, True])
        
        # Create CSV file in memory
        output = BytesIO()
        df.to_csv(output, index=False, encoding='utf-8')
        output.seek(0)
        
        filename = f'attendance_{start_date}_to_{end_date}.csv'
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/export_attendance')
def export_attendance():
    try:
        from_date = request.args.get('from_date')
        to_date = request.args.get('to_date')
        
        attendance = load_attendance()
        
        if from_date and to_date:
            attendance = [r for r in attendance if from_date <= r['date'] <= to_date]
        
        if not attendance:
            return jsonify({'success': False, 'message': 'No records found'})
        
        # Group by employee and date
        grouped = {}
        for record in attendance:
            key = f"{record['emp_id']}_{record['date']}"
            if key not in grouped:
                grouped[key] = {
                    'emp_id': record['emp_id'],
                    'name': record['name'],
                    'date': record['date'],
                    'in': '-',
                    'out': '-'
                }
            if record['type'] == 'IN':
                grouped[key]['in'] = record['time']
            else:
                grouped[key]['out'] = record['time']
        
        # Create DataFrame
        df_data = []
        employees = load_employees()
        emp_dict = {e['emp_id']: e for e in employees}
        
        for key, record in grouped.items():
            emp = emp_dict.get(record['emp_id'], {})
            df_data.append({
                'Date': record['date'],
                'Employee ID': record['emp_id'],
                'Name': record['name'],
                'Location': emp.get('location', '-'),
                'Shift': emp.get('shift', '-'),
                'Time In': record['in'],
                'Time Out': record['out'],
                'Status': 'Complete' if record['in'] != '-' and record['out'] != '-' else 'Incomplete'
            })
        
        df = pd.DataFrame(df_data)
        df = df.sort_values(['Date', 'Employee ID'], ascending=[False, True])
        
        # Create Excel file
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Attendance')
            worksheet = writer.sheets['Attendance']
            
            for idx, col in enumerate(df.columns):
                max_length = max(df[col].astype(str).apply(len).max(), len(col)) + 2
                worksheet.column_dimensions[chr(65 + idx)].width = max_length
        
        output.seek(0)
        
        filename = f"Attendance_{from_date}_to_{to_date}.xlsx"
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/export_employee_attendance')
def export_employee_attendance():
    try:
        emp_id = request.args.get('emp_id')
        from_date = request.args.get('from_date')
        to_date = request.args.get('to_date')
        
        if not emp_id:
            return jsonify({'success': False, 'message': 'Employee ID required'})
        
        employees = load_employees()
        employee = next((e for e in employees if e['emp_id'] == emp_id), None)
        
        if not employee:
            return jsonify({'success': False, 'message': 'Employee not found'})
        
        attendance = load_attendance()
        records = [r for r in attendance if r['emp_id'] == emp_id]
        
        if from_date and to_date:
            records = [r for r in records if from_date <= r['date'] <= to_date]
        
        if not records:
            return jsonify({'success': False, 'message': 'No records found'})
        
        grouped = {}
        for record in records:
            date = record['date']
            if date not in grouped:
                grouped[date] = {'in': '-', 'out': '-'}
            if record['type'] == 'IN':
                grouped[date]['in'] = record['time']
            else:
                grouped[date]['out'] = record['time']
        
        df_data = []
        for date in sorted(grouped.keys()):
            df_data.append({
                'Date': date,
                'Employee ID': emp_id,
                'Employee Name': employee['name'],
                'Location': employee['location'],
                'Shift': employee['shift'],
                'Time In': grouped[date]['in'],
                'Time Out': grouped[date]['out'],
                'Status': 'Complete' if grouped[date]['in'] != '-' and grouped[date]['out'] != '-' else 'Incomplete'
            })
        
        df = pd.DataFrame(df_data)
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Attendance')
            worksheet = writer.sheets['Attendance']
            
            for idx, col in enumerate(df.columns):
                max_length = max(df[col].astype(str).apply(len).max(), len(col)) + 2
                worksheet.column_dimensions[chr(65 + idx)].width = max_length
        
        output.seek(0)
        
        filename = f"Attendance_{employee['name'].replace(' ', '_')}_{from_date}_to_{to_date}.xlsx"
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# Face-API.js integration routes
@app.route('/scan_faceapi')
def scan_faceapi():
    return render_template('scan_attendance_faceapi.html')

@app.route('/api/get_employee_ids')
def get_employee_ids():
    try:
        employees = load_employees()
        employee_ids = [e['emp_id'] for e in employees]
        return jsonify({'success': True, 'employee_ids': employee_ids})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/mark_attendance_by_id', methods=['POST'])
def mark_attendance_by_id():
    try:
        data = request.get_json()
        emp_id = data.get('emp_id')
        attendance_type = data.get('attendance_type')
        confidence = data.get('confidence', 95)
        
        if not all([emp_id, attendance_type]):
            return jsonify({'success': False, 'message': 'Missing required parameters'})
        
        # Get employee info
        employees = load_employees()
        employee = next((e for e in employees if e['emp_id'] == emp_id), None)
        
        if not employee:
            return jsonify({'success': False, 'message': 'Employee not found'})
        
        # Mark attendance
        attendance = load_attendance()
        today = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H:%M:%S')
        
        # Validate attendance type based on today's records
        today_records = [r for r in attendance if r['emp_id'] == emp_id and r['date'] == today]
        
        if attendance_type == 'OUT':
            # Check if there's an IN record today
            has_in_today = any(r['type'] == 'IN' for r in today_records)
            if not has_in_today:
                return jsonify({
                    'success': False,
                    'message': f'{employee["name"]} must mark IN first before marking OUT!'
                })
            # Check if already marked OUT
            has_out_today = any(r['type'] == 'OUT' for r in today_records)
            if has_out_today:
                return jsonify({
                    'success': False,
                    'message': f'{employee["name"]} already marked OUT today. Cannot mark OUT again.'
                })
        elif attendance_type == 'IN':
            # Check if already marked IN without OUT
            if today_records and today_records[-1]['type'] == 'IN':
                return jsonify({
                    'success': False,
                    'message': f'{employee["name"]} already marked IN today. Please mark OUT first.'
                })
            # Check if already completed (both IN and OUT)
            if today_records and today_records[-1]['type'] == 'OUT':
                return jsonify({
                    'success': False,
                    'message': f'{employee["name"]} already marked OUT today. Cannot mark attendance again for today.'
                })
        
        # Generate message
        if attendance_type == 'IN':
            message = f'Welcome {employee["name"]}! Attendance marked IN at {current_time}'
        else:
            message = f'Goodbye {employee["name"]}! Attendance marked OUT at {current_time}'
        
        # Add attendance record
        attendance.append({
            'emp_id': emp_id,
            'name': employee['name'],
            'date': today,
            'time': current_time,
            'type': attendance_type,
            'timestamp': datetime.now().isoformat(),
            'confidence': float(confidence)
        })
        
        save_attendance(attendance)
        
        return jsonify({
            'success': True,
            'employee': {
                'emp_id': employee['emp_id'],
                'name': employee['name'],
                'location': employee['location'],
                'shift': employee['shift']
            },
            'attendance_type': attendance_type,
            'time': current_time,
            'message': message,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('js', filename)

@app.route('/models/<path:filename>')
def serve_models(filename):
    return send_from_directory('models', filename)

@app.route('/labels/<path:filename>')
def serve_labels(filename):
    return send_from_directory('labels', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
