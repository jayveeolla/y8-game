# Face Recognition Attendance System

A complete web-based attendance system using facial recognition technology.

## Features

- **Face Recognition**: Automatic face detection and recognition
- **Register New Faces**: Easy registration with webcam capture
- **Mark Attendance**: Quick attendance marking with face scan
- **Attendance Records**: View and filter attendance history
- **Real-time Processing**: Instant face recognition
- **User-friendly Interface**: Modern and responsive design

## Installation

### 1. Install Python Dependencies

```bash
pip install flask opencv-python face_recognition numpy
```

### 2. Install Additional Requirements (Windows)

For face_recognition to work on Windows, you may need to install:
- Visual C++ build tools
- CMake

Or use pre-built wheels:
```bash
pip install cmake
pip install dlib
pip install face_recognition
```

### 3. Run the Application

```bash
python attendance_system.py
```

The application will start on `http://localhost:5001`

## Usage

### 1. Register Faces
1. Go to "Register Face" page
2. Enter the person's name
3. Click "Start Camera"
4. Capture photo when face is clearly visible
5. Click "Register" to save

### 2. Mark Attendance
1. Go to "Mark Attendance" page
2. Click "Start Camera"
3. Click "Mark Attendance" to scan face
4. System will recognize and mark attendance

### 3. View Records
1. Go to "View Records" page
2. See all attendance records
3. Filter by name or date
4. Export or manage records

## Folder Structure

```
attendance_system.py       - Main Flask application
templates/
  ├── attendance.html      - Mark attendance page
  ├── register_face.html   - Register new faces
  └── attendance_records.html - View records
known_faces/               - Stored face images (auto-created)
attendance_records.json    - Attendance database (auto-created)
```

## Tips for Best Results

- Ensure good lighting when registering and marking attendance
- Look directly at the camera
- Remove glasses or masks for better recognition
- Keep face centered in the camera view
- Use a neutral expression

## Troubleshooting

**Camera not working?**
- Check browser permissions
- Ensure no other app is using the camera

**Face not recognized?**
- Try re-registering with better lighting
- Ensure face is clearly visible
- Check if face was registered correctly

**Installation issues?**
- Make sure Python 3.7+ is installed
- Install Visual C++ build tools on Windows
- Use pre-built wheels for face_recognition

## Technologies Used

- **Backend**: Flask (Python)
- **Face Recognition**: face_recognition library (dlib + OpenCV)
- **Frontend**: HTML5, CSS3, JavaScript
- **Camera**: WebRTC (getUserMedia API)

## Security Notes

- Face data is stored locally
- No cloud processing
- Images are saved in `known_faces/` folder
- Attendance records in JSON format

## Future Enhancements

- Multiple face recognition in one frame
- Export to Excel/PDF
- Email notifications
- Mobile app integration
- Admin dashboard
- Attendance reports and analytics
