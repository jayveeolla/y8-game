@echo off
echo ========================================
echo Face Recognition Attendance System
echo Installation Script
echo ========================================
echo.

echo Installing required packages...
echo.

pip install flask
pip install opencv-python
pip install numpy

echo.
echo ========================================
echo Installing face_recognition...
echo This may take a few minutes...
echo ========================================
echo.

pip install cmake
pip install dlib
pip install face_recognition

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo To run the attendance system:
echo python attendance_system.py
echo.
echo Then open your browser to:
echo http://localhost:5001
echo.
pause
