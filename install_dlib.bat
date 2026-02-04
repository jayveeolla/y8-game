@echo off
echo ========================================
echo Installing Face Recognition Dependencies
echo ========================================
echo.

echo Installing basic packages...
pip install flask opencv-python numpy pandas openpyxl

echo.
echo Installing dlib...
echo Note: dlib requires Visual C++ Build Tools
pip install dlib

echo.
echo ========================================
echo Downloading dlib models (optional)...
echo ========================================
echo.
echo These models enable FaceNet-style 128D embeddings:
echo 1. shape_predictor_68_face_landmarks.dat (99.7 MB)
echo 2. dlib_face_recognition_resnet_model_v1.dat (22.5 MB)
echo.
echo Download from: http://dlib.net/files/
echo - shape_predictor_68_face_landmarks.dat.bz2
echo - dlib_face_recognition_resnet_model_v1.dat.bz2
echo.
echo Extract .dat files to the tic-tac-toe folder
echo.
echo ========================================
echo Installation Complete!
echo ========================================
pause
