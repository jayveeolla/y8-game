@echo off
cd /d "%~dp0"
echo ========================================
echo Starting Face Recognition Attendance System
echo ========================================
echo.
echo Server will start on: http://localhost:5001
echo.
echo Press Ctrl+C to stop the server
echo.
python attendance_system.py
pause
