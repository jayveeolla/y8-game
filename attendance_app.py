from flask import Flask, render_template, request, jsonify, send_file
import json
import os
from datetime import datetime
import pandas as pd
from io import BytesIO
import base64

app = Flask(__name__)

# File paths
EMPLOYEES_FILE = 'employees.json'
ATTENDANCE_FILE = 'attendance_records_enhanced.json'
PHOTOS_DIR = 'employee_photos'

# Create photos directory if it doesn't exist
if not os.path.exists(PHOTOS_DIR):
    os.makedirs(PHOTOS_DIR)

def load_employees():
    """Load employee data"""
    if os.path.exists(EMPLOYEES_FILE):
        with open(EMPLOYEES_FILE, 'r') as f:
            return json.load(f)
    return []

def save_employees(employees):
    """Save employee data"""
    with open(EMPLOYEES_FILE, 'w') as f:
        json.dump(employees, f, indent=4)

def load_attendance():
    """Load attendance records"""
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'r') as f:
            return json.load(f)
    return []

def save_attendance(records):
    """Save attendance records"""
    with open(ATTENDANCE_FILE, 'w') as f:
        json.dump(records, f, indent=4)

@app.route('/')
def index():
    return render_template('attendance_home.html')

@app.route('/register')
def register_page():
    return render_template('register_employee.html')

@app.route('/management')
def management_page():
    return render_template('attendance_management.html')

@app.route('/manage_employees')
def manage_employees_page():
    return render_template('manage_employees.html')

@app.route('/api/register_employee', methods=['POST'])
def register_employee():
    """Register a new employee"""
    try:
        data = request.get_json()
        
        employees = load_employees()
        
        # Check if employee ID already exists
        if any(emp['emp_id'] == data['emp_id'] for emp in employees):
            return jsonify({'success': False, 'message': 'Employee ID already exists!'})
        
        # Save photos if provided
        photo_filenames = []
        if 'photos' in data and data['photos']:
            # Create folder in labels directory based on employee ID
            emp_folder = os.path.join('labels', data['emp_id'])
            os.makedirs(emp_folder, exist_ok=True)
            
            # Save each photo
            for idx, photo_data_url in enumerate(data['photos'], start=1):
                photo_data = photo_data_url.split(',')[1]  # Remove data:image/jpeg;base64,
                photo_bytes = base64.b64decode(photo_data)
                photo_filename = f"{idx}.jpg"
                photo_path = os.path.join(emp_folder, photo_filename)
                
                with open(photo_path, 'wb') as f:
                    f.write(photo_bytes)
                photo_filenames.append(photo_filename)
        
        # Also save first photo in employee_photos for backward compatibility
        if 'photos' in data and data['photos']:
            photo_data = data['photos'][0].split(',')[1]
            photo_bytes = base64.b64decode(photo_data)
            photo_filename = f"{data['emp_id']}.jpg"
            photo_path = os.path.join(PHOTOS_DIR, photo_filename)
            
            with open(photo_path, 'wb') as f:
                f.write(photo_bytes)
        
        # Add new employee
        new_employee = {
            'emp_id': data['emp_id'],
            'name': data['name'],
            'location': data['location'],
            'shift': data['shift'],
            'photo': photo_filename if photo_filenames else None,
            'photos_count': len(photo_filenames),
            'registered_date': datetime.now().isoformat()
        }
        
        employees.append(new_employee)
        save_employees(employees)
        
        return jsonify({'success': True, 'message': f'Employee registered successfully with {len(photo_filenames)} photo(s)!'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/mark_attendance', methods=['POST'])
def mark_attendance():
    """Mark attendance IN or OUT"""
    try:
        data = request.get_json()
        emp_id = data['emp_id']
        
        # Check if employee exists
        employees = load_employees()
        employee = next((emp for emp in employees if emp['emp_id'] == emp_id), None)
        
        if not employee:
            return jsonify({'success': False, 'message': 'Employee ID not found!'})
        
        # Load attendance records
        attendance = load_attendance()
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Check last attendance status for today
        today_records = [r for r in attendance if r['emp_id'] == emp_id and r['date'] == today]
        
        if today_records:
            last_record = today_records[-1]
            if last_record.get('time_out'):
                # Last record already has OUT - cannot mark OUT again
                return jsonify({
                    'success': False,
                    'message': f'{employee["name"]} already marked OUT today at {last_record["time_out"]}. Cannot mark attendance again for today.'
                })
            else:
                # Last record has IN but no OUT, so mark OUT
                status = 'OUT'
                time_key = 'time_out'
                # Update the last record with OUT time
                last_record['time_out'] = datetime.now().strftime('%H:%M:%S')
                last_record['status'] = 'Complete'
                save_attendance(attendance)
                
                return jsonify({
                    'success': True,
                    'message': f'Time OUT marked for {employee["name"]}',
                    'status': 'OUT',
                    'employee': employee,
                    'time': last_record['time_out']
                })
        else:
            # No record for today, mark IN
            status = 'IN'
            time_key = 'time_in'
        
        # Create new attendance record
        new_record = {
            'emp_id': emp_id,
            'name': employee['name'],
            'location': employee['location'],
            'shift': employee['shift'],
            'date': today,
            'time_in': datetime.now().strftime('%H:%M:%S') if status == 'IN' else None,
            'time_out': None,
            'status': 'Pending',
            'timestamp': datetime.now().isoformat()
        }
        
        attendance.append(new_record)
        save_attendance(attendance)
        
        return jsonify({
            'success': True,
            'message': f'Time IN marked for {employee["name"]}',
            'status': status,
            'employee': employee,
            'time': new_record['time_in']
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/employees')
def get_employees():
    """Get all employees"""
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
                last_record = today_records[-1]
                # Check if they have clocked out
                if last_record.get('time_out'):
                    status = 'OUT'
                    time = last_record['time_out']
                else:
                    status = 'IN'
                    time = last_record.get('time_in', '')
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

@app.route('/api/employee/<emp_id>')
def get_employee(emp_id):
    """Get specific employee details"""
    employees = load_employees()
    employee = next((emp for emp in employees if emp['emp_id'] == emp_id), None)
    
    if employee:
        # Get attendance records for this employee
        attendance = load_attendance()
        emp_attendance = [r for r in attendance if r['emp_id'] == emp_id]
        
        return jsonify({
            'success': True,
            'employee': employee,
            'attendance': emp_attendance
        })
    
    return jsonify({'success': False, 'message': 'Employee not found'})

@app.route('/api/update_employee/<emp_id>', methods=['PUT'])
def update_employee(emp_id):
    """Update employee details"""
    try:
        data = request.get_json()
        employees = load_employees()
        
        # Find employee
        emp_index = next((i for i, emp in enumerate(employees) if emp['emp_id'] == emp_id), None)
        
        if emp_index is None:
            return jsonify({'success': False, 'message': 'Employee not found'})
        
        # Update employee data
        employees[emp_index]['name'] = data.get('name', employees[emp_index]['name'])
        employees[emp_index]['location'] = data.get('location', employees[emp_index]['location'])
        employees[emp_index]['shift'] = data.get('shift', employees[emp_index]['shift'])
        
        # Update photo if provided
        if 'photo' in data and data['photo']:
            photo_data = data['photo'].split(',')[1]
            photo_bytes = base64.b64decode(photo_data)
            photo_filename = f"{emp_id}.jpg"
            photo_path = os.path.join(PHOTOS_DIR, photo_filename)
            
            with open(photo_path, 'wb') as f:
                f.write(photo_bytes)
            
            employees[emp_index]['photo'] = photo_filename
        
        save_employees(employees)
        
        return jsonify({'success': True, 'message': 'Employee updated successfully!'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/delete_employee/<emp_id>', methods=['DELETE'])
def delete_employee(emp_id):
    """Delete employee"""
    try:
        employees = load_employees()
        
        # Find and remove employee
        employees = [emp for emp in employees if emp['emp_id'] != emp_id]
        save_employees(employees)
        
        # Delete photo if exists
        photo_path = os.path.join(PHOTOS_DIR, f"{emp_id}.jpg")
        if os.path.exists(photo_path):
            os.remove(photo_path)
        
        return jsonify({'success': True, 'message': 'Employee deleted successfully!'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/attendance')
def get_attendance():
    """Get all attendance records"""
    attendance = load_attendance()
    return jsonify(attendance)

@app.route('/api/attendance/filter', methods=['POST'])
def filter_attendance():
    """Filter attendance by date range"""
    data = request.get_json()
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    attendance = load_attendance()
    
    if start_date and end_date:
        filtered = [r for r in attendance if start_date <= r['date'] <= end_date]
    else:
        filtered = attendance
    
    return jsonify(filtered)

@app.route('/api/export/excel', methods=['POST'])
def export_excel():
    """Export attendance to Excel"""
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
        
        # Create DataFrame
        df = pd.DataFrame(filtered)
        df = df[['date', 'emp_id', 'name', 'location', 'shift', 'time_in', 'time_out', 'status']]
        df.columns = ['Date', 'Employee ID', 'Name', 'Location', 'Shift', 'Time IN', 'Time OUT', 'Status']
        
        # Create Excel file in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Attendance')
        output.seek(0)
        
        filename = f'attendance_{start_date}_to_{end_date}.xlsx'
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )
    
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
        
        # Create DataFrame
        df = pd.DataFrame(filtered)
        df = df[['date', 'emp_id', 'name', 'location', 'shift', 'time_in', 'time_out', 'status']]
        df.columns = ['Date', 'Employee ID', 'Name', 'Location', 'Shift', 'Time IN', 'Time OUT', 'Status']
        
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

@app.route('/api/employee_photo/<emp_id>')
def get_employee_photo(emp_id):
    """Get employee photo"""
    photo_path = os.path.join(PHOTOS_DIR, f'{emp_id}.jpg')
    if os.path.exists(photo_path):
        return send_file(photo_path, mimetype='image/jpeg')
    return '', 404

@app.route('/api/employee_attendance/<emp_id>')
def get_employee_attendance(emp_id):
    """Get attendance records for a specific employee"""
    try:
        attendance = load_attendance()
        employee_records = [record for record in attendance if record['emp_id'] == emp_id]
        return jsonify({'success': True, 'attendance': employee_records})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/export_employee_attendance')
def export_employee_attendance():
    """Export attendance for a specific employee"""
    try:
        emp_id = request.args.get('emp_id')
        from_date = request.args.get('from_date')
        to_date = request.args.get('to_date')
        
        if not emp_id:
            return jsonify({'success': False, 'message': 'Employee ID required'})
        
        # Get employee info
        employees = load_employees()
        employee = next((e for e in employees if e['emp_id'] == emp_id), None)
        
        if not employee:
            return jsonify({'success': False, 'message': 'Employee not found'})
        
        # Get attendance records
        attendance = load_attendance()
        records = [r for r in attendance if r['emp_id'] == emp_id]
        
        # Filter by date if provided
        if from_date and to_date:
            records = [r for r in records if from_date <= r['date'] <= to_date]
        
        if not records:
            return jsonify({'success': False, 'message': 'No records found'})
        
        # Group by date
        grouped = {}
        for record in records:
            date = record['date']
            if date not in grouped:
                grouped[date] = {'in': '-', 'out': '-'}
            if record['type'] == 'IN':
                grouped[date]['in'] = record['time']
            else:
                grouped[date]['out'] = record['time']
        
        # Create Excel file
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
        
        # Create Excel file in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Attendance')
            worksheet = writer.sheets['Attendance']
            
            # Adjust column widths
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
