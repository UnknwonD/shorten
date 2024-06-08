from flask import Flask, request, render_template, redirect, url_for, send_from_directory,jsonify
import os
from werkzeug.utils import secure_filename
import cv2
import platform
from datetime import datetime

from process import process_video_data, get_max_values_and_indices, preprocess_shorts_only_frame

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed_videos'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_shorts', methods=['POST'])
def upload_file():
    if 'videoUpload' not in request.files:
        return redirect(request.url)
    file = request.files['videoUpload']
    video_length = request.form['videoLength']
    outro_length = request.form['outroLength']
    output_path = request.form['outputPath']
    video_ratio = 0
    video_weight = 0.80
    audio_weight = 0.36
    threshold = 0.50

    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        new_video_data, new_audio_data = process_video_data(file_path)
        sorted_data = get_max_values_and_indices(new_video_data, new_audio_data, video_weight, audio_weight, threshold, video_length, video_ratio, outro_length)

        current_time = str(datetime.now().strftime("%Y%m%d_%H%M%S")) + ".mp4"
        final_output_path = os.path.join(app.config['PROCESSED_FOLDER'], current_time)
        
        preprocess_shorts_only_frame(file_path, sorted_data, final_output_path)
        
        return jsonify({'filename': current_time})

@app.route('/processed_videos/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

def process_video(file_path, video_length):
    # Your preprocessing code here
    # Example: Convert video to grayscale and save the processed video
    cap = cv2.VideoCapture(file_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'processed_{os.path.basename(file_path)}', fourcc, 20.0, (640, 480), False)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(gray)
    cap.release()
    out.release()

if __name__ == '__main__':
    app.run(debug=True)
