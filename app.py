from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
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

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.before_request
def log_request_info():
    app.logger.debug('Headers: %s', request.headers)
    app.logger.debug('Body: %s', request.get_data())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_shorts', methods=['POST'])
def upload_file():
    if 'videoUpload' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['videoUpload']
    video_length = int(request.form['videoLength'])
    outro_length = int(request.form['outroLength'])

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        return redirect(url_for('loading', filename=filename, video_length=video_length, outro_length=outro_length))

@app.route('/loading')
def loading():
    filename = request.args.get('filename')
    video_length = request.args.get('video_length')
    outro_length = request.args.get('outro_length')
    
    return render_template('loading.html', filename=filename, video_length=video_length, outro_length=outro_length)

@app.route('/process_video', methods=['GET'])
def process_video():
    filename = request.args.get('filename')
    video_length = int(request.args.get('video_length'))
    outro_length = int(request.args.get('outro_length'))
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(video_path):
        return jsonify({'error': 'File not found'}), 404

    # Processing logic here...
    video_ratio = 0
    video_weight = 0.80
    audio_weight = 0.36
    threshold = 0.50

    new_video_data, new_audio_data = process_video_data(video_path)
    sorted_data = get_max_values_and_indices(new_video_data, new_audio_data, video_weight, audio_weight, threshold, video_length, video_ratio, outro_length)

    current_time = str(datetime.now().strftime("%Y%m%d_%H%M%S")) + ".mp4"
    final_output_path = os.path.join(app.config['PROCESSED_FOLDER'], current_time)

    preprocess_shorts_only_frame(video_path, sorted_data, final_output_path)
    
    # Return a JSON response indicating that processing is complete
    return jsonify({'status': 'complete', 'filename': current_time})


@app.route('/results', methods=['GET'])
def results():
    filename = request.args.get('filename')
    return render_template('results.html', filename=filename)

@app.route('/processed_videos/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/shorts')
def shorts():
    return render_template('shorts.html')

@app.errorhandler(400)
def bad_request_error(error):
    return jsonify({'error': 'Bad Request'}), 400

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not Found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=False)
