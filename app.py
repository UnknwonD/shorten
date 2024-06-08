from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'videoUpload' not in request.files:
        return redirect(request.url)
    file = request.files['videoUpload']
    video_length = request.form['videoLength']

    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        process_video(file_path, int(video_length))
        return redirect(url_for('index'))

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
