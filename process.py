from dependencies import *

def extract_audio(video_path, audio_path):
    '''
    1. 무비 파일 경로\n
    video_path = 'data/원천데이터/2~5분/test.mp4'
    audio_path = 'audio.wav'

    2. 오디오 추출\n
    extract_audio(video_path, audio_path)
    '''

    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path, verbose=False, logger=None)
    video_clip.close()
    
    return audio_path
    
    
def preprocess_audio(audio_path, sample_rate=22050, n_fft=2048, hop_length=512, n_mels=130, segment_duration=3):
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    
    segment_length = int(sr * segment_duration)
    
    segments = []
    num_segments = len(audio) // segment_length
    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        segment = audio[start_idx:end_idx]
        
        mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        segments.append(mel_spectrogram_db)
    
    return np.array(segments)


def preprocess_video_every_3_seconds(video_path: str, frame_size: tuple, frame_rate=3):
    """
    Extracts frames every 3 seconds from a video file, resizing them to frame_size and converting to grayscale.
    
    Args:
    video_path (str): Path to the video file.
    frame_size (tuple): Size (height, width) to resize frames.
    frame_rate (int): Number of frames to extract per second within the 3-second window.

    Returns:
    List[numpy.ndarray]: List of sequences, where each sequence is a numpy array of shape (num_frames, height, width, 1).
    """
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * 3)
    target_frames = int(frame_rate * 3)
    sequences = []

    def read_frames(q):
        while True:
            success, frame = vidcap.read()
            if not success:
                q.put(None)
                break
            q.put(frame)

    frame_queue = queue.Queue(maxsize=100)
    threading.Thread(target=read_frames, args=(frame_queue,)).start()

    while True:
        frames = []
        for _ in range(interval_frames):
            frame = frame_queue.get()
            if frame is None:
                break
            frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = np.expand_dims(gray_frame, axis=-1)
            gray_frame = gray_frame.astype(np.float32) / 255.0
            frames.append(gray_frame)
        
        if len(frames) < interval_frames:
            break
        
        sequences.append(np.array(frames[:target_frames]))
    
    vidcap.release()
    return np.array(sequences)


def pipeline_video(video_path:str):

    if not os.path.exists(video_path):
        print(f"Video Not Found : {video_path}")
        return
    
    audio = extract_audio(video_path, './test.wav')
    audio = preprocess_audio(audio)

    video = preprocess_video_every_3_seconds(video_path, (256, 256), 3)

    print(len(video))
    print(len(audio))

    video_model = load_model("video_3D_model.h5")
    audio_model = load_model("audio_comp_model.h5")

    print("Successfly Load Model &  Start to Predict")
    video_output = video_model.predict(video)
    audio_output = audio_model.predict(audio)

    return video_output, audio_output

def process_video_data(video_path):
    video_data, audio_data = pipeline_video(video_path)
    
    # Prepare audio data
    new_audio_data = np.zeros((audio_data.shape[0], audio_data.shape[1] + 1))
    for i, audio_row in enumerate(audio_data):
        half_value = audio_row[1] / 2
        new_audio_data[i][0] = round(audio_row[0], 5)
        new_audio_data[i][1] = round(half_value, 5)
        new_audio_data[i][2] = round(half_value, 5)
    
    # Prepare video data
    new_video_data = np.round(video_data, 5)

    return new_video_data, new_audio_data

def preprocess_shorts_only_frame(video_path: str, label: list, output_path: str):
    vidcap = cv2.VideoCapture(video_path)
    
    if not vidcap.isOpened():
        print("Error: Could not open video.")
        return
    
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Could not retrieve FPS from video.")
        vidcap.release()
        return
    
    interval = int(fps * 3)  # 3초 단위로 분리
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    sequences = []

    total_labels = len(label)
    
    for idx, lbl in enumerate(label):
        index = lbl[0]
        start_frame = float(index * interval)
        
        print(f"Processing index {index}, start_frame {start_frame}")

        if start_frame >= total_frames:
            print(f"Warning: start_frame {start_frame} is out of bounds for video with {total_frames} frames.")
            continue
        
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        current_pos = vidcap.get(cv2.CAP_PROP_POS_FRAMES)
        if current_pos != start_frame:
            print(f"Error: Could not set video to start frame {start_frame}. Current position is {current_pos}.")
            continue

        frames = []
        for _ in range(interval):
            success, frame = vidcap.read()
            if not success:
                print("Warning: Could not read frame. Ending segment early.")
                break
            frames.append(frame)
        
        if len(frames) == interval:
            sequences.extend(frames)

    if sequences:
        height, width, layers = sequences[0].shape
        size = (width, height)
        if platform.system() == "Windows":
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        elif platform.system() == "Darwin":
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'X264'), fps, size)

        for frame in sequences:
            out.write(frame)
        
        out.release()
    
    vidcap.release()

    def get_max_values_and_indices(video_data, audio_data, video_weight, audio_weight, threshold, video_length, ratio=None, outro_length=0):
    
        # Ensure both arrays have the same length by truncating to the shortest length
        if outro_length != 0:
            outro = outro_length // 3
            min_length = min(video_data.shape[0], audio_data.shape[0]) - outro
        else:
            min_length = min(video_data.shape[0], audio_data.shape[0])
            
        video_data = video_data[:min_length]
        audio_data = audio_data[:min_length]
        
        if video_length == -1:
            video_length = int(min_length * ratio)
            print(video_length)
        else:
            video_length = (video_length // 3) # 가장 큰 3초 단위로 정리
        
        # Compute ensemble scores
        ensemble_scores = (video_data * video_weight + audio_data * audio_weight) / (video_weight + audio_weight)
        ensemble_labels = ensemble_scores.argmax(axis=1)

        # Apply threshold to label "2"
        high_confidence_twos = ensemble_scores[:, 2] >= threshold
        ensemble_labels[high_confidence_twos] = 2
        
        # Format output as (i, label, score)
        output = [(i, ensemble_labels[i], max(ensemble_scores[i])) for i in range(min_length)]
        
        sorted_data = sorted(output, key=lambda x: (x[1], x[2]), reverse=True)
        sorted_data = sorted(sorted_data[:video_length], key=lambda x: x[0])
        print(len(sorted_data))
        return sorted_data
