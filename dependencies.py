import os
import cv2
import numpy as np
import pandas as pd
import platform

if platform.system() == "Darwin":
    os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
    os.environ['FFMPEG_BINARY'] = "/opt/homebrew/bin/ffmpeg"

import threading
import queue

import librosa 
import librosa.display as dsp
from IPython.display import Audio

from moviepy.editor import VideoFileClip
from tensorflow.keras.models import load_model