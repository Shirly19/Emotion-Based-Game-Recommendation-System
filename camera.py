import numpy as np
import cv2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import swish
from tensorflow.keras.layers import Dropout

class FixedDropout(Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=False)

import time
from threading import Thread
import logging

# Load face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Emotion labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load emotion detection model
emotion_model = load_model('models/emotion_model_full_local.h5', custom_objects={'swish': swish, 'FixedDropout': FixedDropout})

cv2.ocl.setUseOpenCL(False)

# Globals
detected_emotion = "None"
emotion_history = []
full_emotion_history = []
detection_start_time = time.time()

# Logging
logging.basicConfig(filename='emotion_detection.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

emotion_change_callbacks = []

def register_emotion_change_callback(callback):
    emotion_change_callbacks.append(callback)

def notify_emotion_change(new_emotion):
    logging.info(f"Emotion changed to: {new_emotion}")
    for callback in emotion_change_callbacks:
        callback(new_emotion)

# Webcam thread
class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.stream.isOpened():
            raise RuntimeError(f"Could not open video source {src}")
        try:
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        except Exception as e:
            print(f"Warning: Could not set frame width: {e}")
        try:
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception as e:
            print(f"Warning: Could not set frame height: {e}")
        try:
            self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception as e:
            print(f"Warning: Could not set FOURCC: {e}")
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.stream.release()

# Main camera logic
class VideoCamera(object):
    def __init__(self):
        self.cap = WebcamVideoStream(src=0).start()

    def __del__(self):
        self.cap.stop()

    def get_frame(self):
        global detected_emotion, emotion_history, full_emotion_history, detection_start_time

        frame = self.cap.read()

        # No color correction applied here — pure webcam frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

        current_time = time.time()
        if current_time - detection_start_time >= 10:
            if emotion_history:
                new_emotion = max(set(emotion_history), key=emotion_history.count)
                if new_emotion != detected_emotion:
                    detected_emotion = new_emotion
                    notify_emotion_change(detected_emotion)
                full_emotion_history.append(detected_emotion)
                emotion_history.clear()
            detection_start_time = current_time

        for (x, y, w, h) in face_rects:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 2)
            roi_color = frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(cv2.resize(roi_color, (224, 224)), 0).astype('float32') / 255.0

            try:
                prediction = emotion_model.predict(cropped_img, verbose=0)
                maxindex = int(np.argmax(prediction))
                predicted_emotion = emotion_dict[maxindex]
                emotion_history.append(predicted_emotion)
                # Removed text display on frame
            except Exception as e:
                print("Prediction Error:", e)

        # Convert frame to JPEG to stream over Flask
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

# Emotion access
def get_current_emotion():
    global detected_emotion
    return detected_emotion

def get_emotion_history():
    global full_emotion_history
    return full_emotion_history

def on_emotion_change(new_emotion):
    pass

register_emotion_change_callback(on_emotion_change)
