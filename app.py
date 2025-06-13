import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import cv2
import numpy as np
from PIL import Image
import joblib
import requests
import time
from collections import deque

# Load model
model = joblib.load("best_modelCNN.joblib")

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Telegram configuration
BOT_TOKEN = st.secrets["telegram"]["bot_token"]
CHAT_ID = st.secrets["telegram"]["chat_id"]

# Alert threshold
FRAME_WINDOW = 30       # Number of recent frames to check
DRUNK_THRESHOLD = 10    # Trigger alert if ≥ this number of drunk frames
COOLDOWN_SECONDS = 30   # Time between alerts


# Send alert via Telegram
def send_telegram_alert(message: str, image=None):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    requests.post(url, data=payload)

    # Optional: send image
    if image is not None:
        photo_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        _, buf = cv2.imencode(".jpg", image)
        files = {"photo": buf.tobytes()}
        data = {"chat_id": CHAT_ID, "caption": message}
        requests.post(photo_url, data=data, files={"photo": ("image.jpg", buf.tobytes())})


# Video transformer with state
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.predictions = deque(maxlen=FRAME_WINDOW)
        self.last_alert_time = 0
        self.alert_sent = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        frame_has_drunk = False

        for (x, y, w, h) in faces:
            face_img = img[y:y+h, x:x+w]
            if face_img.size == 0:
                continue

            face_resized = cv2.resize(face_img, (224, 224))
            face_array = face_resized.astype(np.float32) / 255.0
            face_array = face_array.reshape(1, 224, 224, 3)

            pred = model.predict(face_array)
            label = np.argmax(pred, axis=1)[0]
            label_str = "Drunk" if label == 0 else "Sober"

            if label_str == "Drunk":
                frame_has_drunk = True

            color = (0, 0, 255) if label_str == "Drunk" else (0, 255, 0)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label_str, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Update prediction history
        self.predictions.append(1 if frame_has_drunk else 0)

        # Check alert condition
        if sum(self.predictions) >= DRUNK_THRESHOLD:
            now = time.time()
            if now - self.last_alert_time > COOLDOWN_SECONDS:
                send_telegram_alert("🚨 ALERT: Drunk person detected in live webcam!", image=img)
                self.last_alert_time = now
                self.alert_sent = True
        else:
            self.alert_sent = False

        return img


# Streamlit interface
st.title("CekMabuk AI")
st.markdown("LAI25-SM008")

webrtc_streamer(
    key="live-drunk-detector",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
