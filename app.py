import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import cv2
import numpy as np
from PIL import Image
import joblib

# Load drunk vs sober model
model = joblib.load('best_modelCNN.joblib')

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Define the transformer
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

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

            # Draw rectangle and label
            color = (0, 0, 255) if label_str == "Drunk" else (0, 255, 0)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label_str, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return img


# Streamlit UI
st.title("Live Webcam Drunk Face Detector")

webrtc_streamer(
    key="drunk-detector",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
