import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib
import time
import requests

# Load model
model = joblib.load('best_modelCNN.joblib')

# Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("Periodic Webcam Drunk Detection (Every 5 Seconds)")

# Checkbox to start/stop detection loop
if "detecting" not in st.session_state:
    st.session_state.detecting = False

if st.button("Start Detection"):
    st.session_state.detecting = True
    st.rerun()

if st.button("Stop Detection"):
    st.session_state.detecting = False

if st.session_state.detecting:
    st.info("⏳ Capturing webcam every 5 seconds...")

    img_file = st.camera_input("Auto Capture (takes 1 photo every 5 sec)")

    if img_file is not None:
        img = Image.open(img_file)
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        st.image(img, caption="Captured Image", use_column_width=True)

        if len(faces) == 0:
            st.warning("No face detected.")
        else:
            for i, (x, y, w, h) in enumerate(faces):
                face_img = img_np[y:y+h, x:x+w]
                st.image(face_img, caption=f"Face #{i+1}", width=150)

                # Preprocess
                face_resized = Image.fromarray(face_img).convert('RGB').resize((224, 224))
                face_array = np.array(face_resized) / 255.0
                face_array = face_array.reshape(1, 224, 224, 3)

                # Predict
                pred = model.predict(face_array)
                label = np.argmax(pred, axis=1)[0]
                label_str = "Drunk" if label == 0 else "Sober"
                st.write(f"Prediction: **{label_str}**")

                # Optional: Telegram Alert
                if label_str == "Drunk":
                    st.warning("⚠️ Drunk person detected.")
                    bot_token = st.secrets["telegram"]["bot_token"]
                    chat_id = st.secrets["telegram"]["chat_id"]

                    from io import BytesIO
                    buf = BytesIO()
                    img.save(buf, format='JPEG')
                    buf.seek(0)
                    requests.post(
                        f"https://api.telegram.org/bot{bot_token}/sendPhoto",
                        data={'chat_id': chat_id, 'caption': '🚨 Drunk person detected from webcam auto-capture'},
                        files={'photo': buf}
                    )

    time.sleep(5)
    st.experimental_rerun()
