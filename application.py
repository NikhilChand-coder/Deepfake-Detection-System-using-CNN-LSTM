# ===============================================
# Deepfake Detection Streamlit App
# Models: Xception (Images) + Xception + LSTM (Videos)
# ===============================================

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import os
import matplotlib.pyplot as plt

# ===============================================
# Load Models
# ===============================================
@st.cache_resource
def load_models():
    image_model = tf.keras.models.load_model("saved_models/final_image_deepfake_model_xception.h5", compile=False)
    video_model = tf.keras.models.load_model("saved_models/final_video_deepfake_model_xception.h5", compile=False)
    return image_model, video_model

image_model, video_model = load_models()

# ===============================================
# Helper Functions
# ===============================================

IMG_SIZE = (299, 299)
FRAMES = 10

def preprocess_image(image):
    """Preprocess single image for Xception."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMG_SIZE)
    return image / 255.0

def predict_image(image_model, img_array):
    """Predict real/fake for image."""
    img_exp = np.expand_dims(img_array, axis=0)
    pred = image_model.predict(img_exp)[0][0]
    label = "Real" if pred > 0.5 else "Fake"
    confidence = pred if label == "Real" else 1 - pred
    return label, float(confidence)

def sample_video_frames(video_path, num_frames=FRAMES):
    """Extract sample frames from video for prediction."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, num_frames).astype(int)

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in idxs:
            frame = preprocess_image(frame)
            frames.append(frame)
    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1])
    return np.array(frames)

def predict_video(video_model, video_path):
    """Predict real/fake for video using sampled frames."""
    frames = sample_video_frames(video_path)
    frames_exp = np.expand_dims(frames, axis=0)
    pred = video_model.predict(frames_exp)[0][0]
    label = "Fake" if pred > 0.5 else "Real"
    confidence = pred if label == "Fake" else 1 - pred
    return label, float(confidence)

# ===============================================
# Streamlit UI
# ===============================================
st.set_page_config(page_title="Deepfake Detection (Image + Video)", layout="centered")

st.title("🧠 Deepfake Detection")
st.write("Upload an **image** or **video** to detect whether it's *Real* or *Fake*.")

mode = st.radio("Select Input Type:", ["Image", "Video"], horizontal=True)

# ===============================================
# Image Mode
# ===============================================
if mode == "Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", width=300)   # Reduce width (e.g., 300px)


        if st.button(" Analyze Image"):
            with st.spinner("Analyzing image..."):
                img_array = preprocess_image(image)
                label, confidence = predict_image(image_model, img_array)

            if label == "Real":
                st.success(f" Prediction: **{label}** ({confidence*100:.2f}% confidence)")
            else:
                st.error(f" Prediction: **{label}** ({confidence*100:.2f}% confidence)")

            # Confidence bar chart
            st.subheader("Prediction Confidence")
            fig, ax = plt.subplots(figsize=(3, 2))
            categories = ['Real', 'Fake']
            values = [confidence*100, (1-confidence)*100] if label == 'Real' else [(1-confidence)*100, confidence*100]
            ax.bar(categories, values, color=['green', 'red'])
            ax.set_ylim(0, 100)
            ax.set_ylabel("Confidence (%)")
            st.pyplot(fig)

# ===============================================
# Video Mode
# ===============================================
elif mode == "Video":
    uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        st.video(tfile.name)

        if st.button(" Analyze Video"):
            with st.spinner("Analyzing video frames..."):
                label, confidence = predict_video(video_model, tfile.name)

            if label == "Real":
                st.success(f" Prediction: **{label}** ({confidence*100:.2f}% confidence)")
            else:
                st.error(f" Prediction: **{label}** ({confidence*100:.2f}% confidence)")

            st.subheader("Prediction Confidence")
            fig, ax = plt.subplots(figsize=(3, 2))
            categories = ['Real', 'Fake']
            values = [confidence*100, (1-confidence)*100] if label == 'Real' else [(1-confidence)*100, confidence*100]
            ax.bar(categories, values, color=['green', 'red'])
            ax.set_ylim(0, 100)
            ax.set_ylabel("Confidence (%)")
            st.pyplot(fig)
