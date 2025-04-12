import streamlit as st
import cv2
import dlib
import numpy as np
import pickle
from PIL import Image
import time
import pandas as pd

# Load model-model
@st.cache_resource
def load_models():
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    
    with open("svm_model.pkl", "rb") as f:
        svm = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return detector, sp, facerec, svm, label_encoder

detector, sp, facerec, svm, label_encoder = load_models()

st.title("🎥 Real-Time Face Recognition with Dlib & Streamlit")

# Create a container for live video capture
frame_window = st.empty()

# Create a container for displaying detected names
names_container = st.empty()

# Start capturing from webcam
cap = cv2.VideoCapture(0)

detected_names = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_frame)

    # Clear previous detected names each frame
    frame_detected_names = set()

    # Process each face detected
    for face in faces:
        shape = sp(rgb_frame, face)
        face_descriptor = facerec.compute_face_descriptor(rgb_frame, shape)
        face_descriptor = np.array(face_descriptor).reshape(1, -1)

        pred_label = svm.predict(face_descriptor)[0]
        pred_name = label_encoder.inverse_transform([pred_label])[0]

        frame_detected_names.add(pred_name)

        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(rgb_frame, pred_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Add current frame's detected names to the global set (unique names)
    detected_names.update(frame_detected_names)

    # Convert to Image and display in the Streamlit container
    image = Image.fromarray(rgb_frame)
    frame_window.image(image, channels="RGB", use_container_width=True)

    # Update the table of detected names
    if detected_names:
        detected_names_df = pd.DataFrame(list(detected_names), columns=["Detected Names"])
        names_container.dataframe(detected_names_df)

    # Delay for better frame rate
    time.sleep(0.1)

cap.release()
cv2.destroyAllWindows()
