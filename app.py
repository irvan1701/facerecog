import streamlit as st
import cv2
import dlib
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from collections import defaultdict

# Inisialisasi face detector dan shape predictor
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Dataset wajah yang sudah ada (contoh data manual)
# Biasanya data ini adalah hasil ekstraksi fitur wajah yang sebelumnya telah disimpan dalam array
known_face_descriptors = [
    # Contoh: deskriptor wajah orang yang sudah ada
    np.random.rand(128),  # Representasi wajah orang 1
    np.random.rand(128)   # Representasi wajah orang 2
]

known_face_names = [
    "Person 1",
    "Person 2"
]

# Class untuk deteksi dan prediksi wajah
class FaceRecognitionTransformer(VideoTransformerBase):
    def __init__(self):
        self.detected_names = set()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if img is None:
            print("Frame kosong")
            return img

        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector(rgb_frame)

        self.detected_names.clear()

        for face in faces:
            try:
                shape = sp(rgb_frame, face)
                face_descriptor = np.array(face_rec_model.compute_face_descriptor(rgb_frame, shape))

                # Mencocokkan deskriptor wajah dengan dataset wajah yang sudah ada
                distances = np.linalg.norm(known_face_descriptors - face_descriptor, axis=1)
                min_distance_index = np.argmin(distances)

                # Jika jarak lebih kecil dari threshold, deteksi wajah sesuai nama
                if distances[min_distance_index] < 0.6:  # Threshold untuk kecocokan wajah
                    pred_name = known_face_names[min_distance_index]
                    self.detected_names.add(pred_name)

                # Gambar kotak dan nama
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, pred_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            except Exception as e:
                print("Error saat deskripsi wajah:", e)

        return img

# Streamlit UI
st.set_page_config(page_title="Face Recognition Attendance", layout="centered")
st.title("🎥 Face Recognition for Attendance")
st.write("Arahkan wajah ke kamera...")

ctx = webrtc_streamer(
    key="face-recognition",
    video_processor_factory=FaceRecognitionTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Tampilkan nama-nama yang terdeteksi
if ctx.video_processor:
    st.subheader("🧑‍🤝‍🧑 Nama yang terdeteksi:")
    names = list(ctx.video_processor.detected_names)
    if names:
        for name in names:
            st.success(f"✔️ {name}")
    else:
        st.info("Belum ada wajah yang dikenali.")
