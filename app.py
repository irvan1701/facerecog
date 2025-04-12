import streamlit as st
import cv2
import dlib
import numpy as np
import pickle
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from streamlit_autorefresh import st_autorefresh

# Load model yang sudah dilatih
with open("svm_model.pkl", "rb") as model_file:
    svm = pickle.load(model_file)

with open("label_encoder.pkl", "rb") as le_file:
    label_encoder = pickle.load(le_file)

# Debug: Tampilkan label dari model
print("Label dari model:", label_encoder.classes_)

# Inisialisasi face detector dan shape predictor
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

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
        print("Jumlah wajah terdeteksi:", len(faces))

        self.detected_names.clear()

        for face in faces:
            try:
                shape = sp(rgb_frame, face)
                face_descriptor = np.array(face_rec_model.compute_face_descriptor(rgb_frame, shape)).reshape(1, -1)

                pred_label = svm.predict(face_descriptor)[0]
                pred_name = label_encoder.inverse_transform([pred_label])[0]
                print("Nama terdeteksi:", pred_name)

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

# Auto-refresh setiap 1000 ms (1 detik)
st_autorefresh(interval=1000, key="refresh")

# Tampilkan nama-nama yang terdeteksi
if ctx.video_processor:
    st.subheader("🧑‍🤝‍🧑 Nama yang terdeteksi:")
    names = list(ctx.video_processor.detected_names)
    if names:
        for name in names:
            st.success(f"✔️ {name}")
    else:
        st.info("Belum ada wajah yang dikenali.")
