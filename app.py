import streamlit as st
import cv2
import dlib
import numpy as np
import pickle
import time

# Load model yang sudah dilatih (ganti dengan path model Anda)
with open('svm_model.pkl', 'rb') as model_file:
    svm = pickle.load(model_file)

with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Ganti dengan path predictor Anda

# Initialize a set to store the names of detected faces
detected_names = set()

# Streamlit UI
st.title('Face Recognition for Attendance')

# Menampilkan instruksi di Streamlit
st.write("Menunggu deteksi wajah...")

# Kondisi untuk menjalankan kode kamera hanya jika di lingkungan yang mendukungnya
cap = None
if 'camera_available' in st.session_state and st.session_state.camera_available:
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Camera not accessible.")
    except Exception as e:
        st.warning(f"Camera error: {e}")
        cap = None  # Tidak ada kamera, set cap ke None

# Menangani deteksi wajah (Jika kamera tidak ada, bisa diganti dengan input statis)
if cap:
    stframe = st.empty()  # Tempatkan frame untuk menampilkan hasil deteksi wajah
    while True:
        success, frame = cap.read()
        if not success:
            st.warning("Gagal membaca frame dari kamera.")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector(rgb_frame)

        detected_names.clear()

        for face in faces:
            shape = sp(rgb_frame, face)
            face_descriptor = dlib.face_descriptor(rgb_frame, shape)
            face_descriptor = np.array(face_descriptor).reshape(1, -1)

            # Prediksi menggunakan SVM dan LabelEncoder
            pred_label = svm.predict(face_descriptor)[0]
            pred_name = label_encoder.inverse_transform([pred_label])[0]

            detected_names.add(pred_name)

            # Gambar kotak wajah dan nama di frame
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, pred_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Menampilkan hasil di Streamlit
        stframe.image(frame, channels="RGB", use_column_width=True)

        # Tampilkan daftar nama yang terdeteksi
        st.subheader("Nama yang terdeteksi:")
        for name in detected_names:
            st.write(name)

        # Beri waktu untuk refresh agar frame tidak terlalu cepat (opsional)
        time.sleep(0.1)

    cap.release()
else:
    st.warning("Tidak ada akses ke kamera, menggunakan input statis atau file.")

# Hapus jendela OpenCV jika ada (tidak dibutuhkan dalam headless environment)
# cv2.destroyAllWindows()  # Hapus ini untuk deployment headless
