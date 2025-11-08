# realtime_demo_optimized.py
import tensorflow as tf
print(" Devices:", tf.config.list_physical_devices())

import cv2
import numpy as np
import time
from tensorflow.keras.preprocessing.image import img_to_array
from mental_state_module import get_mental_state, estimate_stress_score

# --- Load Model ---
model = tf.keras.models.load_model("mobilenetv2_fer_final_optimized.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
print(" Model loaded successfully.")

# --- Define Emotion Colors ---
colors = {
    "Angry": (0, 0, 255),
    "Disgust": (153, 51, 255),
    "Fear": (255, 0, 255),
    "Happy": (0, 255, 0),
    "Sad": (255, 0, 0),
    "Surprise": (255, 255, 0),
    "Neutral": (200, 200, 200)
}

# --- Initialize Face Detector ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Preprocessing Function ---
def preprocess_face(face):
    face = cv2.resize(face, (224, 224))
    face = face.astype("float") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    return face

# --- Initialize Webcam ---
cap = cv2.VideoCapture(0)
print("🎥 Press 'q' to quit the demo.")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        face = preprocess_face(roi)
        preds = model.predict(face, verbose=0)[0]
        label = emotion_labels[np.argmax(preds)]
        confidence = np.max(preds) * 100
        mental_state = get_mental_state(label)
        stress_level = estimate_stress_score(preds, emotion_labels)

        # Choose color based on emotion
        color = colors.get(label, (0, 255, 0))

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Display text info
        text = f"{label} ({confidence:.1f}%) | {mental_state} | Stress: {stress_level}"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- FPS Calculation ---
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # --- Display Frame ---
    cv2.imshow("Facial Emotion & Mental State Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(" Demo closed.")
