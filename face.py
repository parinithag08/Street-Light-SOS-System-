import cv2
import numpy as np
from tensorflow.keras.models import load_model

# -------------------------------
# Load Haar Cascade for face detection
# -------------------------------
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# -------------------------------
# Load pre-trained emotion model
# (FER2013 Mini_XCEPTION trained)
# -------------------------------
model = load_model("emotion_model.h5", compile=False)

# Emotion labels (adjust if model has more classes)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# -------------------------------
# Start camera
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("? Cannot access camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("? Failed to grab frame")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face ROI
        roi_gray = gray[y:y+h, x:x+w]

        # Resize to match model input (64x64)
        roi_gray = cv2.resize(roi_gray, (64, 64))
        roi_gray = roi_gray.astype("float32") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=-1)  # (64,64,1)
        roi_gray = np.expand_dims(roi_gray, axis=0)   # (1,64,64,1)

        # Predict emotion
        prediction = model.predict(roi_gray, verbose=0)[0]
        emotion = emotion_labels[np.argmax(prediction)]

        # Draw results
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Show result
    cv2.imshow("Facial Expression Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

