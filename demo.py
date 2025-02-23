import cv2 as cv
import numpy as np
import pickle

# Load trained model and labels
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("./face_recognizer_model.yml")

with open("./labels.pkl", "rb") as f:
    label_dict = pickle.load(f)

# Load Haarcascade dynamically
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam
cap = cv.VideoCapture(0)  # Change to 1 or 2 if using an external camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces_detected:
        face_area = gray[y:y+h, x:x+w]
        face_resized = cv.resize(face_area, (200, 200))

        label_id, confidence = recognizer.predict(face_resized)

        name = label_dict[label_id]


        # Draw bounding box and name
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(frame, name, (x+5, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv.imshow("Live Face Recognition", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv.destroyAllWindows()
