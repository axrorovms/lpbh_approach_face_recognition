import cv2 as cv
import numpy as np
import os
import pickle

dataset_path = "./dataset"
model_path = "./face_recognizer_model.yml"
label_path = "./labels.pkl"

# Initialize LBPH recognizer with tuned parameters
recognizer = cv.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)

# Load Haarcascade dynamically
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

faces = []
labels = []
label_dict = {}
current_id = 0

# Iterate through dataset
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue

    label_dict[current_id] = person_name

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv.imread(img_path)

        if img is None:
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces_detected = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces_detected:
            face_area = gray[y:y+h, x:x+w]
            face_resized = cv.resize(face_area, (200, 200))
            faces.append(face_resized)
            labels.append(current_id)
            break

    current_id += 1

print(f"Total Faces: {len(faces)}, Labels: {len(labels)}")
print("Training model...")

recognizer.train(faces, np.array(labels))
recognizer.save(model_path)
print("Model saved!")

# Save label dictionary for later use
with open(label_path, "wb") as f:
    pickle.dump(label_dict, f)

print("Labels saved!")
