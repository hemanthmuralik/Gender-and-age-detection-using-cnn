import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load trained models
emotion_model = load_model("emotion_model.h5")
age_model = load_model("age_prediction_model.h5")

# Define emotions and age categories
emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
age_categories = ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "60+"]

# Load Haarcascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize emotion probabilities for bar graph
emotion_probs = np.zeros(len(emotions))

# Set up Matplotlib graph
plt.ion()
fig, ax = plt.subplots()
bars = ax.bar(emotions, emotion_probs, color="skyblue")
ax.set_ylim(0, 1)
ax.set_ylabel("Probability")
ax.set_title("Emotion Levels")

# Function to update the probability bar graph
def update_graph(emotion_probs):
    for bar, prob in zip(bars, emotion_probs):
        bar.set_height(prob)
    plt.draw()
    plt.pause(0.01)

# Real-time Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3,
                                          minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_gray, (48, 48))
        face_resized = face_resized / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)
        face_resized = np.expand_dims(face_resized, axis=-1)

        # Predict emotion
        emotion_pred = emotion_model.predict(face_resized)[0]
        emotion_probs = emotion_pred
        predicted_emotion = emotions[np.argmax(emotion_pred)]

        # Predict age
        age_pred = age_model.predict(face_resized)[0]
        predicted_age = age_categories[np.argmax(age_pred)]

        # Draw rectangle & labels
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{predicted_emotion}, {predicted_age}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

        # Update bar graph
        u
