🎭 Emotions and Age Prediction Using CNN

A Real-Time Facial Analysis System using Deep Learning

📌 Overview

This project implements a real-time facial emotion and age prediction system using Convolutional Neural Networks (CNNs).
Using a webcam feed, the system detects human faces, classifies the emotion, and predicts age groups efficiently without requiring high-end hardware.

It is built using TensorFlow/Keras, OpenCV, NumPy, and trained on the FER-2013 and UTKFace datasets.

🚀 Features

Real-time face detection

Emotion classification (Angry, Happy, Sad, Surprise, Fear, Neutral, Disgust)

Age group prediction

Highly optimized CNN architecture for fast inference

Graphical visualization of emotion probabilities

Robust preprocessing pipeline (cropping, resizing, normalization)

🧠 Model Architecture

The system uses two separately trained CNN models:

✔ Emotion Recognition Model

Input: 48×48 grayscale images

Dataset: FER-2013

Output: 7 emotion classes

✔ Age Prediction Model

Input: 48×48 grayscale images

Dataset: UTKFace

Output: Custom age groups (e.g., 0–10, 11–20 …)

Core CNN Layers

Convolution Layers

ReLU Activation

MaxPooling

Flatten

Dense Layers

Softmax Output

📂 Project Structure
📁 Emotion-Age-Prediction
│── train/  
│── test/  
│── emotion_model.h5  
│── age_prediction_model.h5  
│── realtime_detection.py  
│── README.md  
│── requirements.txt  

🛠️ Technologies Used
🧪 Software

Python

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

VS Code

💻 Hardware

Standard PC/Laptop

Webcam

(Optional) GPU for faster training

📊 Datasets Used

FER-2013 — Facial Emotion Recognition dataset

UTKFace — Age-labeled facial dataset

📥 Installation
1️⃣ Clone the Repository
git clone https://github.com/yourusername/emotion-age-prediction.git
cd emotion-age-prediction

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Add Trained Models

Place the following files in the project directory:

emotion_model.h5
age_prediction_model.h5

▶️ Running the Real-Time System
python realtime_detection.py


This will launch the webcam window with predicted emotion, age, and probability bar graph.

🧾 Training Scripts

Training scripts for both models are included in:

train_emotion_model.py

train_age_model.py

Both training pipelines include:

Image preprocessing

Data augmentation

CNN model building

Loss/accuracy visualization

Model saving

📈 Results

Achieves reliable real-time performance

Emotion prediction accuracy improves with augmentation

Fast inference even on CPU systems

Handles lighting and facial variations fairly well

🎯 Applications

Human–Computer Interaction

Security and Surveillance

Customer Behavior Analysis

Healthcare and Therapy

Education and E-learning analytics

🔮 Future Enhancements

Apply transfer learning with deeper CNNs (ResNet / MobileNet)

Add gender prediction

Deploy on mobile / edge devices

Use attention mechanisms for enhanced feature extraction

Integrate multi-modal data such as rPPG or audio

🧑‍💻 Author

Hemanth Murali K
B.Tech in Electronics & Communication Engineering
