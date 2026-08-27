# Real-Time Face Emotion Detection

A real-time facial emotion detection application built using Python, OpenCV, TensorFlow, Keras, and a CNN-based emotion classification model.

The application captures live video from the webcam, detects faces using Haar Cascade, preprocesses each detected face, and predicts the person's facial emotion in real time along with a confidence score.

## 🚀 Features

- Real-time face detection using webcam
- Facial emotion classification using a CNN model
- Detects 7 different emotions
- Displays emotion confidence score
- Face bounding box visualization
- Real-time processing using OpenCV
- Optimized emotion prediction by processing selected frames
- Direct webcam application without requiring a browser or localhost server

## 😊 Supported Emotions

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## 🛠️ Technologies Used

- Python 3.11
- OpenCV
- TensorFlow
- Keras
- NumPy
- Haar Cascade Classifier
- Convolutional Neural Network (CNN)

## 🔄 How It Works

The application follows this pipeline:

Webcam  
↓  
Video Frame Capture  
↓  
Grayscale Conversion  
↓  
Face Detection using Haar Cascade  
↓  
Face Region Extraction  
↓  
Resize to 48 × 48 pixels  
↓  
Pixel Normalization  
↓  
CNN-based Emotion Classification  
↓  
Emotion + Confidence Score

## 🧠 Model

The application uses a trained CNN-based facial emotion classification model.

Each detected face is converted to grayscale and resized to 48 × 48 pixels before being passed to the model.

The model predicts one of seven emotion classes:

`Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise`

The emotion with the highest predicted probability is displayed along with its confidence score.

## ⚡ Performance Optimization

To improve real-time performance on CPU-based systems, emotion prediction is not performed on every webcam frame.

Instead, CNN prediction is performed on selected frames while the latest prediction is maintained between inference frames.

This reduces unnecessary model inference and helps make the webcam feed smoother.

## 📁 Project Structure

```text
Real-Time-Face-Emotion-Detection/
│
├── direct_camera.py
├── model_78.h5
├── model_weights_78.h5
├── haarcascade_frontalface_default.xml
├── requirements.txt
├── .gitignore
└── README.md