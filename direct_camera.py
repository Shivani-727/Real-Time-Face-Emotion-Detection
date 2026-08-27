import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# -----------------------------
# Emotion labels
# -----------------------------
emotion_labels = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise'
]

# -----------------------------
# Load trained emotion model
# -----------------------------
classifier = load_model("model_78.h5")
classifier.load_weights("model_weights_78.h5")

# -----------------------------
# Load Haar Cascade
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("ERROR: Could not load Haar Cascade.")
    exit()

# -----------------------------
# Open webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Camera started!")
print("Press 'q' to quit.")

# -----------------------------
# Variables for performance
# -----------------------------
frame_count = 0
last_emotion = ""
last_confidence = 0.0

# -----------------------------
# Main loop
# -----------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    frame_count += 1

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 255),
            2
        )

        # Extract face
        roi_gray = gray[y:y + h, x:x + w]

        # Resize to model input size
        roi_gray = cv2.resize(
            roi_gray,
            (48, 48),
            interpolation=cv2.INTER_AREA
        )

        # Predict emotion every 3rd frame
        if frame_count % 3 == 0:

            roi = roi_gray.astype("float32") / 255.0

            roi = img_to_array(roi)

            roi = np.expand_dims(
                roi,
                axis=0
            )

            prediction = classifier.predict(
                roi,
                verbose=0
            )[0]

            max_index = int(
                np.argmax(prediction)
            )

            last_emotion = emotion_labels[max_index]

            last_confidence = (
                float(prediction[max_index]) * 100
            )

        # Display emotion
        if last_emotion:

            label = (
                f"{last_emotion}: "
                f"{last_confidence:.1f}%"
            )

            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    # Display camera
    cv2.imshow(
        "Real-Time Face Emotion Detection",
        frame
    )

    # Press Q to quit
    # Check keyboard input
    key = cv2.waitKey(1) & 0xFF

    # Press Q or ESC to quit
    if key == ord("q") or key == 27:
        break

    # Close program if the camera window X is clicked
    if cv2.getWindowProperty(
        "Real-Time Face Emotion Detection",
        cv2.WND_PROP_VISIBLE
    ) < 1:
        break  

# -----------------------------
# Release resources
# -----------------------------
cap.release()
cv2.destroyAllWindows()