import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

# Define emotion labels (assuming we have a 7-class emotion model)
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise', 'Neutral']

# Load pre-trained emotion detection model (replace with your model path)
model = load_model('emotion_model.h5')  # Change the path to your model

# Initialize the face detector (Haar Cascade Classifier or any other detector)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess the image for emotion classification
def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    face = cv2.resize(face, (48, 48))  # Resize to 48x48 (common size for emotion models)
    face = face.astype('float32') / 255.0  # Normalize
    face = np.expand_dims(face, axis=-1)  # Add channel dimension (grayscale channel)
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    return face

# Start video capture
cap = cv2.VideoCapture(0)  # 0 is the default camera

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame to grayscale (for faster face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop over detected faces
    for (x, y, w, h) in faces:
        # Extract the face from the frame
        face = frame[y:y + h, x:x + w]

        # Preprocess the face for emotion prediction
        face_preprocessed = preprocess_face(face)

        # Predict the emotion
        emotion_probabilities = model.predict(face_preprocessed)
        emotion_index = np.argmax(emotion_probabilities[0])
        emotion = emotion_labels[emotion_index]

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with emotion labels
    cv2.imshow("Emotion Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
