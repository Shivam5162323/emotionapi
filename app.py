from flask import Flask, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained Keras model and Haarcascade XML file
model = load_model('model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define emotion labels (ensure they match your model's output)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_emotion(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    emotions = []
    
    for (x, y, w, h) in faces:
        # Extract the region of interest (face) and resize it to match the model's input size
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = img_to_array(roi_gray)
        roi_gray = np.expand_dims(roi_gray, axis=0)

        # Predict the emotion
        preds = model.predict(roi_gray)[0]
        emotion_index = preds.argmax()
        emotion_label = emotion_labels[emotion_index]
        emotions.append(emotion_label)

    return emotions

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    emotions = detect_emotion(image)
    
    if emotions:
        return jsonify({'emotions': emotions})
    else:
        return jsonify({'error': 'No faces detected'}), 400

if __name__ == '__main__':
    app.run(debug=True)
