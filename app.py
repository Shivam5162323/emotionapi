from flask import Flask, request, jsonify
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import io

# Define the Flask app
app = Flask(__name__)

def load_model():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.load_weights('model.h5')
    return model

model = load_model()

def detect_emotion(image_bytes):
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # Dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    image_stream = io.BytesIO(image_bytes)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    detected_emotion = None

    for (x, y, w, h) in faces:
        # Rest of your face detection code...
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))

        # Get the detected emotion
        detected_emotion = emotion_dict[maxindex]

    return detected_emotion

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion_api():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file found'}), 400

        # Get the image file from the request
        image_file = request.files['image'].read()
        detected_emotion = detect_emotion(image_file)
        
        if detected_emotion is None:
            return jsonify({'error': 'No faces detected in the image'}), 400
        
        return jsonify({"emotion": detected_emotion}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
