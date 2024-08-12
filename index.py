from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Initialize the Flask application
app = Flask(__name__)

# Set maximum upload size to 10MB (adjust as needed)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

# Load the pre-trained Keras model and Haarcascade XML file
model = load_model('model.h5')

# Define emotion labels (ensure they match your model's output)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_emotion(image):
    # Convert the image to grayscale
    gray_image = image.convert('L')
    
    # Resize the image to the size expected by the model (48x48 in this case)
    resized_image = gray_image.resize((48, 48))
    
    # Convert the image to an array
    image_array = img_to_array(resized_image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict the emotion
    preds = model.predict(image_array)[0]
    emotion_index = preds.argmax()
    emotion_label = emotion_labels[emotion_index]
    
    return emotion_label

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    # Open the image using Pillow
    try:
        image = Image.open(file.stream)
    except Exception as e:
        return jsonify({'error': 'Invalid image file'}), 400

    emotion = detect_emotion(image)
    
    if emotion:
        return jsonify({'emotion': emotion})
    else:
        return jsonify({'error': 'No emotion detected'}), 400

if __name__ == '__main__':
    app.run(debug=True)
