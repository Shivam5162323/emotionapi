import requests
import urllib.parse
encoded_path = urllib.parse.quote(image_path)


url = 'http://127.0.0.1:5000/detect_emotion'
image_path = 'C:\\Users\\shiva\\Machine Leaning\\m3.png'  # Replace with the actual path

response = requests.get(url, params={'imagePath': encoded_path})

if response.status_code == 200:
    result = response.json()
    detected_emotion = result['emotion']
    print(f"Detected emotion: {detected_emotion}")
else:
    print(f"Error: {response.json()}")