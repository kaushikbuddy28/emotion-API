from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("model.h5")

classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'image' not in data:
        return jsonify({'error': 'No image provided'})

    try:
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes)).convert('L').resize((48, 48))
        img_array = np.expand_dims(np.array(img), axis=(0, -1)) / 255.0
        prediction = model.predict(img_array)[0]
        emotion_index = np.argmax(prediction)

        return jsonify({
            'emotion': classes[emotion_index],
            'confidence': float(prediction[emotion_index])
        })
    except Exception as e:
        return jsonify({'error': str(e)})
