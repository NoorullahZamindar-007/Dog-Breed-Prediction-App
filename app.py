import os
from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
import numpy as np
import cv2

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the model
model = tf.keras.models.load_model("model/dog_breed.keras")

# Define class names (replace with your actual breed names)
CLASS_NAMES = ['Breed1', 'Breed2', 'Breed3', 'Breed4', 'Breed5']

def predict_breed(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0) / 255.0

    # Make prediction
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    breed_name = CLASS_NAMES[predicted_class]
    confidence = predictions[0][predicted_class] * 100

    return breed_name, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        
        file = request.files['image']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Ensure the upload folder exists before saving
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
                
            # Save the uploaded image
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Predict the breed
            breed_name, confidence = predict_breed(file_path)

            # Construct the correct path to the uploaded image
            image_url = url_for('static', filename=f'uploads/{file.filename}')

            return render_template('index.html', breed_name=breed_name, confidence=confidence, image_path=image_url)

    return render_template('index.html')

if __name__ == '__main__':
    # Ensure the upload folder exists before starting the server
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
