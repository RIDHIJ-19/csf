import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from PIL import Image
import io
import easyocr
import requests
import os

app = Flask(__name__)
model_cb = pickle.load(open('decision_tree_model.pkl', 'rb'))

with open("stopwords.txt", "r") as file:
    stopwords = file.read().splitlines()

vectorizer = pickle.load(open("tfidfvectorizer.pkl", "rb"))
reader = easyocr.Reader(['en', 'hi'])

@app.route('/')
def home():
    return render_template('cb_index.html', prediction=None, image_prediction=None)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        user_input = request.form['text']
        transformed_input = vectorizer.transform([user_input])
        prediction = model_cb.predict(transformed_input)[0]
    return render_template('cb_index.html', prediction=prediction, image_prediction=None)

@app.route('/predict_text_api', methods=['POST'])
def predict_api():
    data = request.get_json()
    user_input = data.get('text', '')

    if user_input:
        transformed_input = vectorizer.transform([user_input])
        prediction = model_cb.predict(transformed_input)[0]
    else:
        prediction = "No input provided"

    output = "1" if prediction == 1 else "0"
    return jsonify(output)

@app.route('/predict_image', methods=['GET', 'POST'])
def predict_image():
    if request.method == 'POST':
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        image = np.array(image)

        extracted_text = reader.readtext(image, detail=0)
        extracted_text = ' '.join(extracted_text)

        # Use relative URL assuming same server on Vercel
        response = requests.post(request.url_root + 'predict_text_api', json={'text': extracted_text})
        return render_template('cb_index.html', image_prediction=response.json(), prediction=None)

# Removed app.run() so Vercel handles server execution
