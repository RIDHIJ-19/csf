import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
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
        output = "1" if prediction == 1 else "0"
    else:
        output = "No input provided"

    return jsonify(output)

@app.route('/predict_image', methods=['GET', 'POST'])
def predict_image():
    if request.method == 'POST':
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        image = np.array(image)

        # Use dynamic URL for API call
        api_url = url_for('predict_api', _external=True)
        response = requests.post(api_url, json={'text': ' '.join(reader.readtext(image, detail=0))})
        return render_template('cb_index.html', image_prediction=response.json(), prediction=None)

# Expose app for gunicorn
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
