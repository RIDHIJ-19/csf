import os
import pickle
import numpy as np
import requests
from flask import Flask, render_template, request, jsonify, url_for
from PIL import Image
import easyocr


app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models and resources
model_cb = pickle.load(open('decision_tree_model.pkl', 'rb'))
vectorizer = pickle.load(open("tfidfvectorizer.pkl", "rb"))
reader = easyocr.Reader(['en', 'hi'])
with open("stopwords.txt", "r") as file:
    stopwords = file.read().splitlines()

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/cb')
def cb_home():
    return render_template('cb_index.html', prediction=None, image_prediction=None, image_path=None, extracted_text=None)

@app.route('/cb/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        user_input = request.form['text']
        transformed_input = vectorizer.transform([user_input])
        prediction = model_cb.predict(transformed_input)[0]

    return render_template('cb_index.html', prediction=prediction, image_prediction=None, image_path=None, extracted_text=user_input)

@app.route('/cb/predict_text_api', methods=['POST'])
def predict_api():
    data = request.get_json()
    user_input = data.get('text', '')

    if user_input:
        transformed_input = vectorizer.transform([user_input])
        prediction = model_cb.predict(transformed_input)[0]
    else:
        prediction = "No input provided"

    return jsonify("1" if prediction == 1 else "0")

@app.route('/cb/predict_image', methods=['GET', 'POST'])
def predict_image():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Save the image to static folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
            file.save(filepath)

            # Extract text from the image
            image = Image.open(filepath)
            image = np.array(image)
            extracted_text = reader.readtext(image, detail=0)
            extracted_text = ' '.join(extracted_text)

            # Predict using extracted text
            response = requests.post('http://127.0.0.1:5000/cb/predict_text_api', json={'text': extracted_text})

            return render_template(
                'cb_index.html',
                image_prediction=response.json(),
                prediction=None,
                image_path=url_for('static', filename='uploaded_image.jpg'),
                extracted_text=extracted_text
            )
    return render_template('cb_index.html', prediction=None, image_prediction=None, image_path=None, extracted_text=None)


 
if __name__ == '__main__':
    app.run(debug=True)
