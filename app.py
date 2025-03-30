import numpy as np
from flask import Flask, request, jsonify, render_template
 



app = Flask(__name__)
 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/cb')
def cb_home():
    return render_template('cb_index.html')

 

# ------------------------------------ DEEPFAKE MODEL ------------------------------------
 
 

@app.route('/df')
def dfhome():
    return render_template('df_index.html')
 
if __name__ == '__main__':
    app.run(debug=True)
