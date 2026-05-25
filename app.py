import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('housing_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():

    data = request.json['data']
    print(data)

    input_data = np.array(list(data.values())).reshape(1, -1)
    print(input_data)

    new_data = scaler.transform(input_data)

    new_data = new_data[:, [0,1,2,5,6,7]]

    output = model.predict(new_data)


    print(output[0])

    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)