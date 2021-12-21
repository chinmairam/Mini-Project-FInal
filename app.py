from flask import Flask, render_template, url_for, request
import os
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

model1 = pickle.load(open('chronic.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final = [np.array(int_features)]
    predictions = model1.predict(final)
    output = predictions[0]
    #return render_template('predict.html', prediction_text="Prediction is {}".format(output))

    #if predictions == 0:
    #    return render_template('predict.html', prediction_text=f'Person is not having CKD')
    #else:
    #     return render_template('predict.html', prediction_text=f'Person is having CKD')    

    if int(output) == 0:
        return render_template('predict.html', picture="https://image.shutterstock.com/image-vector/fun-people-healthy-life-logo-600w-1018248916.jpg", prediction_text=f'Person is not having CKD')
    else:
        return render_template('predict.html', picture="https://image.shutterstock.com/image-vector/chronic-kidney-disease-bad-health-600w-1131056297.jpg", prediction_text=f'Person is having CKD')

if __name__ == '__main__':
    app.run(debug=True)
