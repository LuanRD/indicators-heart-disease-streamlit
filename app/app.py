import pickle
from flask import Flask, request, render_template
import pandas as pd
import numpy as np

app = Flask(__name__, static_url_path='/static')
pipeline = pickle.load(open('../models/pipeline.pkl', 'rb'))

columns = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth',
           'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic',
           'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease',
           'SkinCancer']


@app.route('/')
def display():
    return render_template('index.html')


@app.route('/predict/', methods=['POST'])
def predict():
    data = request.form.to_dict()
    inputs = list()
    for x in data.values():
        if x.isnumeric() == True:
            x = float(x)
        else:
            pass

        inputs.append(x)

    values = list()
    for i, j in enumerate(inputs):
        i = list()
        i.append(j)
        values.append(i)

    zip_obj = zip(columns, values)
    df = pd.DataFrame(dict(zip_obj))

    prediction = pipeline.predict(df)
    probability = pipeline.predict_proba(df)

    if prediction[0] != 0:
        result = f'Result: You probably have heart disease. Probability = {(probability[0][1] * 100):.2f}%'
        return render_template("index.html", result=result)
    else:
        result = f'Result: You probably do not have heart disease. Probability = {(probability[0][1] * 100):.2f}%'
        return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
