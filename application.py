import pickle
from flask import Flask, request, render_template
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Load Ridge model and scaler
ridge_model = pickle.load(open("ridge.pkl", "rb"))
standard_scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def index():
    return render_template("home.html")  # make sure this file exists in templates/

@app.route('/predictdata', methods=["GET", "POST"])
def predict_datapoint():

    if request.method == "POST":
        Temperature = float(request.form.get('Temperature'))
        Rh = float(request.form.get('RH'))
        ws = float(request.form.get('Ws'))
        rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        dmc = float(request.form.get('DMC'))
        isi = float(request.form.get('ISI'))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        input_data = [Temperature, Rh, ws, rain, ffmc, dmc, isi, Classes, Region]
        scaled_input = standard_scaler.transform([input_data])
        result = ridge_model.predict(scaled_input)
        return render_template("home.html", results=result[0])

    else:
        return render_template("home.html")


if __name__ == '__main__':
    app.run(host="0.0.0.0")
