
from flask import Flask, render_template, request
import pickle
import pandas as pd

# Load trained pipeline
filename = "diabetes-prediction-rfc-model.pkl"

with open(filename, "rb") as file:
    classifier = pickle.load(file)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if request.method == "POST":

        data = {
            "Pregnancies": int(request.form["pregnancies"]),
            "Glucose": int(request.form["glucose"]),
            "BloodPressure": int(request.form["bloodpressure"]),
            "SkinThickness": int(request.form["skinthickness"]),
            "Insulin": int(request.form["insulin"]),
            "BMI": float(request.form["bmi"]),
            "DPF": float(request.form["dpf"]),
            "Age": int(request.form["age"])
        }

        input_data = pd.DataFrame([data])

        prediction = classifier.predict(input_data)

        return render_template(
            "result.html",
            prediction=prediction
        )


if __name__ == "__main__":
    app.run(debug=True)
