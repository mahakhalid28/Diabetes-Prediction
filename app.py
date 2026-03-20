from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)


# Use the full path so the server always knows where to look
model = pickle.load(
    open(
        "/home/MahaKhalid/Diabetes-Prediction/Trained_Models/diabetes_trained_model.pkl",
        "rb",
    )
)


@app.route("/")
def home():
    # This looks for the index.html file inside the 'templates' folder
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]

    # Make prediction
    prediction = model.predict(final_features)

    result = "DIABETIC" if prediction[0] == 1 else "NON-DIABETIC"

    return f"""
        <div style="text-align:center; margin-top:50px; font-family:Arial;">
            <h1>Prediction Result</h1>
            <p style="font-size:24px;">The patient is predicted to be: <strong>{result}</strong></p>
            <a href="/">Go Back</a>
        </div>
    """


if __name__ == "__main__":
    app.run()
