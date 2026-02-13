from flask import Flask,render_template,jsonify,request
import pickle
import numpy as np


app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f) 

@app.route("/")
def home():
      return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if request is JSON or form data
        if request.is_json:
            data = request.get_json()
            features = [
                float(data["age"]),
                float(data["sex"]),
                float(data["cp"]),
                float(data["trestbps"]),
                float(data["chol"]),
                float(data["fbs"]),
                float(data["restecg"]),
                float(data["thalach"]),
                float(data["exang"]),
                float(data["oldpeak"]),
                float(data["slope"]),
                float(data["ca"]),
                float(data["thal"])
            ]
        else:
            # Form data
            features = [
                float(request.form["age"]),
                float(request.form["sex"]),
                float(request.form["cp"]),
                float(request.form["trestbps"]),
                float(request.form["chol"]),
                float(request.form["fbs"]),
                float(request.form["restecg"]),
                float(request.form["thalach"]),
                float(request.form["exang"]),
                float(request.form["oldpeak"]),
                float(request.form["slope"]),
                float(request.form["ca"]),
                float(request.form["thal"])
            ]

        final_features = np.array([features])
        prediction = model.predict(final_features)

        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"

        # Return JSON if request was JSON
        if request.is_json:
            return {"prediction": result, "prediction_value": int(prediction[0])}
        else:
            return render_template("index.html", prediction_text=result)

    except Exception as e:
        if request.is_json:
            return {"error": str(e)}, 400
        else:
            return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)


