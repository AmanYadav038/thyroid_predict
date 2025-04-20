from flask import Flask, render_template
import joblib
import json
import os
from routes.predict import predict_bp

app = Flask(__name__)
app.register_blueprint(predict_bp)

# Load selected features for frontend
selected_features = joblib.load("models/selected_features.pkl")

@app.route("/")
def home():
    with open('symptoms_config.json') as f:
        symptoms_data = json.load(f)
    sorted_symptoms = sorted(symptoms_data['symptoms'], key=lambda x: x['priority'])
    return render_template("index.html", features=sorted_symptoms)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
