import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model (must be in the same folder as app.py, named SVM.pkl)
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SVM.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = ["Age", "Gender", "Region", "Occupation", "Income"]

# ---------------------------------------------------------------------------
# IMPORTANT: the model was trained on NUMERIC codes. Edit these labels so the
# numbers match the encoding you used while training (e.g. LabelEncoder order).
# Format -> "label shown in dropdown": numeric code sent to the model
# ---------------------------------------------------------------------------
GENDER_MAP = {"Female": 0, "Male": 1}
REGION_MAP = {"Region 0": 0, "Region 1": 1, "Region 2": 2, "Region 3": 3}
OCCUPATION_MAP = {
    "Occupation 0": 0, "Occupation 1": 1, "Occupation 2": 2,
    "Occupation 3": 3, "Occupation 4": 4,
}


@app.route("/")
def home():
    return render_template(
        "index.html",
        genders=GENDER_MAP,
        regions=REGION_MAP,
        occupations=OCCUPATION_MAP,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        row = pd.DataFrame(
            [[
                float(data["age"]),
                float(data["gender"]),
                float(data["region"]),
                float(data["occupation"]),
                float(data["income"]),
            ]],
            columns=FEATURES,
        )
        label = str(model.predict(row)[0])
        score = float(model.decision_function(row)[0])  # distance from boundary
        # squash to a 0-100 "confidence" for display (SVC has probability=False)
        confidence = round(100 / (1 + pow(2.718281828, -abs(score) * 2)), 1)
        return jsonify(ok=True, prediction=label, confidence=confidence)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
