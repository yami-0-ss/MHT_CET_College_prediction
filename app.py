import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder="../templates")

# Resolve model path dynamically for Vercel's serverless environment
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "MHT_CET_model_under_19MB.pkl")

model = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        percentile = float(data.get("percentile", 0.0))
        category = data.get("category", "General")
        branch = data.get("branch", "Computer Engineering")

        if model is not None:
            # Format inputs to match your pipeline's training schema
            # Example DataFrame schema:
            input_df = pd.DataFrame([{
                "percentile": percentile,
                "category": category,
                "branch": branch
            }])
            
            try:
                prediction_raw = model.predict(input_df)[0]
            except Exception:
                # Fallback to numerical numpy array if model expects raw float array
                prediction_raw = model.predict(np.array([[percentile]]))[0]
            
            pred_college = str(prediction_raw)
        else:
            pred_college = "Model not initialized"

        # Calculate analytics metrics for dashboard visualization
        competition_index = min(100.0, max(5.0, round((100.0 - percentile) * 1.8, 1)))
        admission_chance = min(98.0, max(2.0, round((percentile / 100.0) ** 1.3 * 100, 1)))

        return jsonify({
            "status": "success",
            "prediction": pred_college,
            "metrics": {
                "percentile": percentile,
                "admission_chance": admission_chance,
                "competition_index": competition_index
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# Entry point for local testing
if __name__ == "__main__":
    app.run(debug=True, port=3000)
