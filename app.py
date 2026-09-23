import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the trained model safely
MODEL_PATH = "MHT_CET_model_under_19MB.pkl"
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully!")
    else:
        print(f"Warning: {MODEL_PATH} not found in current directory.")
except Exception as e:
    print(f"Error loading model: {e}")

# Attractive Professional Dashboard Template with Embedded Analytics & Styling
TEMPLATE = """
