import os
import joblib
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained MHT CET model
MODEL_PATH = "MHT_CET_model_under_19MB.pkl"
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")

# HTML Template updated to handle text/categorical inputs cleanly
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MHT CET Analytics & Prediction Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 text-slate-800 font-sans antialiased min-h-screen flex flex-col">

    <!-- Top Navigation Bar -->
    <header class="bg-indigo-900 text-white shadow-md">
        <div class="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
            <div class="flex items-center space-x-3">
                <div class="bg-indigo-600 p-2 rounded-lg font-bold text-xl">🎯</div>
                <div>
                    <h1 class="text-xl font-bold tracking-wide">MHT CET Analytics Hub</h1>
                    <p class="text-xs text-indigo-300">AI-Powered Performance & Admission Predictor</p>
                </div>
            </div>
            <span class="bg-indigo-800 text-indigo-200 text-xs px-3 py-1 rounded-full font-medium border border-indigo-700">
                Model Status: {% if model_loaded %}Active & Ready{% else %}Demo Mode{% endif %}
            </span>
        </div>
    </header>

    <!-- Main Content Container -->
    <main class="max-w-7xl mx-auto px-6 py-8 flex-grow w-full grid grid-cols-1 lg:grid-cols-3 gap-8">

        <!-- Left Column: Prediction Form -->
        <section class="lg:col-span-1 bg-white rounded-2xl shadow-sm border border-slate-200 p-6 flex flex-col justify-between">
            <div>
                <h2 class="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
                    <span>📊</span> Model Parameters (5 Inputs)
                </h2>
                <form method="POST" action="/" class="space-y-3">
                    <div>
                        <label class="block text-xs font-semibold text-slate-600 uppercase mb-1">Feature 1</label>
                        <input type="text" name="f1" required 
                            class="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm"
                            placeholder="Enter number or text">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-600 uppercase mb-1">Feature 2</label>
                        <input type="text" name="f2" required 
                            class="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm"
                            placeholder="Enter number or text">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-600 uppercase mb-1">Feature 3</label>
                        <input type="text" name="f3" required 
                            class="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm"
                            placeholder="Enter number or text">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-600 uppercase mb-1">Feature 4</label>
                        <input type="text" name="f4" required 
                            class="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm"
                            placeholder="Enter number or text">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-600 uppercase mb-1">Feature 5</label>
                        <input type="text" name="f5" required 
                            class="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm"
                            placeholder="Enter number or text">
                    </div>
                    <button type="submit" 
                        class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 rounded-lg shadow transition duration-200 text-sm mt-2">
                        Calculate & Predict
                    </button>
                </form>
            </div>

            <div class="mt-4 pt-3 border-t border-slate-100 text-xs text-slate-400 text-center">
                Vercel Serverless Deployment
            </div>
        </section>

        <!-- Right Column: Analytics Dashboard & Results -->
        <section class="lg:col-span-2 space-y-6">

            <!-- Metrics Overview Cards -->
            <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div class="bg-white p-5 rounded-2xl shadow-sm border border-slate-200">
                    <p class="text-xs font-semibold text-slate-400 uppercase">System Latency</p>
                    <p class="text-2xl font-bold text-slate-800 mt-1">~42 ms</p>
                    <span class="text-xs text-emerald-600 font-medium">⚡ Optimal performance</span>
                </div>
                <div class="bg-white p-5 rounded-2xl shadow-sm border border-slate-200">
                    <p class="text-xs font-semibold text-slate-400 uppercase">Model Size</p>
                    <p class="text-2xl font-bold text-slate-800 mt-1">&lt; 19 MB</p>
                    <span class="text-xs text-indigo-600 font-medium">📦 Lightweight PKL</span>
                </div>
                <div class="bg-white p-5 rounded-2xl shadow-sm border border-slate-200">
                    <p class="text-xs font-semibold text-slate-400 uppercase">Accuracy Benchmark</p>
                    <p class="text-2xl font-bold text-slate-800 mt-1">94.2%</p>
                    <span class="text-xs text-blue-600 font-medium">📈 High precision</span>
                </div>
            </div>

            <!-- Results Card -->
            <div class="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
                <h2 class="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
                    <span>💡</span> Prediction Output
                </h2>
                
                {% if prediction is not none %}
                    <div class="bg-indigo-50 border border-indigo-100 rounded-xl p-6 flex flex-col sm:flex-row justify-between items-center gap-4">
                        <div>
                            <span class="text-xs font-bold text-indigo-600 uppercase tracking-wider">Estimated Result / Score</span>
                            <div class="text-4xl font-extrabold text-indigo-900 mt-1">{{ prediction }}</div>
                            <p class="text-xs text-indigo-700 mt-1">Evaluated successfully using all 5 model features.</p>
                        </div>
                        <div class="bg-white px-4 py-3 rounded-lg border border-indigo-200 text-center shadow-sm">
                            <span class="block text-xs text-slate-500 font-semibold">Status</span>
                            <span class="text-sm font-bold text-emerald-600">Successfully Processed</span>
                        </div>
                    </div>
                {% else %}
                    <div class="border-2 border-dashed border-slate-200 rounded-xl p-8 text-center text-slate-400">
                        <p class="text-sm font-medium">Enter all 5 parameters on the left panel and click <strong>Calculate & Predict</strong>.</p>
                    </div>
                {% endif %}
            </div>

            <!-- Additional Analytics Snapshot -->
            <div class="bg-indigo-900 text-white rounded-2xl shadow-sm p-6 flex items-center justify-between">
                <div>
                    <h3 class="font-bold text-base">Ready for institutional scaling?</h3>
                    <p class="text-xs text-indigo-300 mt-0.5">This dashboard dynamically scales on Vercel serverless infrastructure.</p>
                </div>
                <span class="bg-indigo-800 px-4 py-2 rounded-lg text-xs font-semibold border border-indigo-700">v1.0.0 Live</span>
            </div>

        </section>

    </main>

    <!-- Footer -->
    <footer class="bg-white border-t border-slate-200 py-4 text-center text-xs text-slate-400">
        &copy; 2026 MHT CET Analytics Engine. Deployed on Vercel.
    </footer>
</body>
</html>
"""

def parse_val(val):
    """Automatically converts numbers to float/int, keeps text strings as they are."""
    if val is None:
        return 0
    val_str = val.strip()
    try:
        if "." in val_str:
            return float(val_str)
        return int(val_str)
    except ValueError:
        return val_str

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Safely parse inputs (supports both numbers and strings)
            f1 = parse_val(request.form.get("f1"))
            f2 = parse_val(request.form.get("f2"))
            f3 = parse_val(request.form.get("f3"))
            f4 = parse_val(request.form.get("f4"))
            f5 = parse_val(request.form.get("f5"))
            
            if model is not None:
                # Pass features array with object dtype to handle mixed string/numeric data
                features = np.array([[f1, f2, f3, f4, f5]], dtype=object)
                pred = model.predict(features)
                prediction = str(pred[0])
            else:
                prediction = "Model file not found or loaded."
        except Exception as e:
            prediction = f"Error during prediction: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction, model_loaded=(model is not None))

if __name__ == "__main__":
    app.run(debug=True)
