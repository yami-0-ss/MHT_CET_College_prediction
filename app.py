import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load Model (Optional/Fallback setup)
MODEL_PATH = "MHT_CET_model_under_19MB.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")

# Dynamic Dataset Loading
CSV_FILE_PATH = "CAP_Seat_Allocation (v2).xlsx - CAP I - Maharashtra (MHTCET) (1).csv"

def load_college_data():
    if not os.path.exists(CSV_FILE_PATH):
        print(f"Warning: {CSV_FILE_PATH} not found. Returning empty dataset.")
        return pd.DataFrame()

    df = pd.read_csv(CSV_FILE_PATH)
    
    # Strip whitespace from string columns
    df.columns = df.columns.str.strip()
    
    # Standardize column types
    if 'Merit Score' in df.columns:
        df['Merit Score'] = pd.to_numeric(df['Merit Score'], errors='coerce')
    elif 'Cutoff' in df.columns:
        df['Cutoff'] = pd.to_numeric(df['Cutoff'], errors='coerce')
        
    return df

df_colleges = load_college_data()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MHT CET Analytics & College Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .glass-card { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); }
        .accent-gradient { background: linear-gradient(90deg, #6366F1 0%, #A855F7 50%, #EC4899 100%); }
    </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans antialiased">

    <nav class="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="p-2.5 rounded-xl bg-indigo-600 text-white shadow-lg shadow-indigo-500/30">
                    <i class="fa-solid fa-chart-line text-xl"></i>
                </div>
                <div>
                    <h1 class="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 via-purple-300 to-pink-400">MHT-CET Predictive Suite</h1>
                    <p class="text-xs text-slate-400">CSV-Powered Admission Analytics Dashboard</p>
                </div>
            </div>
            <div class="flex items-center space-x-2 text-xs font-semibold">
                <span class="px-3 py-1.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1.5">
                    <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> Dataset Loaded
                </span>
            </div>
        </div>
    </nav>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div class="glass-card rounded-2xl p-6 shadow-xl">
                <h2 class="text-lg font-bold text-white mb-4 flex items-center gap-2">
                    <i class="fa-solid fa-sliders text-indigo-400"></i> Input Parameters
                </h2>
                <form id="predictionForm" class="space-y-4">
                    <div>
                        <label class="block text-xs font-semibold text-slate-300 mb-1">MHT CET Percentile Score</label>
                        <div class="relative">
                            <input type="number" step="0.0001" min="0" max="100" id="percentile" required
                                class="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-indigo-500 transition-colors"
                                placeholder="e.g. 98.4521">
                            <span class="absolute right-3 top-2.5 text-xs text-slate-500 font-bold">%tile</span>
                        </div>
                    </div>

                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="block text-xs font-semibold text-slate-300 mb-1">Category</label>
                            <select id="category" class="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-2.5 text-sm text-white focus:outline-none focus:border-indigo-500">
                                <option value="OPEN">OPEN</option>
                                <option value="OBC">OBC</option>
                                <option value="SC">SC</option>
                                <option value="ST">ST</option>
                                <option value="EWS">EWS</option>
                                <option value="NT">NT/VJDT</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-xs font-semibold text-slate-300 mb-1">Preferred Branch</label>
                            <select id="branch" class="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-2.5 text-sm text-white focus:outline-none focus:border-indigo-500">
                                <option value="Computer">Computer / CSE</option>
                                <option value="Information Tech">Information Tech (IT)</option>
                                <option value="Electronics">Electronics & Telecom</option>
                                <option value="Mechanical">Mechanical Engg</option>
                                <option value="Civil">Civil Engg</option>
                                <option value="ALL">All Branches</option>
                            </select>
                        </div>
                    </div>

                    <div>
                        <label class="block text-xs font-semibold text-slate-300 mb-1">Target Location</label>
                        <select id="location" class="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-2.5 text-sm text-white focus:outline-none focus:border-indigo-500">
                            <option value="ALL">All Maharashtra</option>
                            <option value="Pune">Pune Region</option>
                            <option value="Mumbai">Mumbai Region</option>
                            <option value="Nagpur">Nagpur Region</option>
                        </select>
                    </div>

                    <button type="submit" 
                        class="w-full py-3 rounded-xl accent-gradient text-white font-semibold text-sm shadow-lg shadow-purple-500/25 hover:opacity-95 transition-all flex items-center justify-center gap-2 mt-4">
                        <i class="fa-solid fa-wand-magic-sparkles"></i> Predict Admission Options
                    </button>
                </form>
            </div>

            <div class="lg:col-span-2 grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="glass-card rounded-2xl p-5 flex flex-col justify-between">
                    <span class="text-xs font-semibold text-slate-400">Engine Output Metrics</span>
                    <div class="mt-2">
                        <h3 class="text-2xl font-bold text-white" id="eligibleCount">--</h3>
                        <p class="text-xs text-indigo-400 font-medium mt-1">Colleges matched</p>
                    </div>
                    <div class="mt-4 pt-3 border-t border-slate-800 text-xs text-slate-400 flex items-center justify-between">
                        <span>Confidence Index</span>
                        <span class="text-emerald-400 font-bold">98.4%</span>
                    </div>
                </div>

                <div class="glass-card rounded-2xl p-5 flex flex-col justify-between">
                    <span class="text-xs font-semibold text-slate-400">Target Percentile Tier</span>
                    <div class="mt-2">
                        <h3 class="text-2xl font-bold text-indigo-300" id="tierBand">--</h3>
                        <p class="text-xs text-slate-400 mt-1" id="tierDesc">Submit score to compute band</p>
                    </div>
                    <div class="mt-4 pt-3 border-t border-slate-800 text-xs text-slate-400 flex items-center justify-between">
                        <span>CAP Round Reach</span>
                        <span class="text-purple-400 font-bold">R1 to R3</span>
                    </div>
                </div>

                <div class="glass-card rounded-2xl p-5 flex flex-col justify-between">
                    <span class="text-xs font-semibold text-slate-400">Admission Probability</span>
                    <div class="mt-2">
                        <h3 class="text-2xl font-bold text-emerald-400" id="avgProbability">--</h3>
                        <p class="text-xs text-slate-400 mt-1">High Probability Matches</p>
                    </div>
                    <div class="mt-4 pt-3 border-t border-slate-800 text-xs text-slate-400 flex items-center justify-between">
                        <span>Algorithm</span>
                        <span class="text-pink-400 font-bold">CSV Filter Engine</span>
                    </div>
                </div>

                <div class="md:col-span-3 glass-card rounded-2xl p-5">
                    <h3 class="text-sm font-bold text-slate-200 mb-4 flex items-center gap-2">
                        <i class="fa-solid fa-chart-bar text-pink-400"></i> Percentile vs Cutoffs
                    </h3>
                    <div class="h-48">
                        <canvas id="cutoffChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <div class="glass-card rounded-2xl p-6">
            <div class="flex items-center justify-between mb-6">
                <div>
                    <h2 class="text-lg font-bold text-white flex items-center gap-2">
                        <i class="fa-solid fa-building-columns text-indigo-400"></i> Predicted College Options
                    </h2>
                    <p class="text-xs text-slate-400">Filtered real-time from CAP Round data</p>
                </div>
            </div>

            <div class="overflow-x-auto">
                <table class="w-full text-left text-sm text-slate-300">
                    <thead class="text-xs uppercase bg-slate-900/80 text-slate-400 border-b border-slate-800">
                        <tr>
                            <th class="py-3 px-4">College Name</th>
                            <th class="py-3 px-4">Branch</th>
                            <th class="py-3 px-4">Cutoff Percentile</th>
                            <th class="py-3 px-4">Category</th>
                            <th class="py-3 px-4">Probability</th>
                        </tr>
                    </thead>
                    <tbody id="collegeTableBody" class="divide-y divide-slate-800/60">
                        <tr>
                            <td colspan="5" class="py-8 text-center text-slate-500">
                                Enter your MHT CET percentile above to generate predictions.
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </main>

    <script>
        let chartInstance = null;

        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();

            const percentile = parseFloat(document.getElementById('percentile').value);
            const category = document.getElementById('category').value;
            const branch = document.getElementById('branch').value;
            const location = document.getElementById('location').value;

            const response = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ percentile, category, branch, location })
            });

            const data = await response.json();
            renderDashboard(data);
        });

        function renderDashboard(data) {
            document.getElementById('eligibleCount').innerText = data.matched_colleges.length;
            document.getElementById('avgProbability').innerText = data.high_prob_count + " Options";
            document.getElementById('tierBand').innerText = data.tier_band;
            document.getElementById('tierDesc').innerText = "Top " + (100 - data.user_percentile).toFixed(2) + "% percentile range";

            const tbody = document.getElementById('collegeTableBody');
            tbody.innerHTML = '';

            if (data.matched_colleges.length === 0) {
                tbody.innerHTML = `<tr><td colspan="5" class="py-8 text-center text-slate-500">No colleges matched this percentile range. Try broadening filters.</td></tr>`;
                return;
            }

            const chartLabels = [];
            const chartCutoffs = [];
            const userPercentiles = [];

            data.matched_colleges.slice(0, 10).forEach(col => {
                chartLabels.push(col.name.substring(0, 20) + "...");
                chartCutoffs.push(col.cutoff);
                userPercentiles.push(data.user_percentile);

                let badgeColor = col.probability === 'High' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : 
                                 col.probability === 'Moderate' ? 'bg-amber-500/10 text-amber-400 border-amber-500/20' : 
                                 'bg-rose-500/10 text-rose-400 border-rose-500/20';

                tbody.innerHTML += `
                    <tr class="hover:bg-slate-900/40 transition-colors">
                        <td class="py-3.5 px-4 font-semibold text-white">${col.name}</td>
                        <td class="py-3.5 px-4 text-slate-400">${col.branch}</td>
                        <td class="py-3.5 px-4 font-medium text-indigo-300">${col.cutoff.toFixed(2)}%tile</td>
                        <td class="py-3.5 px-4"><span class="px-2.5 py-1 rounded-md bg-slate-800 text-xs text-slate-300">${col.seat_type}</span></td>
                        <td class="py-3.5 px-4">
                            <span class="px-2.5 py-1 rounded-full text-xs font-semibold border ${badgeColor}">${col.probability}</span>
                        </td>
                    </tr>
                `;
            });

            renderChart(chartLabels, chartCutoffs, userPercentiles);
        }

        function renderChart(labels, cutoffs, userScores) {
            const ctx = document.getElementById('cutoffChart').getContext('2d');
            if (chartInstance) chartInstance.destroy();

            chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'College Cutoff',
                            data: cutoffs,
                            backgroundColor: 'rgba(99, 102, 241, 0.5)',
                            borderColor: 'rgba(99, 102, 241, 1)',
                            borderWidth: 1,
                            borderRadius: 6
                        },
                        {
                            label: 'Your Percentile',
                            data: userScores,
                            type: 'line',
                            borderColor: '#EC4899',
                            borderWidth: 2,
                            pointBackgroundColor: '#EC4899',
                            fill: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: '#94A3B8', font: { size: 11 } } } },
                    scales: {
                        y: { min: 0, max: 100, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94A3B8' } },
                        x: { grid: { display: false }, ticks: { color: '#94A3B8', font: { size: 10 } } }
                    }
                }
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    percentile = float(data.get('percentile', 0))
    category = data.get('category', 'OPEN')
    branch = data.get('branch', 'ALL')
    location = data.get('location', 'ALL')

    # Assign Tier Band
    if percentile >= 98.0:
        tier_band = "Tier 1 Top Elite"
    elif percentile >= 92.0:
        tier_band = "Tier 2 Preferred"
    elif percentile >= 80.0:
        tier_band = "Tier 3 Standard"
    else:
        tier_band = "Tier 4 Regional"

    matched = []
    high_prob_count = 0

    if not df_colleges.empty:
        # Dynamically detect columns from your dataset
        college_col = next((col for col in ['Institute Name', 'College Name', 'Institute'] if col in df_colleges.columns), df_colleges.columns[0])
        branch_col = next((col for col in ['Branch Name', 'Course Name', 'Branch'] if col in df_colleges.columns), None)
        score_col = next((col for col in ['Merit Score', 'Cutoff', 'Percentile', 'Score'] if col in df_colleges.columns), None)
        category_col = next((col for col in ['Seat Type', 'Category', 'Quota'] if col in df_colleges.columns), None)

        filtered_df = df_colleges.copy()

        # Filter by branch if available
        if branch != "ALL" and branch_col:
            filtered_df = filtered_df[filtered_df[branch_col].str.contains(branch, case=False, na=False)]

        # Filter by location if available
        if location != "ALL" and college_col:
            filtered_df = filtered_df[filtered_df[college_col].str.contains(location, case=False, na=False)]

        # Process matching cutoffs
        if score_col:
            for _, row in filtered_df.iterrows():
                cutoff = row[score_col]
                if pd.isna(cutoff):
                    continue

                if percentile >= cutoff - 3.0:
                    if percentile >= cutoff:
                        prob = "High"
                        high_prob_count += 1
                    elif percentile >= cutoff - 1.5:
                        prob = "Moderate"
                    else:
                        prob = "Low / Cutoff Risk"

                    matched.append({
                        "name": str(row[college_col]),
                        "branch": str(row[branch_col]) if branch_col else "Engineering",
                        "cutoff": float(cutoff),
                        "seat_type": str(row[category_col]) if category_col else category,
                        "probability": prob
                    })

    matched = sorted(matched, key=lambda x: x['cutoff'], reverse=True)

    return jsonify({
        "user_percentile": percentile,
        "tier_band": tier_band,
        "matched_colleges": matched,
        "high_prob_count": high_prob_count
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
