# MHT_CET_College_prediction

An AI/ML-based MHT-CET College Prediction System that predicts suitable colleges and courses based on a student's MHT-CET marks, percentile, rank, category, and other admission-related details.

live project link
https://mht-cet-college-prediction-9.onrender.com/

📌 Project Overview

The MHT-CET College Prediction System is a machine learning web application designed to help students explore possible college and course options based on their MHT-CET performance.

The system takes student details as input and provides a predicted College + Course recommendation.

This project demonstrates the complete machine learning workflow:

Data Collection → Data Preprocessing → Feature Engineering → Model Training → Prediction → Web Deployment

🚀 Features
🎓 MHT-CET College Prediction
📊 Prediction based on student performance
🏫 College and course recommendation
📈 Uses machine learning for prediction
🖥️ User-friendly web interface
⚡ Fast prediction
🌐 Deployed online using Render
📱 Responsive and easy-to-use interface
🧠 Input Features

The model can use admission-related features such as:

MHT-CET Marks
MHT-CET Percentile
Rank
Category
Gender
Other relevant student/admission information
🎯 Prediction Output

The system predicts:

College Name + Course Name

Example:

Predicted College:
ABC Institute of Technology

Predicted Course:
Computer Engineering
🛠️ Technologies Used
Technology	Purpose
Python	Programming
Pandas	Data preprocessing
NumPy	Numerical operations
Scikit-learn	Machine Learning
Flask	Web application/API
HTML/CSS	Frontend
Pickle	Model serialization
Render	Deployment
GitHub	Version control
🤖 Machine Learning Workflow
1. Data Collection

Collected MHT-CET-related college admission data containing student performance and college/course information.

2. Data Preprocessing
Handled missing values
Cleaned data
Converted categorical variables
Prepared features for machine learning
3. Feature Engineering

Relevant admission features such as marks, percentile, rank, category, and other parameters were prepared for model training.

4. Model Training

A machine learning classification approach was used to learn relationships between student information and college/course outcomes.

5. Model Serialization

The trained model was saved using Python's pickle module.

import pickle

with open("MHT_CET_model.pkl", "wb") as file:
    pickle.dump(model, file)
6. Deployment

The application was deployed on Render and connected with the trained machine learning model.

📂 Project Structure
MHT-CET-College-Prediction/
│
├── app.py
├── MHT_CET_model.pkl
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── dataset/
    └── mht_cet_data.csv
▶️ How to Run Locally
Step 1: Clone the Repository
git clone https://github.com/your-username/MHT-CET-College-Prediction.git
Step 2: Open the Project
cd MHT-CET-College-Prediction
Step 3: Create Virtual Environment
python -m venv venv

Activate it on Windows:

venv\Scripts\activate
Step 4: Install Dependencies
pip install -r requirements.txt
Step 5: Run the Application
python app.py
Step 6: Open in Browser
http://127.0.0.1:5000/
📦 Requirements

Example requirements.txt:

Flask
pandas
numpy
scikit-learn
gunicorn
🌐 Live Application

Try the deployed application:

MHT-CET College Prediction System

https://mht-cet-college-prediction-9.onrender.com/

📊 Machine Learning Application

This project can help students understand potential college and course options based on their entrance-exam performance.

Important: The prediction is an ML-based estimate and should not be treated as an official admission result. Actual allotment depends on official counselling rules, seat availability, cutoffs, category, and other admission factors.

🔮 Future Enhancements
Add official historical MHT-CET cutoff data
Add CAP round-wise prediction
Add college comparison
Add course comparison
Add college location/map integration
Add historical cutoff visualization
Add personalized college recommendations
Add probability/confidence information
Improve model accuracy with larger datasets
Add user authentication and prediction history
👩‍💻 Author

Yamini More

Aspiring Data Analyst / Data Scientist

Skills Demonstrated

Python • Machine Learning • Pandas • NumPy • Scikit-learn • Flask • Power BI • Data Analysis • Data Visualization
