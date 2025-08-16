from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("life_expectancy_model.pkl")

# Home route that renders the input form
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route to handle form submission
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Retrieve form inputs and convert to proper types
        age = float(request.form.get("Age"))
        height_cm = float(request.form.get("Height_cm"))
        weight_kg = float(request.form.get("Weight_kg"))
        bmi = weight_kg / ((height_cm / 100.0) ** 2)
        smoking = int(request.form.get("Smoking"))
        alcohol = int(request.form.get("Alcohol"))
        bp_sys = float(request.form.get("BP_Systolic"))
        bp_dia = float(request.form.get("BP_Diastolic"))
        pulse = float(request.form.get("Pulse"))
        exercise = int(request.form.get("Exercise"))
        water_intake = float(request.form.get("Water_Intake_L"))
        sleep = float(request.form.get("Sleep_Hours"))
        education = int(request.form.get("Education"))
        income = float(request.form.get("Income_INR"))
        hypertension = int(request.form.get("Hypertension"))
        diabetes = int(request.form.get("Diabetes"))
        thyroid = int(request.form.get("Thyroid"))
        mental_health = int(request.form.get("Mental_Health"))
        fatty_liver = int(request.form.get("Fatty_Liver"))

        # Create a DataFrame with the input (columns must match those used during training)
        input_data = pd.DataFrame([{
            'Age': age,
            'Height_cm': height_cm,
            'Weight_kg': weight_kg,
            'BMI': bmi,
            'Smoking': smoking,
            'Alcohol': alcohol,
            'BP_Systolic': bp_sys,
            'BP_Diastolic': bp_dia,
            'Pulse': pulse,
            'Exercise': exercise,
            'Water_Intake_L': water_intake,
            'Sleep_Hours': sleep,
            'Education': education,
            'Income_INR': income,
            'Hypertension': hypertension,
            'Diabetes': diabetes,
            'Thyroid': thyroid,
            'Mental_Health': mental_health,
            'Fatty_Liver': fatty_liver
        }])
        
        # Predict life expectancy using the trained model
        prediction = model.predict(input_data)[0]
        prediction = np.round(prediction, 1)

        return render_template("index.html", prediction_text=f"Predicted Life Expectancy: {prediction} years")
    except Exception as e:
        return render_template("index.html", prediction_text="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
