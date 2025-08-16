


# 🧬 Life Expectancy Prediction Web App

A complete machine learning pipeline and web application to predict **life expectancy** based on health, lifestyle, and socio-economic factors.  

The project generates synthetic data, trains a machine learning model, and serves predictions through a Flask-based web interface.  



## 🚀 Features
- 📊 **Synthetic Data Generation** – Creates realistic health & lifestyle datasets  
- 🧠 **ML Model Training** – Random Forest Regressor for predicting life expectancy  
- 🌐 **Flask Web App** – User-friendly interface to input health parameters and get predictions  
- ⚡ **Real-Time Predictions** – Fast inference with a trained model (`life_expectancy_model.pkl`)  
- 📂 **Modular Code** – Separate scripts for data generation, training, and serving  


## 🛠 Tech Stack
- **Backend:** Python, Flask  
- **Machine Learning:** scikit-learn, pandas, numpy, joblib  
- **Frontend:** HTML (via Flask templates)  
- **Model:** Random Forest Regressor  

---

## 📂 Project Structure
```

├── app.py                      # Flask web app for predictions
├── train\_model.py              # Script to train the model
├── generate\_synthetic\_data.py  # Script to generate synthetic datasets
├── life\_expectancy\_model.pkl   # Trained ML model (created after training)
├── synthetic\_merged.csv        # Combined synthetic dataset (generated)
├── templates/
│   └── index.html              # Web UI form + results
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation

````

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YourUsername/Life-Expectancy-Predictor.git
cd Life-Expectancy-Predictor
````

### 2️⃣ Create Virtual Environment & Install Requirements

```bash
python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 3️⃣ Generate Synthetic Data

```bash
python generate_synthetic_data.py
```

This will create `synthetic_group1.csv`, `synthetic_group2.csv`, `synthetic_group3.csv`, and `synthetic_merged.csv`.

### 4️⃣ Train the Model

```bash
python train_model.py
```

This will train a Random Forest model and save it as `life_expectancy_model.pkl`.

### 5️⃣ Run the Flask App

```bash
python app.py
```

Open your browser at **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**.

---

## 📋 Usage

1. Enter your details (age, height, weight, lifestyle, medical history, etc.) in the form.
2. Click **Predict**.
3. The app displays your **predicted life expectancy**.

---

## 🌟 Future Improvements

* 📈 Visualization of feature impact on life expectancy
* 🌍 Deployment to Heroku / Render / AWS
* 📱 Mobile-friendly responsive design
* 🤖 Explore deep learning models for better accuracy

---

## 📜 License

This project is licensed under the **MIT License**.

---


Do you want me to also create a **ready `requirements.txt`** for you so contributors can install everything in one go?
```
