import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Load the synthetic data CSV (make sure this file is in the project folder)
data = pd.read_csv("synthetic_merged.csv")

# Use all columns as features except the targets ('Predicted_Life_Expectancy' and 'Final_Life_Expectancy')
X = data.drop(columns=['Predicted_Life_Expectancy', 'Final_Life_Expectancy'])
y = data['Final_Life_Expectancy']

# Split data into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model using Mean Squared Error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Save the trained model to a pickle file
joblib.dump(model, "life_expectancy_model.pkl")
print("Model saved as life_expectancy_model.pkl")
