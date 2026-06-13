
import pandas as pd
import joblib

# Load model and feature names
model = joblib.load("churn_model.pkl")
features = joblib.load("feature_names.pkl")

# Create empty customer record
customer = pd.DataFrame(0, index=[0], columns=features)

# User input
customer["tenure"] = int(input("Tenure (months): "))
customer["MonthlyCharges"] = float(input("Monthly Charges: "))
customer["TotalCharges"] = float(input("Total Charges: "))

# Predict
result = model.predict(customer)[0]

print(
    "\nCustomer likely to churn"
    if result == 1
    else "\nCustomer likely to stay"
)
