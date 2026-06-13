
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load cleaned dataset
df = pd.read_csv("data/cleaned_churn.csv")

# Create output folder
os.makedirs("eda_charts", exist_ok=True)

# Churn distribution
print("Churn Distribution:")
print(df["Churn"].value_counts())

plt.figure(figsize=(6,4))
df["Churn"].value_counts().plot(kind="bar")
plt.title("Churn Distribution")
plt.savefig("eda_charts/churn_distribution.png")
plt.close()

# Contract Type Analysis
df["Contract_Type"] = "Month-to-Month"
df.loc[df["Contract_One year"] == 1, "Contract_Type"] = "One Year"
df.loc[df["Contract_Two year"] == 1, "Contract_Type"] = "Two Year"

contract_churn = pd.crosstab(df["Contract_Type"], df["Churn"])

print("\nContract Type vs Churn:")
print(contract_churn)

contract_churn.plot(kind="bar", figsize=(8,5))
plt.title("Churn by Contract Type")
plt.savefig("eda_charts/churn_by_contract.png")
plt.close()

# Internet Service Analysis
df["Internet_Type"] = "DSL"
df.loc[df["InternetService_Fiber optic"] == 1, "Internet_Type"] = "Fiber Optic"
df.loc[df["InternetService_No"] == 1, "Internet_Type"] = "No Internet"

internet_churn = pd.crosstab(df["Internet_Type"], df["Churn"])

print("\nInternet Service vs Churn:")
print(internet_churn)

internet_churn.plot(kind="bar", figsize=(8,5))
plt.title("Churn by Internet Service")
plt.savefig("eda_charts/churn_by_internet.png")
plt.close()

# Payment Method Analysis
df["Payment_Type"] = "Bank Transfer"
df.loc[df["PaymentMethod_Electronic check"] == 1, "Payment_Type"] = "Electronic Check"
df.loc[df["PaymentMethod_Credit card (automatic)"] == 1, "Payment_Type"] = "Credit Card"
df.loc[df["PaymentMethod_Mailed check"] == 1, "Payment_Type"] = "Mailed Check"

payment_churn = pd.crosstab(df["Payment_Type"], df["Churn"])

print("\nPayment Method vs Churn:")
print(payment_churn)

payment_churn.plot(kind="bar", figsize=(8,5))
plt.title("Churn by Payment Method")
plt.savefig("eda_charts/churn_by_payment.png")
plt.close()

# Monthly Charges Distribution
plt.figure(figsize=(8,5))

plt.hist(
    df[df["Churn"] == 0]["MonthlyCharges"],
    bins=30,
    alpha=0.7,
    label="No Churn"
)

plt.hist(
    df[df["Churn"] == 1]["MonthlyCharges"],
    bins=30,
    alpha=0.7,
    label="Churn"
)

plt.legend()
plt.title("Monthly Charges Distribution")
plt.savefig("eda_charts/monthly_charges.png")
plt.close()

# Tenure Distribution
plt.figure(figsize=(8,5))

plt.hist(
    df[df["Churn"] == 0]["tenure"],
    bins=30,
    alpha=0.7,
    label="No Churn"
)

plt.hist(
    df[df["Churn"] == 1]["tenure"],
    bins=30,
    alpha=0.7,
    label="Churn"
)

plt.legend()
plt.title("Tenure Distribution")
plt.savefig("eda_charts/tenure_distribution.png")
plt.close()

print("\nEDA Completed Successfully")
print("Charts saved in eda_charts folder")
