
import pandas as pd

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Shape:", df.shape)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Drop customerID
df.drop("customerID", axis=1, inplace=True)

# Find categorical columns
categorical_cols = df.select_dtypes(include="object").columns.tolist()

binary_cols = []
multi_class_cols = []

for col in categorical_cols:
    if df[col].nunique() == 2:
        binary_cols.append(col)
    else:
        multi_class_cols.append(col)

# Label Encode Binary Columns
for col in binary_cols:
    unique_vals = sorted(df[col].unique())
    mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
    df[col] = df[col].map(mapping)

# One-Hot Encode Multi-Class Columns
df = pd.get_dummies(
    df,
    columns=multi_class_cols,
    drop_first=True,
    dtype=int
)

# Final Checks
print("\nFinal Shape:", df.shape)
print("Missing Values Remaining:", df.isnull().sum().sum())

# Save cleaned dataset
df.to_csv("data/cleaned_churn.csv", index=False)

print("\nCleaned dataset saved successfully.")
