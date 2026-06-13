
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load cleaned dataset
df = pd.read_csv("data/cleaned_churn.csv")

# Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Save feature names
joblib.dump(X.columns.tolist(), "feature_names.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )
}

results = {}

print("\nMODEL RESULTS")
print("=" * 60)

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Model": model
    }

    print(f"\n{name}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

# Select best model
best_model_name = max(
    results,
    key=lambda x: results[x]["F1 Score"]
)

best_model = results[best_model_name]["Model"]

print("\n" + "=" * 60)
print(f"Best Model: {best_model_name}")
print(f"Best F1 Score: {results[best_model_name]['F1 Score']:.4f}")

# Save best model
joblib.dump(best_model, "churn_model.pkl")

print("\nModel saved as churn_model.pkl")
print("Feature names saved as feature_names.pkl")

# Verify model
loaded_model = joblib.load("churn_model.pkl")

sample_pred = loaded_model.predict(X_test.iloc[:3])

print("\nVerification Successful")
print("Predictions:", sample_pred)
print("Actual:", y_test.iloc[:3].values)
