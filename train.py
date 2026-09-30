import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

ART = "artifacts"
os.makedirs(ART, exist_ok=True)
os.makedirs("data", exist_ok=True)

# 1. Load the UCI AI4I 2020 dataset
dataset = fetch_ucirepo(id=601)
X = dataset.data.features.copy()
y = dataset.data.targets.copy()

# UCI may expose the target as a one-column DataFrame.
target_col = y.columns[0]
y = y[target_col].astype(int)

# Keep useful predictive-maintenance variables.
rename = {
    "Air temperature [K]": "air_temperature",
    "Process temperature [K]": "process_temperature",
    "Rotational speed [rpm]": "rotational_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Type": "product_type",
}
X = X.rename(columns=rename)

# Drop identifiers that should not drive failure prediction.
drop_cols = [c for c in ["UDI", "Product ID"] if c in X.columns]
X = X.drop(columns=drop_cols)

# Save a local copy for reproducibility after download.
local = X.copy()
local["machine_failure"] = y.values
local.to_csv("data/ai4i_prepared.csv", index=False)

cat_cols = [c for c in ["product_type"] if c in X.columns]
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)

pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
prob = pipe.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred, zero_division=0)),
    "f1": float(f1_score(y_test, pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, prob)),
}

print("\nEvaluation metrics")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

print("\nClassification report")
print(classification_report(y_test, pred, zero_division=0))

cm = confusion_matrix(y_test, pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix — Industrial Machine Failure Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{ART}/confusion_matrix.png", dpi=180)
plt.close()

with open(f"{ART}/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

joblib.dump(pipe, f"{ART}/machine_failure_model.joblib")

print(f"\nSaved model to {ART}/machine_failure_model.joblib")
print(f"Saved metrics to {ART}/metrics.json")
