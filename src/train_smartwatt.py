import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "cpu_log.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "smartwatt_model.pkl")

# ---------------- Load data ----------------
df = pd.read_csv(DATA_PATH)

# ---------------- Feature engineering ----------------
WINDOW = 5
cpu_vals = df["cpu_util"].values

features = []
targets = []

for i in range(WINDOW, len(cpu_vals)):
    window_vals = cpu_vals[i-WINDOW:i]

    feat = list(window_vals)
    feat.extend(np.diff(window_vals))
    feat.append(np.mean(window_vals))
    feat.append(np.std(window_vals))

    features.append(feat)

if cpu_vals[i] > 30:
    targets.append(1)  # HIGH
else:
    targets.append(0)  # LOW


X = np.array(features)
y = np.array(targets)

# ---------------- Train-test split (time-aware) ----------------
split_idx = int(0.7 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# ---------------- Train model ----------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- Evaluate ----------------
acc = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {acc*100:.2f}%")

print("Smart-Watt ML Model Trained")
print(f"MAE: {mae:.2f} %")
print(f"R2 Score: {r2:.3f}")

# ---------------- Save model ----------------
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
