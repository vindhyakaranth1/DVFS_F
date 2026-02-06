import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------- Paths --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

X_PATH = os.path.join(PROJECT_ROOT, "data", "X_features.npy")
Y_PATH = os.path.join(PROJECT_ROOT, "data", "y_class.npy")
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "smartwatt_classifier.pkl")

# -------- Load data --------
X = np.load(X_PATH)
y = np.load(Y_PATH)

# Align feature matrix with horizon labels
X = X[:len(y)]


# -------- Time-aware split (NO SHUFFLE) --------
split_idx = int(0.7 * len(X))

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# -------- Model --------
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=14,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# -------- Train --------
model.fit(X_train, y_train)

# -------- Evaluate --------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X)[:, 1]


acc = accuracy_score(y_test, y_pred)

print("\n===== SMART-WATT CLASSIFIER RESULTS =====")
print("Accuracy:", round(acc * 100, 2), "%\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

# -------- Save model --------
PROB_PATH = os.path.join(PROJECT_ROOT, "data", "y_prob.npy")
np.save(PROB_PATH, y_prob)

print("Prediction probabilities saved to:", PROB_PATH)


joblib.dump(model, MODEL_PATH)
print("\nModel saved to:", MODEL_PATH)
