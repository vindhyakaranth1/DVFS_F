import numpy as np
import pandas as pd
import os
import joblib

# -------- Paths --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "cpu_log_fixed.csv")
X_PATH = os.path.join(PROJECT_ROOT, "data", "X_features.npy")
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "smartwatt_classifier.pkl")

# -------- Load data --------
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
X = np.load(X_PATH)

WINDOW = 5
df_sim = df.iloc[WINDOW:].copy()

# -------- Predict DVFS class --------
pred_class = model.predict(X)
df_sim["pred_class"] = pred_class

# -------- Frequency mapping --------
LOW_FREQ = 1520
HIGH_FREQ = 2400

df_sim["smart_freq"] = df_sim["pred_class"].apply(
    lambda x: HIGH_FREQ if x == 1 else LOW_FREQ
)

# -------- Energy proxy --------
df_sim["smart_energy"] = df_sim["smart_freq"] ** 2

total_energy = df_sim["smart_energy"].sum()

print("Smart-Watt (Classifier) DVFS complete")
print("Total Smart-Watt energy (proxy):", total_energy)
