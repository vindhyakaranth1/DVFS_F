import numpy as np
import pandas as pd
import os
import joblib

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "cpu_log.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "smartwatt_model.pkl")

# ---------------- Load data ----------------
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

# ---------------- Feature engineering ----------------
WINDOW = 5
cpu_vals = df["cpu_util"].values

# Smooth CPU signal (3-point moving average)
cpu_vals = pd.Series(cpu_vals).rolling(
    window=3, min_periods=1
).mean().values

features = []
for i in range(WINDOW, len(cpu_vals)):
    window_vals = cpu_vals[i-WINDOW:i]
    feat = list(window_vals)
    feat.extend(np.diff(window_vals))
    feat.append(np.mean(window_vals))
    feat.append(np.std(window_vals))
    features.append(feat)

X = np.array(features)

# ---------------- Predict next CPU ----------------
predicted_cpu = model.predict(X)

# Align dataframe
df_sim = df.iloc[WINDOW:].copy()
df_sim["predicted_cpu"] = predicted_cpu

# ---------------- Smart-Watt DVFS ----------------
LOW_FREQ = 1520
HIGH_FREQ = 2400
UTIL_THRESHOLD = 30.0

smart_freq = []
energy = []

for util in df_sim["predicted_cpu"]:
    if util > UTIL_THRESHOLD:
        freq = HIGH_FREQ
    else:
        freq = LOW_FREQ

    smart_freq.append(freq)
    energy.append(freq)

df_sim["smart_freq"] = smart_freq
df_sim["smart_energy"] = energy

total_energy = sum(energy)

print("Smart-Watt DVFS simulation complete")
print("Total Smart-Watt energy (proxy):", total_energy)
