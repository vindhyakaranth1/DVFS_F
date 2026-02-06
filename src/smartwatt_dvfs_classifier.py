import numpy as np
import pandas as pd
import os
import joblib

# ================= CONSTANTS =================
LOW_FREQ = 1520
MID_FREQ = 2000
HIGH_FREQ = 2400

HOLD_HIGH = 5
HOLD_LOW = 3

ALPHA = 0.5
LOGICAL_CORES = 8

current_freq = None
hold_counter = 0

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "cpu_log_fixed.csv")
X_PATH = os.path.join(PROJECT_ROOT, "data", "X_features.npy")
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "smartwatt_classifier.pkl")
PROB_PATH = os.path.join(PROJECT_ROOT, "data", "y_prob.npy")

# ================= LOAD DATA =================
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
X = np.load(X_PATH)
y_prob = np.load(PROB_PATH)

WINDOW = 5
df_sim = df.iloc[WINDOW:].copy()

min_len = min(len(df_sim), len(y_prob))
df_sim = df_sim.iloc[:min_len].copy()
y_prob = y_prob[:min_len]

# ================= SMART GOVERNOR =================
cpu_window = []
WINDOW_CPU = 5
smart_freqs = []

for idx in range(len(df_sim)):
    cpu_util = df_sim.iloc[idx]["cpu_util"]
    prob = y_prob[idx]

    cpu_window.append(cpu_util)
    if len(cpu_window) > WINDOW_CPU:
        cpu_window.pop(0)

    recent_cpu_mean = sum(cpu_window) / len(cpu_window)

    if prob > 0.85 and recent_cpu_mean > 0.7:
        target_freq = HIGH_FREQ
    elif prob > 0.55:
        target_freq = MID_FREQ
    else:
        target_freq = LOW_FREQ

    if current_freq is None:
        current_freq = target_freq
        hold_counter = HOLD_HIGH if target_freq == HIGH_FREQ else HOLD_LOW

    elif hold_counter > 0:
        hold_counter -= 1

    else:
        if target_freq != current_freq:
            current_freq = target_freq
            hold_counter = HOLD_HIGH if target_freq == HIGH_FREQ else HOLD_LOW

    smart_freqs.append(current_freq)

df_sim["smart_freq"] = smart_freqs

# ================= STACK A =================
df_sim["freq_delta"] = df_sim["smart_freq"].diff().abs().fillna(0)

# ================= STACK B =================
active_ratio = np.minimum(
    1.0, df_sim["num_processes"] / LOGICAL_CORES
)

# ================= ENERGY MODEL =================
df_sim["smart_energy"] = (
    df_sim["smart_freq"] ** 2
    + ALPHA * df_sim["freq_delta"] * df_sim["smart_freq"]
) * active_ratio

total_energy = df_sim["smart_energy"].sum()

print("Smart-Watt (Stack A+B) DVFS complete")
print("Total Smart-Watt energy (proxy):", total_energy)
