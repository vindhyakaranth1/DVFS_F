import pandas as pd
import numpy as np
import os

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "cpu_log_fixed.csv")
Y_LABELS_PATH = os.path.join(PROJECT_ROOT, "data", "y_labels.npy")
Y_CLASS_PATH = os.path.join(PROJECT_ROOT, "data", "y_class.npy")

# ================= PARAMETERS =================
HORIZON = 5          # 5 samples ≈ 1 second
THRESHOLD = 0.30     # normalized CPU (30%)

# ================= LOAD DATA =================
df = pd.read_csv(INPUT_PATH)

# ================= STEP 1: BUILD CONTINUOUS LABELS =================
# Normalize CPU utilization (0–100 → 0–1)
y_continuous = df["cpu_util"].values / 100.0

np.save(Y_LABELS_PATH, y_continuous)

print("Step 1: Continuous labels (y_labels.npy) created")
print("Total samples:", len(y_continuous))
print("Min:", y_continuous.min(), "Max:", y_continuous.max())

# ================= STEP 2: BUILD HORIZON-BASED CLASS LABELS =================
y_horizon = []

for i in range(len(y_continuous) - HORIZON):
    future_avg = np.mean(y_continuous[i:i + HORIZON])
    y_horizon.append(1 if future_avg > THRESHOLD else 0)

y_horizon = np.array(y_horizon)

np.save(Y_CLASS_PATH, y_horizon)

print("\nStep 2: Horizon-based class labels (y_class.npy) created")
print("Total samples:", len(y_horizon))
print("HIGH freq samples:", y_horizon.sum())
print("LOW freq samples:", len(y_horizon) - y_horizon.sum())
