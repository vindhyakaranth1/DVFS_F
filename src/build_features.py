import pandas as pd
import numpy as np
import os

# -------- Paths --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "cpu_log_fixed.csv")
OUTPUT_X = os.path.join(PROJECT_ROOT, "data", "X_features.npy")
OUTPUT_Y = os.path.join(PROJECT_ROOT, "data", "y_class.npy")

# -------- Load data --------
df = pd.read_csv(INPUT_PATH)

cpu_vals = df["cpu_util"].values

WINDOW = 5

X = []
y = []

for i in range(WINDOW, len(cpu_vals)):
    window = cpu_vals[i - WINDOW:i]

    features = []

    # Raw values
    features.extend(window)

    # Deltas
    features.extend(np.diff(window))

    # Statistics
    features.append(np.mean(window))
    features.append(np.std(window))

    X.append(features)

    # Binary class label (used by classifier)
    y.append(1 if cpu_vals[i] > 30 else 0)

X = np.array(X)
y = np.array(y)

# -------- Save --------
np.save(OUTPUT_X, X)
np.save(OUTPUT_Y, y)

print("Feature engineering complete")
print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)
print("Example feature:", X[0])
print("Example label:", y[0])
