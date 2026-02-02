import pandas as pd
import numpy as np
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "cpu_log.csv")

# Load data
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Parameters
WINDOW = 5  # number of past samples

features = []
targets = []

cpu_vals = df["cpu_util"].values

for i in range(WINDOW, len(cpu_vals)):
    window_vals = cpu_vals[i-WINDOW:i]

    # Raw values
    feat = list(window_vals)

    # Deltas
    deltas = np.diff(window_vals)
    feat.extend(deltas)

    # Statistics
    feat.append(np.mean(window_vals))
    feat.append(np.std(window_vals))

    features.append(feat)
    targets.append(cpu_vals[i])

X = np.array(features)
y = np.array(targets)

print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)
print("Sample feature vector:", X[0])
print("Sample target:", y[0])
