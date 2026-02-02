import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

Y_INPUT = os.path.join(PROJECT_ROOT, "data", "y_labels.npy")
Y_OUTPUT = os.path.join(PROJECT_ROOT, "data", "y_class.npy")

y = np.load(Y_INPUT)

HORIZON = 5          # 5 samples ≈ 1 second
THRESHOLD = 0.30     # normalized CPU

y_horizon = []

for i in range(len(y) - HORIZON):
    future_avg = np.mean(y[i:i + HORIZON])
    y_horizon.append(1 if future_avg > THRESHOLD else 0)

y_horizon = np.array(y_horizon)

np.save(Y_OUTPUT, y_horizon)

print("Horizon-based labels created")
print("Total samples:", len(y_horizon))
print("HIGH freq samples:", y_horizon.sum())
print("LOW freq samples:", len(y_horizon) - y_horizon.sum())
