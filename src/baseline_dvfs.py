import pandas as pd
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "cpu_log.csv")

# Load data
df = pd.read_csv(DATA_PATH)

# DVFS parameters
LOW_FREQ = 1520   # MHz
HIGH_FREQ = 2400  # MHz
UTIL_THRESHOLD = 30.0

baseline_freq = []
energy = []

for util in df["cpu_util"]:
    if util > UTIL_THRESHOLD:
        freq = HIGH_FREQ
    else:
        freq = LOW_FREQ

    baseline_freq.append(freq)
    energy.append(freq)  # energy proxy per timestep

df["baseline_freq"] = baseline_freq
df["baseline_energy"] = energy

total_energy = sum(energy)

print("Baseline DVFS simulation complete")
print("Total baseline energy (proxy):", total_energy)
