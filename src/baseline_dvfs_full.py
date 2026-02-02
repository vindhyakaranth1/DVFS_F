import pandas as pd
import numpy as np
import os

# -------- Paths --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "cpu_log_fixed.csv")

# -------- Load data --------
df = pd.read_csv(DATA_PATH)

WINDOW = 5
df_sim = df.iloc[WINDOW:].copy()

# -------- Baseline DVFS logic (reactive) --------
LOW_FREQ = 1520
HIGH_FREQ = 2400
UTIL_THRESHOLD = 30.0

baseline_freq = []

for util in df_sim["cpu_util"]:
    if util > UTIL_THRESHOLD:
        baseline_freq.append(HIGH_FREQ)
    else:
        baseline_freq.append(LOW_FREQ)

df_sim["baseline_freq"] = baseline_freq

# -------- Energy proxy --------
df_sim["baseline_energy"] = df_sim["baseline_freq"] ** 2

total_energy = df_sim["baseline_energy"].sum()

print("Baseline DVFS (full dataset) complete")
print("Total baseline energy (proxy):", total_energy)
