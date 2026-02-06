import pandas as pd
import numpy as np
import os

# ================= CONSTANTS =================
LOW_FREQ = 1520
HIGH_FREQ = 2400
UTIL_THRESHOLD = 30.0

ALPHA = 0.5
LOGICAL_CORES = 8
AVG_PROCS = 250

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "cpu_log_fixed.csv")

# ================= LOAD DATA =================
df = pd.read_csv(DATA_PATH)

WINDOW = 5
df_sim = df.iloc[WINDOW:].copy()

# ================= BASELINE DVFS =================
baseline_freq = []
for util in df_sim["cpu_util"]:
    if util > UTIL_THRESHOLD:
        baseline_freq.append(HIGH_FREQ)
    else:
        baseline_freq.append(LOW_FREQ)

df_sim["baseline_freq"] = baseline_freq

# ================= STACK A: BURST PENALTY =================
df_sim["freq_delta"] = df_sim["baseline_freq"].diff().abs().fillna(0)

# ================= STACK B: CORE-IDLE AWARE =================
active_ratio = np.minimum(
    1.0, df_sim["num_processes"] / LOGICAL_CORES
)

# ================= ENERGY MODEL =================
df_sim["baseline_energy"] = (
    df_sim["baseline_freq"] ** 2
    + ALPHA * df_sim["freq_delta"] * df_sim["baseline_freq"]
) * active_ratio

total_energy = df_sim["baseline_energy"].sum()

print("Baseline DVFS (Stack A+B) complete")
print("Total baseline energy (proxy):", total_energy)
