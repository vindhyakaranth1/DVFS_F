import pandas as pd
import os

# -------- Paths --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "cpu_log.csv")
FIXED_PATH = os.path.join(PROJECT_ROOT, "data", "cpu_log_fixed.csv")

# -------- Load raw data (no header assumed) --------
df = pd.read_csv(RAW_PATH, header=None)

# -------- Assign column names --------
df.columns = [
    "timestamp",
    "cpu_util",
    "cpu_freq_mhz",
    "num_processes",
    "session"
]

# -------- Convert to numeric safely --------
df["cpu_util"] = pd.to_numeric(df["cpu_util"], errors="coerce")
df["cpu_freq_mhz"] = pd.to_numeric(df["cpu_freq_mhz"], errors="coerce")
df["num_processes"] = pd.to_numeric(df["num_processes"], errors="coerce")

# -------- Clean --------
before = len(df)
df = df.dropna()
after = len(df)

# -------- Save --------
df.to_csv(FIXED_PATH, index=False)

print("Dataset fixed successfully")
print("Original rows:", before)
print("Valid rows:", after)
print("Saved to:", FIXED_PATH)
