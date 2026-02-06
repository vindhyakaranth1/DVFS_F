import pandas as pd
import os

# -------- Paths --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "cpu_log.csv")
FIXED_PATH = os.path.join(PROJECT_ROOT, "data", "cpu_log_fixed.csv")

# -------- Load WITHOUT header --------
df = pd.read_csv(RAW_PATH, header=None)

# -------- Assign correct column names --------
df.columns = [
    "timestamp",
    "cpu_util",
    "cpu_freq_mhz",
    "num_processes",
    "session"
]

# -------- Convert data types --------
df["cpu_util"] = pd.to_numeric(df["cpu_util"], errors="coerce")
df["cpu_freq_mhz"] = pd.to_numeric(df["cpu_freq_mhz"], errors="coerce")
df["num_processes"] = pd.to_numeric(df["num_processes"], errors="coerce")

# -------- Drop bad rows --------
df = df.dropna()

# -------- Save fixed file --------
df.to_csv(FIXED_PATH, index=False)

print("Header fixed successfully")
print("Total valid samples:", len(df))
print("Saved to:", FIXED_PATH)
