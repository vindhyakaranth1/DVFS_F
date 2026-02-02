import pandas as pd
import os

# -------- Paths --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "cpu_log.csv")
FIXED_PATH = os.path.join(PROJECT_ROOT, "data", "fixed_cpu_log.csv")

print("Loading dataset with flexible parser...")

# Read without header enforcement
df = pd.read_csv(
    RAW_PATH,
    header=None,
    engine="python"
)

print("Raw shape:", df.shape)

# Decide column structure
if df.shape[1] == 4:
    df.columns = [
        "timestamp",
        "cpu_util",
        "cpu_freq_mhz",
        "num_processes"
    ]
    df["session"] = "unknown"
elif df.shape[1] == 5:
    df.columns = [
        "timestamp",
        "cpu_util",
        "cpu_freq_mhz",
        "num_processes",
        "session"
    ]
else:
    raise ValueError("Unexpected number of columns")

# Convert data types safely
df["cpu_util"] = pd.to_numeric(df["cpu_util"], errors="coerce")
df["cpu_freq_mhz"] = pd.to_numeric(df["cpu_freq_mhz"], errors="coerce")
df["num_processes"] = pd.to_numeric(df["num_processes"], errors="coerce")

# Fill missing session names
df["session"] = df["session"].fillna("unknown")

# Drop bad rows
before = len(df)
df = df.dropna()
after = len(df)

# Save fixed dataset
df.to_csv(FIXED_PATH, index=False)

print("Fix complete")
print("Original rows:", before)
print("Valid rows:", after)
print("Saved to:", FIXED_PATH)
