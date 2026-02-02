import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

# -------- Paths --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "cpu_log_fixed.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "cpu_log_prepared.csv")

# -------- Load data --------
df = pd.read_csv(INPUT_PATH)

# -------- Parse & sort time --------
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# -------- Select numeric columns --------
numeric_cols = ["cpu_util", "cpu_freq_mhz", "num_processes"]

# -------- Normalize numeric features --------
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# -------- Save prepared dataset --------
df.to_csv(OUTPUT_PATH, index=False)

print("Dataset preparation complete")
print("Saved to:", OUTPUT_PATH)
print("Total samples:", len(df))
