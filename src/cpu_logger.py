import psutil
import pandas as pd
import time
from datetime import datetime
import os

# ===================== CONFIG =====================
SAMPLE_INTERVAL = 0.2       # 200 ms
DURATION = 600              # 10 minutes
SESSION_NAME = "session_6"  # CHANGE THIS EVERY RUN
# ================================================

print("Starting CPU data collection...")
print("Session:", SESSION_NAME)

data = []

start_time = time.time()
iterations = int(DURATION / SAMPLE_INTERVAL)

for _ in range(iterations):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    cpu_util = psutil.cpu_percent(interval=None)

    freq_info = psutil.cpu_freq()
    cpu_freq = freq_info.current if freq_info else 0

    num_processes = len(psutil.pids())

    data.append([
        timestamp,
        cpu_util,
        cpu_freq,
        num_processes,
        SESSION_NAME
    ])

    time.sleep(SAMPLE_INTERVAL)

df = pd.DataFrame(
    data,
    columns=[
        "timestamp",
        "cpu_util",
        "cpu_freq_mhz",
        "num_processes",
        "session"
    ]
)

# -------- SAFE PATH HANDLING --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_path = os.path.join(OUTPUT_DIR, "cpu_log.csv")

# -------- APPEND MODE --------
if os.path.exists(output_path):
    df.to_csv(output_path, mode="a", header=False, index=False)
else:
    df.to_csv(output_path, index=False)

print("Session complete.")
print(f"Saved {len(df)} samples to cpu_log.csv")
