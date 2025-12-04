import pandas as pd
import random

# Define how many samples we want (10,000 is plenty for a mini-project)
NUM_SAMPLES = 10000
data = []

print("Generating synthetic process data...")

for i in range(NUM_SAMPLES):
    # DECISION: Is this a "Heavy" process or a "Light" process?
    # 30% chance of being Heavy (Gaming/Rendering)
    # 70% chance of being Light (Typing/Browsing)
    is_cpu_heavy = random.random() < 0.3
    
    if is_cpu_heavy:
        # Heavy processes have bursts between 50ms and 150ms
        base_burst = random.randint(50, 150)
    else:
        # Light processes have bursts between 1ms and 10ms
        base_burst = random.randint(1, 10)

    # CREATE THE PATTERN
    # We generate 3 previous bursts (History) and 1 target burst (Future)
    # We add small random noise (-5 to +5) so it's not perfectly identical
    burst_1 = max(1, base_burst + random.randint(-5, 5))
    burst_2 = max(1, base_burst + random.randint(-5, 5))
    burst_3 = max(1, base_burst + random.randint(-5, 5))
    
    # The Target is what the AI needs to predict
    target_next_burst  = max(1, base_burst + random.randint(-5, 5))
    
    data.append([burst_1, burst_2, burst_3, target_next_burst])

# Save to CSV
df = pd.DataFrame(data, columns=['Burst_T-3', 'Burst_T-2', 'Burst_T-1', 'Target_Next_Burst'])
df.to_csv('cpu_burst_dataset.csv', index=False)

print(f"Success! Generated {NUM_SAMPLES} rows of data.")
print("File saved as: cpu_burst_dataset.csv")
print("Check your folder to ensure the file exists.")