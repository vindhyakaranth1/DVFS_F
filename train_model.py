import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

print("1. Loading dataset...")
# Load the data you generated in Step 1
df = pd.read_csv('cpu_burst_dataset.csv')

# 2. Separate Inputs (X) and Output (y)
# X = The things the AI "sees" (The history of the last 3 bursts)
X = df[['Burst_T-3', 'Burst_T-2', 'Burst_T-1']]
# y = The thing the AI tries to predict (The next burst)
y = df['Target_Next_Burst']

# 3. Split into Training and Testing
# We keep 20% of data hidden to test the AI later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("2. Training the AI model...")
# We use Linear Regression (Best for simple numeric patterns)
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluate the Model
print("3. Testing accuracy...")
predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)
print(f"   Mean Absolute Error: {error:.2f} ms")
# NOTE: If error is low (e.g., < 5ms), your model is working!

# 5. Save the Brain
joblib.dump(model, 'burst_predictor.pkl')
print("\nSuccess! Model saved as 'burst_predictor.pkl'")

# --- QUICK TEST ---
print("\n--- Live Test ---")
# Let's ask the AI to predict for a "Heavy" process sequence
sample_input = [[100, 102, 98]] 
predicted_val = model.predict(sample_input)[0]
print(f"Input History: {sample_input[0]}")
print(f"AI Prediction: {predicted_val:.2f} ms")
print("-----------------")