import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('insurance.csv')
target_count = 8000
needed_new_rows = target_count - len(df)

print(f"Current Rows: {len(df)}")
print(f"Generating {needed_new_rows} new realistic rows...")

# 2. Prepare Data for the "Teacher" Model
# We must encode text to numbers so the model can learn the patterns
df_encoded = df.copy()
encoders = {}
for col in ['sex', 'smoker', 'region']:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df_encoded.drop('charges', axis=1)
y = df['charges']

# 3. Train the "Teacher" (Random Forest)
# This learns: "If Smoker=Yes and BMI=30, Price is High"
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Measure the natural variance (Noise) in the real data
# We will add this to our new data so it doesn't look "too perfect"
std_dev_noise = (y - model.predict(X)).std()

# 4. Generate New Patients (The Inputs)
# We sample from original data to keep the same % of Smokers/Sex/Region
new_X = X.sample(n=needed_new_rows, replace=True, random_state=42).copy()

# Add "Jitter" (Small variations) so they aren't exact clones
# BMI: Vary by +/- 1.5 points
new_X['bmi'] = new_X['bmi'] + np.random.normal(0, 1.5, size=len(new_X))
new_X['bmi'] = new_X['bmi'].round(2)

# Age: Vary by +/- 1 year (Clip to keep within 18-64)
new_X['age'] = new_X['age'] + np.random.randint(-1, 2, size=len(new_X))
new_X['age'] = new_X['age'].clip(18, 64)

# 5. Calculate the Bills for New Patients (The Target)
predicted_charges = model.predict(new_X)

# Add the natural noise we calculated earlier
synthetic_charges = predicted_charges + np.random.normal(0, std_dev_noise, size=len(new_X))
synthetic_charges = np.maximum(synthetic_charges, 1000) # Ensure no bill is below $1000
synthetic_charges = synthetic_charges.round(2)

# 6. Reconstruct the Final Dataset
# Decode numbers back to text (e.g., 0 -> 'female')
new_data = new_X.copy()
new_data['charges'] = synthetic_charges

for col, le in encoders.items():
    new_data[col] = le.inverse_transform(new_data[col].astype(int))

# Combine Old + New
final_df = pd.concat([df, new_data], axis=0).reset_index(drop=True)

# Save
final_df.to_csv('insurance_augmented.csv', index=False)
print(f"Success! New dataset has {len(final_df)} rows.")
print("Saved as 'insurance_augmented.csv'")