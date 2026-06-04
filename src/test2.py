import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import randint
from scipy import stats

# --- STEP 1: LOAD & SPLIT ---
np.random.seed(42)
# Simulating the data again for reproducibility
data = {
    'age': np.random.randint(18, 65, 1338),
    'sex': np.random.choice(['male', 'female'], 1338),
    'bmi': np.random.normal(30, 6, 1338),
    'children': np.random.randint(0, 5, 1338),
    'smoker': np.random.choice(['yes', 'no'], 1338, p=[0.2, 0.8]),
    'region': np.random.choice(['southwest', 'southeast', 'northwest', 'northeast'], 1338),
    'charges': np.random.exponential(13000, 1338) 
}
housing = pd.DataFrame(data)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["smoker"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

housing_train = strat_train_set.drop("charges", axis=1)
housing_labels = strat_train_set["charges"].copy()

# Manual Log Transform for Target (Fixes the Right Skew)
housing_labels_log = np.log1p(housing_labels)

# --- STEP 2: THE ENHANCEMENT (Custom Feature Engineering) ---
# This class fixes the "BMI Trap" and "3 Lines" issues we saw in the charts.

class MedicalFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # Expects a DataFrame with 'bmi', 'age', 'smoker'
        # We need to handle if X is a NumPy array (from previous steps) or DataFrame
        # Ideally, we put this at the START of the pipeline when X is still a DataFrame.
        
        X_out = X.copy()
        
        # 1. Convert smoker to numeric for math (yes=1, no=0)
        # We assume 'smoker' column exists. If strictly pipeline, we might need care here.
        # For safety in this specific script, we map it manually.
        if 'smoker' in X_out.columns:
            smoker_num = X_out['smoker'].map({'yes': 1, 'no': 0})
            
            # Enhancement 1: The "BMI Trap"
            # High BMI is dangerous mostly for smokers.
            X_out['bmi_smoker'] = X_out['bmi'] * smoker_num
            
            # Enhancement 2: The "Age Slope"
            # Costs rise faster with age for smokers.
            X_out['age_smoker'] = X_out['age'] * smoker_num
            
        return X_out

# --- STEP 3: BUILD THE PIPELINE ---

# 1. Feature Engineering (Happens FIRST on the raw DataFrame)
feature_engineering = MedicalFeatureEngineer()

# 2. Column Transformer (Happens SECOND)
# Note: We now have 2 new numerical columns: 'bmi_smoker' and 'age_smoker'
# We must include them in the numerical list.

num_attribs = ["age", "bmi", "children", "bmi_smoker", "age_smoker"]
cat_attribs = ["sex", "smoker", "region"]

num_pipeline = make_pipeline(
    SimpleImputer(strategy="mean"),
    StandardScaler()
)

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

# 3. Master Pipeline (Features -> Preprocessing)
# We need a wrapper to ensure FeatureEngineer receives the DataFrame
# The standard Pipeline converts to Array after the first step, which is tricky.
# PRO TIP: We apply Feature Engineering manually on X_train before the pipeline 
# to keep the code robust and simple for this example.

# --- APPLY ENHANCEMENTS MANUALLY ---
housing_train_enhanced = feature_engineering.transform(housing_train)

print("Enhancements Added: 'bmi_smoker' and 'age_smoker' columns created.")

# --- STEP 4: TRAIN & TUNE (Random Forest) ---
# We skip the comparison loop since we know Random Forest wins. 
# We go straight to Tuning.

print("\n--- Tuning Enhanced Model ---")

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("model", RandomForestRegressor(random_state=42))
])

param_distribs = {
    'model__n_estimators': randint(low=100, high=400),
    'model__max_features': randint(low=4, high=10), # Higher max_features because we added columns
    'model__max_depth': randint(low=10, high=50)
}

rnd_search = RandomizedSearchCV(
    full_pipeline, 
    param_distributions=param_distribs,
    n_iter=15, cv=3, 
    scoring='neg_root_mean_squared_error',
    random_state=42
)

# Train on ENHANCED data (with Log Labels)
rnd_search.fit(housing_train_enhanced, housing_labels_log)
final_model = rnd_search.best_estimator_

print(f"Best Parameters: {rnd_search.best_params_}")

# --- STEP 5: FINAL EVALUATION ---

# 1. Prepare Test Data (Must apply same Enhancements!)
X_test_enhanced = feature_engineering.transform(strat_test_set.drop("charges", axis=1))
y_test = strat_test_set["charges"].copy()

# 2. Predict (Log Scale)
final_predictions_log = final_model.predict(X_test_enhanced)

# 3. Inverse Transform (Real Dollars)
final_predictions = np.expm1(final_predictions_log)

# 4. RMSE
final_rmse = mean_squared_error(y_test, final_predictions, squared=False)

print(f"\nFinal Test RMSE (With Enhancements): ${final_rmse:,.2f}")

# 5. Confidence Interval
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
interval = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))

print(f"95% Confidence Interval: ${interval[0]:,.2f} — ${interval[1]:,.2f}")