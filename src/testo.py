import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import randint

# --- 1. LOAD DATA ---
medic = pd.read_csv("insurance.csv")

# --- 2. STRATIFIED SPLIT ---
# We split first to avoid data leakage
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(medic, medic["smoker"]):
    strat_train_set = medic.loc[train_index]
    strat_test_set = medic.loc[test_index]

housing = strat_train_set.drop("charges", axis=1)
housing_labels = strat_train_set["charges"].copy()

# --- 3. THE FIX: FEATURE ENGINEERING CLASS ---
# This runs on the RAW dataframe, BEFORE OneHotEncoding.
class MedicalFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_out = X.copy()
        
        # We manually map 'yes/no' to 1/0 just for the math calculation.
        # This DOES NOT replace the column for the final model (OneHot will do that later).
        # It creates a TEMPORARY numeric helper.
        if 'smoker' in X_out.columns:
            smoker_num = X_out['smoker'].map({'yes': 1, 'no': 0})
            
            # Create the interaction columns
            X_out['bmi_smoker'] = X_out['bmi'] * smoker_num
            X_out['age_smoker'] = X_out['age'] * smoker_num
            
        return X_out

# --- 4. DEFINE PREPROCESSING PIPELINES ---

# A. Numerical Pipeline
# Note: We MUST include the NEW columns ('bmi_smoker', 'age_smoker') in this list
# so the StandardScaler knows to scale them too.
num_attribs = ["age", "bmi", "children", "bmi_smoker", "age_smoker"]
num_pipeline = make_pipeline(
    SimpleImputer(strategy="mean"),
    StandardScaler()
)

# B. Categorical Pipeline (User's Code)
cat_attribs = ["sex", "smoker", "region"]
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

# C. The Column Transformer (The Processor)
# It applies the num_pipeline to the NEW numerical columns
# and the cat_pipeline to the ORIGINAL categorical columns.
preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

# --- 5. THE MASTER PIPELINE ---
# This is the "Production" Chain. 
# 1. Engineer Features (Raw -> Enhanced)
# 2. Preprocess (Enhanced -> Scaled/Encoded Matrix)
# 3. Model (Matrix -> Prediction)

full_pipeline = Pipeline([
    ('feature_engineering', MedicalFeatureEngineer()), # <--- HAPPENS FIRST!
    ('preprocessing', preprocessing),                  # <--- HAPPENS SECOND!
    ('model', TransformedTargetRegressor(              # <--- HAPPENS LAST!
        regressor=RandomForestRegressor(random_state=42),
        func=np.log1p,
        inverse_func=np.expm1
    ))
])

# --- 6. EXECUTION ---
print("Training the full pipeline...")
# We fit the whole chain on the raw training data
full_pipeline.fit(housing, housing_labels)

# --- 7. EVALUATION ---
X_test = strat_test_set.drop("charges", axis=1)
y_test = strat_test_set["charges"].copy()

predictions = full_pipeline.predict(X_test)
final_rmse = mean_squared_error(y_test, predictions, squared=False)

print(f"Final RMSE: ${final_rmse:,.2f}")