# %%
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The least populated class in y has only 1 members")

# Use environment variable or relative path for the dataset directory
directory = os.getenv('Football_Dataset.csv')

# Load data
data = pd.read_csv(directory)
# Replace 'MISSING' with np.nan in the 'GAIN' column
data['GAIN'] = data['GAIN'].replace('MISSING', np.nan)
data.head()

# %%
# Display summary statistics and data types
data.info()
data.describe()

# Check the unique values and data types in the 'GAIN' column
print("Unique values in 'GAIN' column:", data['GAIN'].unique())
print("Data types of columns:")
print(data.dtypes)

# %%
# Identify rows with missing GAIN
missing_gain_rows = data[data['GAIN'].isna()]
missing_gain_indices = missing_gain_rows.index

# Display the shape of the dataset
print(f"Total rows: {data.shape[0]}, Rows with missing GAIN: {missing_gain_rows.shape[0]}")

# %%
# Impute missing values for DOWN and DIST using median in the original dataset
data['DOWN'] = data['DOWN'].fillna(data['DOWN'].median())
data['DIST'] = data['DIST'].fillna(data['DIST'].median())

# Display summary of imputed data
print("Summary statistics after imputing missing values for the entire dataset:")
data.describe()


# %%
# Identify rows with missing GAIN
missing_gain_rows = data[data['GAIN'].isna()]
missing_gain_indices = missing_gain_rows.index

# Display the shape of the dataset
print(f"Total rows: {data.shape[0]}, Rows with missing GAIN: {missing_gain_rows.shape[0]}")

# Define numerical and categorical features
numeric_features = ['DOWN', 'DIST', 'LOS', 'SCOREDIFF']
categorical_features = ['FORMATION', 'PLAYCALL', 'PLAYTYPE', 'PASSER', 'DEFTEAM', 'KEY.PLAYER.POSITION', 'PASSRESULT']

# Handle outliers using the IQR method
def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

# Apply outlier handling to numerical features
for feature in numeric_features:
    data = handle_outliers(data, feature)

# Display summary statistics after handling outliers
data.describe()


# %%
import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The least populated class in y has only 1 members")

# Impute and scale numerical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Impute and encode categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Separate data into known and unknown GAIN for training purposes
known_gain_data = data[data['GAIN'].notna()].copy()
known_gain_data['GAIN'] = known_gain_data['GAIN'].astype(float)

# Ensure there are actual missing values in the 'GAIN' column
missing_gain_data = data[data['GAIN'].isna()].drop(columns=['GAIN', 'PLAYID'])

# Verify the missing_gain_data to ensure it's not empty
print("Shape of missing_gain_data:", missing_gain_data.shape)
if missing_gain_data.empty:
    raise ValueError("missing_gain_data is empty. Check the 'GAIN' column for proper labeling of missing values.")

# Prepare the data
X_known = known_gain_data.drop(columns=['PLAYID', 'GAIN'])
y_known = known_gain_data['GAIN']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_known, y_known, test_size=0.2, random_state=42)

# Fit the preprocessor on the training data
preprocessor.fit(X_train)
X_train_preprocessed = preprocessor.transform(X_train)
X_val_preprocessed = preprocessor.transform(X_val)

# Save the preprocessor
joblib.dump(preprocessor, 'preprocessor.pkl')

# Define models
models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'CatBoost': CatBoostRegressor(random_state=42, verbose=0),
    'GradientBoosting': GradientBoostingRegressor(random_state=42)
}

# Cross-validation and model selection
best_rmse = float('inf')
best_model_name = None
best_model = None

# Use StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    # Note: We need to bin the target variable to use StratifiedKFold
    y_train_binned = pd.cut(y_train, bins=5, labels=False)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error')
    mean_rmse = -scores.mean()
    print(f'{name} RMSE: {mean_rmse}')
    if mean_rmse < best_rmse:
        best_rmse = mean_rmse
        best_model_name = name
        best_model = pipeline  # Use pipeline as the best model

print(f'Best model: {best_model_name} with RMSE: {best_rmse}')

# Hyperparameter tuning for the best model
if best_model_name == 'RandomForest':
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [None, 10, 20],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    }
elif best_model_name == 'XGBoost':
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [3, 5, 7],
        'regressor__learning_rate': [0.01, 0.1, 0.2],
        'regressor__colsample_bytree': [0.8, 0.9, 1.0]
    }
elif best_model_name == 'CatBoost':
    param_grid = {
        'regressor__iterations': [100, 200, 300],
        'regressor__depth': [4, 6, 10],
        'regressor__learning_rate': [0.01, 0.1, 0.2],
        'regressor__l2_leaf_reg': [1, 3, 5]
    }
elif best_model_name == 'GradientBoosting':
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [3, 5, 7],
        'regressor__learning_rate': [0.01, 0.1, 0.2],
        'regressor__subsample': [0.8, 0.9, 1.0]
    }

random_search = RandomizedSearchCV(best_model, param_distributions=param_grid, n_iter=50, cv=cv, verbose=2, n_jobs=-1, random_state=42, scoring='neg_root_mean_squared_error')
random_search.fit(X_train, y_train)

# Get the best model
best_model = random_search.best_estimator_

# Validate the best model
y_pred = best_model.predict(X_val)
rmse = root_mean_squared_error(y_val, y_pred)
print(f'Validation RMSE after hyperparameter tuning: {rmse}')

# Save the model for later use
joblib.dump(best_model, 'gain_prediction_model.pkl')

# %%
# Load the entire pipeline (preprocessor + model)
best_model = joblib.load('gain_prediction_model.pkl')

# Predict the GAIN values for the missing rows using the entire pipeline
predicted_gain = best_model.predict(missing_gain_data)

# Calculate the prediction interval for the sum of GAIN values using bootstrapping
n_iterations = 1000
bootstrapped_sums = []

for _ in range(n_iterations):
    # Resample with replacement
    bootstrapped_sample = resample(predicted_gain)
    bootstrapped_sums.append(bootstrapped_sample.sum())

# Calculate the 5th and 95th percentiles for the 90% prediction interval
lower_bound = np.percentile(bootstrapped_sums, 5)
upper_bound = np.percentile(bootstrapped_sums, 95)

sum_gain = predicted_gain.sum()
print(f'Sum of predicted GAIN: {sum_gain}')
print(f'90% prediction interval for the sum of GAIN: ({lower_bound}, {upper_bound})')

# Store PLAYID values before replacing missing GAIN values
missing_playids = data.loc[data['GAIN'].isna(), 'PLAYID'].values

# Insert the predicted values back into the original dataset
data.loc[data['GAIN'].isna(), 'GAIN'] = predicted_gain

# Create the output DataFrame
output = pd.DataFrame({
    'PLAYID': missing_playids,
    'GAIN': predicted_gain
})

# Display the first few predictions for missing GAIN values
print("First few predictions for missing GAIN values:")
print(output.head())


# %%
# Define individual models
random_forest = RandomForestRegressor(random_state=42)
xgboost = XGBRegressor(random_state=42)
catboost = CatBoostRegressor(random_state=42, verbose=0)
gradient_boosting = GradientBoostingRegressor(random_state=42)

# Hyperparameter tuning for individual models
param_grid = {
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'xgboost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'colsample_bytree': [0.8, 0.9, 1.0]
    },
    'catboost': {
        'iterations': [100, 200, 300],
        'depth': [4, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'l2_leaf_reg': [1, 3, 5]
    },
    'gradient_boosting': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }
}

models = {
    'random_forest': random_forest,
    'xgboost': xgboost,
    'catboost': catboost,
    'gradient_boosting': gradient_boosting
}

best_estimators = {}

for name, model in models.items():
    search = RandomizedSearchCV(model, param_distributions=param_grid[name], n_iter=50, cv=5, verbose=2, n_jobs=-1, random_state=42, scoring='neg_root_mean_squared_error')
    search.fit(X_train_preprocessed, y_train)
    best_estimators[name] = search.best_estimator_
    print(f'Best {name} model: {search.best_estimator_}')

# Create the stacking ensemble
stacking_ensemble = VotingRegressor(estimators=[
    ('random_forest', best_estimators['random_forest']),
    ('xgboost', best_estimators['xgboost']),
    ('catboost', best_estimators['catboost']),
    ('gradient_boosting', best_estimators['gradient_boosting'])
])

# Fit the ensemble model
stacking_ensemble.fit(X_train_preprocessed, y_train)

# Validate the ensemble model
y_pred = stacking_ensemble.predict(X_val_preprocessed)
rmse = root_mean_squared_error(y_val, y_pred)
print(f'Validation RMSE with Stacking Ensemble: {rmse}')

# Save the ensemble model
joblib.dump(stacking_ensemble, 'stacking_ensemble_model.pkl')

# Load the entire pipeline (preprocessor + ensemble model)
best_model = joblib.load('stacking_ensemble_model.pkl')

# Predict the GAIN values for the missing rows using the entire pipeline
missing_gain_data_preprocessed = preprocessor.transform(missing_gain_data)
predicted_gain = best_model.predict(missing_gain_data_preprocessed)

# Ensure the lengths match before assigning
assert len(predicted_gain) == len(missing_gain_indices), "Length mismatch between predictions and missing data."

# Insert the predicted values back into the original dataset
data.loc[missing_gain_indices, 'GAIN'] = predicted_gain

# Calculate the prediction interval for the sum of GAIN values using bootstrapping
n_iterations = 1000
bootstrapped_sums = []

for _ in range(n_iterations):
    # Resample with replacement
    bootstrapped_sample = resample(predicted_gain)
    bootstrapped_sums.append(bootstrapped_sample.sum())

# Calculate the 5th and 95th percentiles for the 90% prediction interval
lower_bound = np.percentile(bootstrapped_sums, 5)
upper_bound = np.percentile(bootstrapped_sums, 95)

sum_gain = predicted_gain.sum()
print(f'Sum of predicted GAIN: {sum_gain}')
print(f'90% prediction interval for the sum of GAIN: ({lower_bound}, {upper_bound})')

# Store PLAYID values before replacing missing GAIN values
missing_playids = data.loc[missing_gain_indices, 'PLAYID'].values

# Create the output DataFrame
ensemble_output = pd.DataFrame({
    'PLAYID': missing_playids,
    'GAIN': predicted_gain
})

# Display the first few predictions for missing GAIN values
print("First few predictions for missing GAIN values:")
print(ensemble_output.head())


# Save the predictions to a CSV file
ensemble_output.to_csv('Predictions.csv', index=False)
