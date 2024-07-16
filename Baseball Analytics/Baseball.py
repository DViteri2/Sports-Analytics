# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import plotly.express as px

# %%
# Set the directory where your files are located
directory = "C:/path/"

# %%
# Load three files of baseball pitches over the years
year1 = pd.read_csv(directory + 'year1.csv')
year2 = pd.read_csv(directory + 'year2.csv')
year3 = pd.read_csv(directory + 'year3.csv')

# %%
def perform_eda(df, name):
    try:
        df['pitch_id'] = df['pitch_id'].fillna('').astype(str).str.rstrip('.0')
    except Exception as e:
        print(f"Error converting 'pitch_id': {e}")

    try:
        df['batter'] = df['batter'].astype('object')
    except Exception as e:
        print(f"Error converting 'batter': {e}")

    try:
        df['pitcher'] = df['pitcher'].astype('object')
    except Exception as e:
        print(f"Error converting 'pitcher': {e}")

    print(f"\nEDA for {name}")
    print("-" * 40)

    try:
        print("First few rows:")
        print(df.head())
    except Exception as e:
        print(f"Error printing first few rows: {e}")

    try:
        print("\nInfo:")
        print(df.info())
    except Exception as e:
        print(f"Error printing info: {e}")

    try:
        print("\nMissing values:")
        print(df.isnull().sum())
    except Exception as e:
        print(f"Error printing missing values: {e}")

    try:
        print("\nSummary statistics:")
        print(df.describe())
    except Exception as e:
        print(f"Error printing summary statistics: {e}")

    try:
        print("\nUnique values in 'season':")
        print(df['season'].unique())
    except Exception as e:
        print(f"Error printing unique values in 'season': {e}")

    try:
        print("\nUnique values in 'description':")
        print(df['description'].unique())
    except Exception as e:
        print(f"Error printing unique values in 'description': {e}")

    try:
        print("\nUnique values in 'pitch_type':")
        print(df['pitch_type'].unique())
    except Exception as e:
        print(f"Error printing unique values in 'pitch_type': {e}")

    print("-" * 40)


# %%
perform_eda(year1, 'year1')
perform_eda(year2, 'year2')
perform_eda(year3, 'year3')

# %%
# Concatenate year1 and year2 DataFrames
combined_df = pd.concat([year1, year2], ignore_index=True)

# Define swing events based on description for combined DataFrame
swing_keywords = ['hit_into_play', 'swinging_strike', 'foul', 'foul_tip', 'hit_by_pitch', 'foul_bunt', 'swinging_strike_blocked', 'missed_bunt', 'bunt_foul_tip']
combined_df['swing'] = combined_df['description'].isin(swing_keywords).astype(int)
combined_df

# %%
# One-hot encode categorical features
combined_df = pd.get_dummies(combined_df, columns=['pitch_type'], prefix='pitch_type')

# Drop rows with missing values
combined_df.dropna(inplace=True)
combined_df

# %%
# Features for modeling
numeric_features = ['release_speed', 'balls', 'strikes', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'sz_top', 'sz_bot']

# Extract numeric features
X_numeric = combined_df[numeric_features]

# Extract one-hot encoded features
X_pitch_type = combined_df.filter(regex='^pitch_type_')

# Concatenate numeric and one-hot encoded features
X_train = pd.concat([X_numeric, X_pitch_type], axis=1)

# Target variable
y_train = combined_df['swing']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# %%
# List of models to evaluate
models = [
    LogisticRegression(max_iter=2000),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=100),
    GradientBoostingClassifier()
]

# Cross-validation
n_folds = 3

for model in models:
    scores = cross_val_score(model, X_train, y_train, cv=n_folds, scoring='roc_auc')
    print(f"{model.__class__.__name__}: Mean ROC AUC = {scores.mean():.3f} (+/- {scores.std():.3f})")

# %%
# Initialize and train the Random Forest Classifier
best_model = RandomForestClassifier(n_estimators=100) 
best_model.fit(X_train, y_train)

# Predict swing probabilities for the testing set using the best model
y_prob = best_model.predict_proba(X_test)[:, 1]

# Convert probabilities to binary predictions using a threshold of 0.5
y_pred = np.where(y_prob > 0.5, 1, 0)

# %%
# Evaluate the best model using accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

# Define custom labels
custom_labels = ['True', 'False']

# Print confusion matrix with custom labels
print("Confusion Matrix:")
print(pd.DataFrame(conf_matrix, index=custom_labels, columns=custom_labels))

# %%
# Compute ROC curve and ROC AUC for the best model
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

# Plot ROC curve using Plotly
fig_roc = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC = {roc_auc:.2f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
fig_roc.add_shape(
    type='line', line=dict(dash='dash', color='#EF3340', width=2),
    x0=0, x1=1, y0=0, y1=1
)

# Update plot colors and background
fig_roc.update_traces(line=dict(color='#00A3E0'))
fig_roc.update_layout(plot_bgcolor='rgba(255, 255, 255, 0)', paper_bgcolor='rgba(255, 255, 255, 0)')
fig_roc.update_xaxes(showline=True, linecolor='#000000', showgrid=False)
fig_roc.update_yaxes(showline=True, linecolor='#000000', showgrid=False)

# Save the ROC curve plot as HTML
fig_roc.write_html('roc_curve.html')

# %%
# Get the feature importances from the best Random Forest Classifier model
importances = best_model.feature_importances_

# Get the feature names from X_train
feature_names = X_train.columns

# Create a DataFrame to hold feature names and their corresponding importances
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort the DataFrame by the importance values
importance_df = importance_df.sort_values(by='Importance', ascending=True)

# Create bar chart
fig_importance = px.bar(importance_df,
                         x='Importance',
                         y='Feature',
                         title='Feature Importance Plot',
                         labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
                         orientation='h',
                         color='Importance',
                         color_continuous_scale=['#41748D', '#41748D'])

# Update plot colors and background
fig_importance.update_layout(plot_bgcolor='rgba(255, 255, 255, 0)', paper_bgcolor='rgba(255, 255, 255, 0)')
fig_importance.update_xaxes(showline=True, linecolor='#000000', showgrid=False)
fig_importance.update_yaxes(showline=True, linecolor='#000000', showgrid=False)

# Save the feature importance plot as HTML
fig_importance.write_html('feature_importance.html')


# %%
# Preprocess year3 data
year3_processed = year3.copy()  # Create a copy to avoid modifying the original DataFrame
year3_processed.dropna(inplace=True)  # Drop rows with missing values, if any

# One-hot encode categorical features
year3_processed = pd.get_dummies(year3_processed, columns=['pitch_type'], prefix='pitch_type')

# Align the columns of year3_processed with the columns used during model training
year3_processed = year3_processed.reindex(columns=X_train.columns, fill_value=0)

# Predict swing probabilities for year3 data using the Random Forest Classifier
swing_probabilities = best_model.predict_proba(year3_processed.dropna())[:, 1]

# Create a copy of the processed year3 data to avoid modifying the original DataFrame
year3_with_predictions = year3_processed.copy()

# Append swing probabilities to year3 data
year3_with_predictions['SwingProbability'] = swing_probabilities

# Save the modified year3 data with swing probabilities as 'validation.csv'
year3_with_predictions.to_csv('validation.csv', index=False)

# %%



