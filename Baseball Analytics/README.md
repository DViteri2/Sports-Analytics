
# Baseball Swing Probability Prediction

This repository contains the code for predicting the probability of a swing in baseball pitches using machine learning techniques. The goal is to provide a SwingProbability for all pitches in season 3 of the dataset.

## Overview

In this project, I aimed to predict whether a batter will swing at a given pitch during a baseball game. The approach involved training and evaluating various machine learning models on a dataset containing pitch-related features and swing/no-swing labels. Accurate prediction of swings is crucial for strategic planning and decision-making in baseball, as it helps teams understand and anticipate the outcomes of different pitches.

### Steps Involved

1. **Data Preprocessing**:
    - Loaded datasets from three seasons (year1, year2, and year3).
    - Performed Exploratory Data Analysis (EDA) to understand data structure, identify missing values, and explore unique values in different columns.
    - Concatenated year1 and year2 datasets into a combined dataset (combined_df).
    - Defined swing events based on descriptions and created a binary column 'swing'.
    - One-hot encoded the 'pitch_type' categorical feature.
    - Dropped rows with missing values from the combined dataset.

2. **Feature Engineering**:
    - Selected numeric features (release_speed, balls, strikes, pfx_x, pfx_z, plate_x, plate_z, sz_top, sz_bot) and one-hot encoded pitch_type features.
    - Combined numeric and one-hot encoded features to create the final feature matrix (X_train).
    - Set the target variable (y_train) as the 'swing' column.

3. **Model Selection and Evaluation**:
    - Evaluated four different models: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
    - Performed 3-fold cross-validation using the ROC AUC metric to estimate model performance.
    - Selected the Random Forest Classifier based on the highest mean ROC AUC score.

4. **Best Model Training and Testing**:
    - Trained the Random Forest Classifier on the entire training set (X_train, y_train).
    - Predicted swing probabilities for the testing set (X_test) using the trained model.
    - Converted predictions to binary labels using a threshold of 0.5.
    - Evaluated model performance using accuracy, confusion matrix, and ROC AUC.

5. **Results**:
    - The Random Forest Classifier achieved an accuracy of 0.841 and an ROC AUC score of 0.92.
    - The confusion matrix revealed insights into true positives, true negatives, false positives, and false negatives.
    - Feature importance analysis highlighted pitch location and movement as crucial factors for predicting swing events.

## Project Structure

- `baseball_swing_prediction.py`: Main script for data processing, model training, and prediction.

## Installation

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/DViteri2/Sports-Analytics.git
   cd Sports-Analytics

## Disclaimer

**Note**: The dataset used in this project is not included in this repository due to privacy and licensing constraints. Please use your own dataset to replicate the results.

## Installation

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/DViteri2/Sports-Analytics.git
   cd Sports-Analytics
   ```

2. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Main Script**:
   ```sh
   python baseball_swing_prediction.py
   ```

