
# Sports Analytics

This repository contains the code and data for a football sports analytics project.

## Overview

In this project, I aimed to analyze and predict key performance metrics in football games using machine learning techniques. The primary focus was on predicting the GAIN values, which represent the yards gained in a play. Accurate prediction of GAIN is crucial for strategic planning and decision-making in football, as it helps teams understand and anticipate the outcomes of different plays.

The workflow involved several key steps:

1. **Data Preprocessing**: Cleaning and preparing the raw dataset by handling missing values, encoding categorical variables, and scaling numerical features.
2. **Model Training**: Training multiple regression models, including RandomForest, XGBoost, CatBoost, and GradientBoosting, to predict the GAIN values.
3. **Hyperparameter Tuning**: Performing hyperparameter tuning to optimize the models for better performance.
4. **Prediction**: Using the trained models to predict missing GAIN values in the dataset and validating the model performance.
5. **Ensemble Modeling**: Combining the predictions of multiple models using a stacking ensemble to improve the accuracy of the predictions.

## Project Structure

- `Sports_Analytics_Football.py`: Main script for data processing, model training, and prediction.
- `preprocessor.pkl`: Preprocessor for data transformation.
- `gain_prediction_model.pkl`: Trained model for predicting GAIN values.
- `stacking_ensemble_model.pkl`: Trained ensemble model for prediction.

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
   python Sports_Analytics_Football.py
   ```

