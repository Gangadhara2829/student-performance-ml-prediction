# Student Performance Predictor

A generic Machine Learning project that predicts whether a student will pass or fail based on their study habits and past performance. Built with Python, Pandas, and Scikit-Learn.

## Overview
This project demonstrates the fundamental steps of building a classification model:
1. **Data Loading:** Reads student records from `student_data.csv`.
2. **Preprocessing:** 
   - Handles missing values (fills missing numeric data with the column mean, and categorical data with the mode).
   - Encodes categorical data (converts "Yes"/"No" text into `1`/`0`).
3. **Model Training:** Splits data into training and testing sets, then trains a Logistic Regression model.
4. **Evaluation:** Outputs the model's accuracy on the test set.
5. **Interactive Prediction:** Allows the user to input custom student data interactively via the command line to get a Pass/Fail prediction.

## Requirements
Ensure you have Python installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to Run
Run the Python script from your terminal:

```bash
python predict_performance.py
```

## Project Structure
- `student_data.csv`: Dataset containing features like `study_hours`, `attendance_percentage`, and the target variable `passed` (1 for pass, 0 for fail).
- `predict_performance.py`: Main script containing data preprocessing, model training, and the interactive prompt.
- `requirements.txt`: List of required Python packages (`pandas`, `scikit-learn`).

## Algorithm
This project uses **Logistic Regression** for binary classification to categorize a student into one of two outcomes: Pass (1) or Fail (0).
