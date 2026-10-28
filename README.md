# Stock Sentiment Prediction Using Machine Learning

A machine learning project that combines **social media sentiment analysis (NLP)** with historical stock market data to predict stock price movements. The project explores different classification algorithms and evaluates their performance using time-series validation.

## Features

- Combines NLP-based sentiment scores with financial market features.
- Implements feature selection and robust scaling for preprocessing.
- Trains and compares multiple machine learning classifiers.
- Uses walk-forward validation to evaluate performance on time-series data.
- Supports inference using saved models and preprocessing artifacts.

## Machine Learning Models

The project implements and evaluates:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Voting Ensemble

## Results

- **Prediction Accuracy:** 69.9%
- **Evaluation:** 54 walk-forward validation windows
- **Additional Metrics:** F1-score and ROC AUC

Walk-forward validation evaluates models on future observations relative to their training data, providing a more realistic assessment of stock prediction performance than random data splitting.

## Technologies

**Python | Scikit-learn | Pandas | NumPy | NLP**

## Model Pipeline

1. **Data Preparation:** Combine historical stock market features with social media sentiment data.
2. **Preprocessing:** Apply feature selection and robust scaling.
3. **Model Training:** Train and compare multiple classification algorithms.
4. **Evaluation:** Assess performance using walk-forward validation, accuracy, F1-score, and ROC AUC.
5. **Inference:** Load saved models and preprocessing artifacts to generate predictions on new data.

## Project Goal

Investigate whether combining social media sentiment with financial market indicators can improve machine learning-based stock movement prediction.


## Disclaimer

This project is intended for educational and research purposes only and should not be used as financial advice.
