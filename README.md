# Cardiovascular Disease Prediction with Explainable Machine Learning

Final Degree Project in Computer Engineering at CUNEF Universidad.

## Overview

This project develops and evaluates machine learning models for predicting cardiovascular disease using the Heart Disease Cleveland dataset. It also applies SHAP to explain model predictions and identify the clinical variables with the greatest influence on classification.

## Dataset

The project uses a public, anonymized version of the Heart Disease Cleveland dataset with 303 observations, 13 predictor variables and one binary target indicating the presence or absence of heart disease.

- Original source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
- CSV source: [Heart Disease Cleveland on Kaggle](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland)

## Methodology

The analysis includes exploratory data analysis, preprocessing, five-fold cross-validation and comparison of four classification models:

- Naive Bayes
- Logistic Regression
- Random Forest
- Gradient Boosting

Performance is evaluated using accuracy, precision, recall, F1-score, ROC-AUC and confusion matrices. Recall receives particular attention because false negatives are especially important in a clinical classification problem.

## Explainability

SHAP is used to interpret model predictions and compare the influence of the predictor variables across models. The analysis identifies `ca`, `cp` and `thal` among the most influential variables.

## Key results

- Naive Bayes achieved the highest test recall: **0.800**.
- Random Forest achieved the highest test ROC-AUC: **0.900**.
- Simpler models remained competitive on this moderate-sized dataset.

## Technologies

Python, Pandas, NumPy, Scikit-learn, SHAP, Matplotlib and Seaborn.

## Repository structure

```text
data/          Public anonymized dataset and source information
images/        Confusion matrices and generated visualizations
results/       Generated CSV summaries
presentation/  Final defense presentation
report/        Final Degree Project report
src/           Python source code
```

## Run locally

1. Clone the repository.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the analysis from the repository directory:

   ```bash
   python src/tfg_machine_learning.py
   ```

Generated charts and result tables are saved automatically in `images/` and `results/`.

## Author

Ignacio Campo Fernández
Computer Engineering and Business Administration graduate, CUNEF Universidad
