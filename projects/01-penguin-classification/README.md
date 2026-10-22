# Palmer Penguins Species Classification

## Overview

An end-to-end machine learning project that predicts penguin species from physical measurements and categorical features using Logistic Regression.

## Dataset

- **Source:** Palmer Penguins dataset
- **Records:** 344
- **Target:** `species`
- **Classes:** Adelie, Chinstrap, Gentoo

## Workflow

1. Load the dataset and remove rows with missing target labels.
2. Split the data into training and test sets using stratification.
3. Impute missing numerical and categorical values.
4. Scale numerical features and one-hot encode categorical features.
5. Train a Logistic Regression classifier in a scikit-learn pipeline.
6. Evaluate the model using accuracy, precision, recall, and F1-score.
7. Save the complete trained pipeline with `joblib`.

## Results

| Metric | Result |
|---|---:|
| Training samples | 275 |
| Test samples | 69 |
| Test accuracy | 100% |
| Adelie F1-score | 1.00 |
| Chinstrap F1-score | 1.00 |
| Gentoo F1-score | 1.00 |

These results apply to the current train-test split and do not guarantee identical performance on other data.

## Run the Project

From the repository root:

```bash
python projects/01-penguin-classification/train.py
```

## Technologies

- Python
- pandas
- scikit-learn
- joblib

## Key Learnings

- Building a complete classification workflow
- Handling missing values and categorical features
- Using preprocessing pipelines to avoid train-test leakage
- Evaluating classification models
- Saving trained models for later use
