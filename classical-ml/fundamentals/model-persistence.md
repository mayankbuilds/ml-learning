# Model Persistence

## Objective

Learn how to save a trained machine learning pipeline and load it later without retraining the model.

## Dataset

- **Dataset:** Palmer Penguins
- **Records:** 344
- **Target:** Penguin species
- **Features:** Bill length, bill depth, flipper length, body mass, island, and sex
- **Missing values:** Handled through pipeline-based imputation

## Implementation

The accompanying `model-persistence.py` script:

1. Loads the dataset and removes records with a missing target.
2. Splits the data into training and testing sets.
3. Uses median imputation and standardization for numerical features.
4. Uses most-frequent imputation and one-hot encoding for categorical features.
5. Trains a Logistic Regression classifier inside a scikit-learn pipeline.
6. Saves the complete pipeline using `joblib`.
7. Loads the saved pipeline and compares its predictions with the original model's predictions.

## Results

- Total records: 344
- Training samples: 275
- Test samples: 69
- Test accuracy: 1.0000
- Original and loaded model predictions: identical

The test accuracy is specific to this split and does not guarantee perfect performance on new data.

## Key Learnings

- Model persistence avoids retraining when a saved model can be reused.
- Saving the complete preprocessing and model pipeline keeps inference steps consistent.
- `joblib` is useful for persisting scikit-learn models.
- Only load model files from trusted sources because loading serialized objects can execute malicious code.
