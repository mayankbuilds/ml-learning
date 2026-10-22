# Feature Engineering

## Objective

Feature engineering means creating or transforming input features to make data more useful for machine learning models.

## Dataset

- **Dataset:** Palmer Penguins
- **Target:** `species`
- **Input:** Penguin measurements and categorical attributes

## Techniques Demonstrated

### 1. Feature Creation

Created `body_mass_kg` by converting `body_mass_g` into kilograms:

`body_mass_kg = body_mass_g / 1000`

Created `bill_length_depth_ratio` using bill length divided by bill depth:

`bill_length_depth_ratio = bill_length_mm / bill_depth_mm`

These are example derived features; they are not guaranteed to improve model performance.

### 2. Missing and Invalid Values

Original missing measurements can produce missing derived features. Zero denominators can produce infinite ratios, so infinite values are converted to missing values.

Imputation should be handled using the preprocessing pipeline.

### 3. Target Separation

The target `species` is separated from input features before training. The `year` column is excluded from this demonstration.

### 4. Train-Test Split

The data is split into training and test sets using stratification and a fixed random seed.

- Training rows: 275
- Test rows: 69

## Key Learnings

- Domain knowledge can help create useful derived features.
- Feature creation does not guarantee better model performance; evaluate it experimentally.
- Handle missing and invalid values appropriately.
- Keep the target separate from input features to avoid target leakage.
