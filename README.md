# Titanic Dataset Analysis - Predictive Model for Survival

## Overview
This project implements a comprehensive machine learning pipeline to predict Titanic passenger survival using data cleaning, feature engineering, and feature selection techniques. The models are trained on the famous Titanic - Machine Learning from Disaster dataset.

## Project Structure
```
titanic_assignment/
├── data/
│   ├── train.csv                 # Raw training dataset
│   ├── test.csv                  # Test dataset for predictions
│   ├── train_cleaned.csv         # Cleaned training data
│   ├── train_engineered.csv      # Data with engineered features
│   └── train_final.csv           # Final selected features
│
├── notebooks/
│   └── Titanic_Analysis.ipynb    # Full exploration & analysis
│
├── scripts/
│   ├── data_cleaning.py          # Data cleaning module
│   ├── feature_engineering.py    # Feature creation module
│   ├── feature_selection.py      # Feature selection module
│   └── __init__.py
│
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

---

## Approach

### 1. **Data Cleaning (Part 1 - 10 Marks)**

#### Missing Value Handling
- **Age**: Imputed with **median by Pclass** (stratified, accounts for class differences)
- **Embarked**: Imputed with **mode** (most common port: Southampton)
- **Fare**: Imputed with **median by Pclass** (class-based imputation)
- **Cabin**: Converted to binary **HasCabin** indicator (90% missing - too sparse to impute)

#### Outlier Detection & Handling
- **Method**: IQR (Interquartile Range) - 1.5 × IQR rule
- **Approach**: Cap outliers rather than remove to preserve data
  - Extreme values may represent real edge cases
  - Capping preserves data structure while limiting model influence
- **Fare**: Capped at Q3 + 1.5 × IQR (high fares represent luxury accommodations)
- **Age**: Naturally bounded [0, 120], minimal adjustment needed

#### Data Consistency
- Standardized **Sex** values to lowercase (`male`, `female`)
- Verified numeric types for key columns
- Removed duplicate records (if any)

---

### 2. **Feature Engineering (Part 2 - 30 Marks)**

#### Family Features
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `FamilySize` | SibSp + Parch + 1 | Larger families may have different survival rates |
| `IsAlone` | 1 if FamilySize == 1 else 0 | Isolated passengers had distinct survival patterns |

#### Name-Based Features
- **Title Extraction**: Extracted titles (Mr, Mrs, Miss, Master, etc.) from passenger names
  - Aggregated rare titles (< 10 occurrences) as "Other"
  - Titles proxy for age, gender, social status

#### Age-Based Features
- **AgeGroup**: Binned into categories
  - `Child` [0-12): High survival (especially females)
  - `Teen` [12-18): Moderate survival
  - `Adult` [18-60): Lower survival (higher risk)
  - `Senior` [60, 120]: Lower survival

#### Fare-Based Features
| Feature | Description |
|---------|-------------|
| `FarePerPerson` | Fare / FamilySize - normalized fare by group size |
| `FareCategory` | Quartile bins (Low, Medium, High, VeryHigh) |
| `LogFare` | Log transformation of Fare (handles right skew) |

#### Categorical Encoding
- **One-Hot Encoding** for:
  - `Sex`: male/female split
  - `Embarked`: Southampton/Cherbourg/Queenstown
  - `Title`: Extracted titles
  - `Pclass`: Passenger class (1st/2nd/3rd)

#### Interaction Features
- **Age × Title**: (Age_Title_Mr, Age_Title_Mrs, Age_Title_Miss)
  - Different title groups had different age-survival relationships
- **Fare × Pclass**: Higher classes paid more; interaction captures class-fare synergy
- **FamilySize × Pclass**: Family dynamics differed by class

#### Transformations
- **Log Transformation** (using `log1p`):
  - Handles right-skewed distributions in Fare and Age
  - Reduces impact of extreme values without removing them
  - Improves model performance for tree-based models

---

### 3. **Feature Selection (Part 3 - 10 Marks)**

#### Selection Criteria
1. **Correlation Analysis**: Identified features with meaningful target correlation (|r| > 0.1)
2. **Feature Importance**: Random Forest importance ranking
3. **Multicollinearity Check**: Removed redundant features (correlation > 0.95)

#### Feature Selection Strategy
- Combined **Feature Importance** (Random Forest) with **Correlation** analysis
- Removed highly correlated feature pairs (kept higher importance feature)
- Retained top 15-20 features balancing predictive power and model simplicity

#### Top Selected Features (by importance)
1. Titles (extracted from names)
2. Fare-based features (basic fare, log fare, fare per person)
3. Family features (FamilySize, IsAlone)
4. Passenger class (Pclass)
5. Sex (male/female binary)
6. Age-based features
7. Embarkation port
8. Interaction features (Age × Title, Fare × Pclass)

---

## Key Findings

### Survival Patterns
1. **Sex**: Strong predictor - females ~73% survival, males ~19%
2. **Pclass**: Inverse relationship - 1st class ~63%, 3rd class ~24%
3. **Age**: Children had higher survival; elderly lower
4. **Title**: Mrs/Miss titles → high survival; Mr → low
5. **Fare**: Positive correlation - paid more → higher priority
6. **Family**: Traveling alone → lower survival; small families beneficial

### Data Insights
- **Missing Data**: 
  - Age: 177/891 (20%)
  - Cabin: 687/891 (77%) - mostly missing
  - Embarked: 2/891 (0.2%)
- **Outliers**: Few extreme fare values (luxury suites)
- **Imbalance**: 38% survived, 62% perished

---

## Data Cleaning Decisions

| Issue | Decision | Justification |
|-------|----------|---------------|
| Missing Age (20%) | Impute with median by class | Preserves class distributions |
| Missing Embarked (0.2%) | Impute with mode | Minimal data loss |
| Missing Fare (0.1%) | Impute with median by class | Class-based structure preserved |
| Missing Cabin (77%) | Convert to binary indicator | Too sparse for meaningful imputation |
| Age/Fare outliers | Cap using IQR method | Preserves data while limiting influence |
| Duplicates | Remove completely | No genuine duplicate passengers |

---

## Running the Pipeline

### Prerequisites
```bash
pip install -r requirements.txt
```

### Execute Full Pipeline
```bash
# 1. Data Cleaning
python scripts/data_cleaning.py
# Output: data/train_cleaned.csv

# 2. Feature Engineering
python scripts/feature_engineering.py
# Output: data/train_engineered.csv

# 3. Feature Selection
python scripts/feature_selection.py
# Output: data/train_final.csv
```

### Run Jupyter Notebook
```bash
jupyter notebook notebooks/Titanic_Analysis.ipynb
```

---

## Results Summary

- **Final Features**: 20 selected from 40+ engineered
- **Data Retention**: 100% of training data (no removal, only transformation)
- **Missing Values**: 0 remaining after cleaning
- **Outliers Handled**: Capped using IQR method
- **Ready for Modeling**: Dataset prepared for ML classifier training

---

## Technologies Used
- **Python 3.13**
- **pandas**: Data manipulation & analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning & preprocessing
- **matplotlib & seaborn**: Visualizations
- **jupyter**: Interactive exploration

---

## Author
Ian Pkiach Lokur
