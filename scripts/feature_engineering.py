"""
Feature Engineering Module for Titanic Dataset
===============================================
Creates derived features, encodings, and transformations.

Author: Titanic Assignment
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os


def engineer_family_features(df):
    """Engineer family-related features."""
    df = df.copy()
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    return df


def extract_title_from_name(df):
    """Extract title from passenger name."""
    df = df.copy()
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.')
    title_counts = df['Title'].value_counts()
    rare_titles = title_counts[title_counts < 10].index
    df['Title'] = df['Title'].replace(list(rare_titles), 'Other')
    return df


def create_age_groups(df):
    """Create age group categories."""
    df = df.copy()
    bins = [0, 12, 18, 60, 120]
    labels = ['Child', 'Teen', 'Adult', 'Senior']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    return df


def create_fare_features(df):
    """Create fare-related features."""
    df = df.copy()
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['FarePerPerson'] = df['FarePerPerson'].fillna(0)
    df['FareCategory'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'], duplicates='drop')
    return df


def encode_categorical_features(df):
    """One-hot encode categorical features."""
    df = df.copy()
    
    df['Sex_male'] = (df['Sex'] == 'male').astype(int)
    df['Sex_female'] = (df['Sex'] == 'female').astype(int)
    
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=False)
    df = pd.concat([df, embarked_dummies], axis=1)
    
    title_dummies = pd.get_dummies(df['Title'], prefix='Title', drop_first=False)
    df = pd.concat([df, title_dummies], axis=1)
    
    pclass_dummies = pd.get_dummies(df['Pclass'], prefix='Pclass', drop_first=False)
    df = pd.concat([df, pclass_dummies], axis=1)
    
    return df


def create_interaction_features(df):
    """Create interaction features."""
    df = df.copy()
    
    df['Age_Title_Mr'] = df['Age'] * (df['Title'] == 'Mr').astype(int)
    df['Age_Title_Mrs'] = df['Age'] * (df['Title'] == 'Mrs').astype(int)
    df['Age_Title_Miss'] = df['Age'] * (df['Title'] == 'Miss').astype(int)
    
    df['Fare_x_Pclass'] = df['Fare'] * df['Pclass']
    df['FamilySize_x_Pclass'] = df['FamilySize'] * df['Pclass']
    
    return df


def apply_log_transformation(df):
    """Apply log transformation to skewed features."""
    df = df.copy()
    df['LogFare'] = np.log1p(df['Fare'])
    df['LogAge'] = np.log1p(df['Age'])
    return df


def engineer_features(input_path, output_path):
    """Main feature engineering pipeline."""
    
    df = pd.read_csv(input_path)
    
    print("\n" + "=" * 70)
    print("TITANIC FEATURE ENGINEERING PIPELINE")
    print("=" * 70)
    
    print("\n1. INITIAL FEATURE COUNT:")
    print(f"   Features: {len(df.columns)}")
    
    print("\n2. CREATING FAMILY FEATURES:")
    df = engineer_family_features(df)
    print("   ✓ FamilySize created (SibSp + Parch + 1)")
    print("   ✓ IsAlone created (1 if FamilySize == 1)")
    
    print("\n3. EXTRACTING TITLE FROM NAME:")
    df = extract_title_from_name(df)
    print("   ✓ Title extracted from Name column")
    
    print("\n4. CREATING AGE GROUPS:")
    df = create_age_groups(df)
    print("   ✓ AgeGroup created (Child, Teen, Adult, Senior)")
    
    print("\n5. CREATING FARE FEATURES:")
    df = create_fare_features(df)
    print("   ✓ FarePerPerson created (Fare / FamilySize)")
    print("   ✓ FareCategory created (quartiles)")
    
    print("\n6. ENCODING CATEGORICAL FEATURES:")
    df = encode_categorical_features(df)
    print("   ✓ Sex one-hot encoded")
    print("   ✓ Embarked one-hot encoded")
    print("   ✓ Title one-hot encoded")
    print("   ✓ Pclass one-hot encoded")
    
    print("\n7. CREATING INTERACTION FEATURES:")
    df = create_interaction_features(df)
    print("   ✓ Age × Title interactions created")
    print("   ✓ Fare × Pclass interaction created")
    print("   ✓ FamilySize × Pclass interaction created")
    
    print("\n8. APPLYING LOG TRANSFORMATIONS:")
    df = apply_log_transformation(df)
    print("   ✓ LogFare created (log1p transformation)")
    print("   ✓ LogAge created (log1p transformation)")
    
    print("\n9. FINAL FEATURE SET:")
    print(f"    Total features: {len(df.columns)}")
    
    df.to_csv(output_path, index=False)
    print(f"\n✓ Engineered data saved to: {output_path}")
    print("=" * 70 + "\n")
    
    return df


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    input_file = project_root / "data" / "train_cleaned.csv"
    output_file = project_root / "data" / "train_engineered.csv"
    
    os.makedirs(project_root / "data", exist_ok=True)
    
    df_engineered = engineer_features(str(input_file), str(output_file))
