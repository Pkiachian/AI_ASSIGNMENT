"""
Data Cleaning Module for Titanic Dataset
=========================================
Handles missing values, outliers, and data consistency.

Author: Titanic Assignment
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os


def load_data(filepath):
    """Load the Titanic dataset."""
    return pd.read_csv(filepath)


def identify_missing_values(df):
    """Identify and report missing values in all columns."""
    missing_info = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
    })
    return missing_info[missing_info['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)


def handle_missing_values(df):
    """
    Handle missing values with appropriate strategies.
    
    Strategy:
    - Age: Impute with median by Pclass (considers passenger class differences)
    - Embarked: Impute with mode (most common embarkation port)
    - Fare: Impute with median by Pclass
    - Cabin: Create binary indicator (most records are missing)
    """
    df = df.copy()
    
    # Age: Impute median by Pclass (more nuanced than overall median)
    df['Age'] = df.groupby('Pclass')['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Embarked: Impute with mode (most common port)
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Fare: Impute with median by Pclass for any missing values
    df['Fare'] = df.groupby('Pclass')['Fare'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Create indicator for missing cabin (useful feature - most records missing)
    df['HasCabin'] = (~df['Cabin'].isnull()).astype(int)
    df.drop('Cabin', axis=1, inplace=True)
    
    return df


def detect_and_handle_outliers(df):
    """
    Detect outliers in numerical columns (Fare, Age).
    Uses IQR method and caps extreme values rather than removing.
    
    Rationale for capping vs removing:
    - Removing data loses information
    - Extreme values may be real edge cases
    - Capping preserves structure while limiting influence on models
    """
    df = df.copy()
    
    # Handle Fare outliers using IQR method
    Q1_fare = df['Fare'].quantile(0.25)
    Q3_fare = df['Fare'].quantile(0.75)
    IQR_fare = Q3_fare - Q1_fare
    upper_bound_fare = Q3_fare + 1.5 * IQR_fare
    lower_bound_fare = Q1_fare - 1.5 * IQR_fare
    
    fare_before = len(df[(df['Fare'] < lower_bound_fare) | (df['Fare'] > upper_bound_fare)])
    df['Fare'] = df['Fare'].clip(lower=max(0, lower_bound_fare), upper=upper_bound_fare)
    
    # Age: Naturally bounded (0-120), minimal handling needed
    age_before = len(df[(df['Age'] < 0) | (df['Age'] > 120)])
    df['Age'] = df['Age'].clip(lower=0, upper=120)
    
    print(f"   Fare outliers handled: {fare_before} records")
    print(f"   Age outliers handled: {age_before} records")
    
    return df


def ensure_data_consistency(df):
    """
    Fix inconsistencies in the data.
    
    Actions:
    - Standardize Sex values to lowercase
    - Verify numeric types for ID columns
    - Check and remove duplicates
    """
    df = df.copy()
    
    # Standardize Sex values
    df['Sex'] = df['Sex'].str.lower().str.strip()
    
    # Ensure correct data types
    if 'Pclass' in df.columns:
        df['Pclass'] = df['Pclass'].astype('int64')
    if 'Survived' in df.columns:
        df['Survived'] = df['Survived'].astype('int64')
    
    # Check and remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    
    return df, duplicates_removed


def clean_data(input_path, output_path):
    """
    Main function to perform all data cleaning steps.
    
    Parameters
    ----------
    input_path : str
        Path to the raw train.csv file
    output_path : str
        Path to save the cleaned dataset
    
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe
    dict
        Dictionary with cleaning statistics
    """
    # Load data
    df = load_data(input_path)
    
    print("\n" + "=" * 70)
    print("TITANIC DATA CLEANING PIPELINE")
    print("=" * 70)
    
    print("\n1. INITIAL DATA INSPECTION:")
    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"   Columns: {', '.join(df.columns)}")
    
    print("\n2. MISSING VALUES ANALYSIS (BEFORE):")
    missing_before = identify_missing_values(df)
    if len(missing_before) > 0:
        print(missing_before.to_string())
    else:
        print("   No missing values found")
    
    # Apply cleaning steps
    print("\n3. HANDLING MISSING VALUES:")
    df = handle_missing_values(df)
    print("   ✓ Age imputed with median (stratified by Pclass)")
    print("   ✓ Embarked imputed with mode")
    print("   ✓ Fare imputed with median (stratified by Pclass)")
    print("   ✓ Cabin replaced with 'HasCabin' binary indicator")
    
    print("\n4. DETECTING & HANDLING OUTLIERS:")
    df = detect_and_handle_outliers(df)
    print("   ✓ Outliers capped using IQR method (1.5 × IQR boundaries)")
    
    print("\n5. ENSURING DATA CONSISTENCY:")
    df, duplicates = ensure_data_consistency(df)
    print("   ✓ Sex values standardized to lowercase")
    print(f"   ✓ Duplicates removed: {duplicates}")
    
    print("\n6. MISSING VALUES CHECK (AFTER):")
    missing_after = identify_missing_values(df)
    if len(missing_after) > 0:
        print(missing_after.to_string())
    else:
        print("   ✓ No missing values remaining!")
    
    print("\n7. FINAL DATASET SHAPE:")
    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"   Data types:\n{df.dtypes}")
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"\n✓ Cleaned data saved to: {output_path}")
    print("=" * 70 + "\n")
    
    return df, {
        'initial_rows': len(df),
        'duplicates_removed': duplicates,
        'missing_values_handled': True,
        'columns': list(df.columns)
    }


if __name__ == "__main__":
    # Determine file paths dynamically
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    input_file = project_root / "data" / "train.csv"
    output_file = project_root / "data" / "train_cleaned.csv"
    
    # Ensure data directory exists
    os.makedirs(project_root / "data", exist_ok=True)
    
    # Run cleaning
    df_cleaned, stats = clean_data(str(input_file), str(output_file))
