"""
Feature Selection Module for Titanic Dataset
=============================================
Selects best features using correlation analysis and tree models.

Author: Titanic Assignment
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import os


def analyze_correlations(df, target_col='Survived'):
    """
    Analyze feature correlations with target variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Engineered dataframe
    target_col : str
        Name of target column
    
    Returns
    -------
    pd.Series
        Correlation values sorted by absolute magnitude
    """
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlations
    corr = numerical_df.corr()[target_col].sort_values(ascending=False)
    
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)
    print("\nTop 15 Features (by correlation with Survived):")
    print("\n" + "-" * 50)
    for i, (feature, value) in enumerate(corr.head(15).items(), 1):
        if feature != target_col:
            print(f"{i:2}. {feature:30} : {value:7.4f}")
    
    return corr


def get_feature_importance(df, target_col='Survived'):
    """
    Get feature importance using Random Forest classifier.
    
    Parameters
    ----------
    df : pd.DataFrame
        Engineered dataframe with target column
    target_col : str
        Name of target column
    
    Returns
    -------
    pd.DataFrame
        Feature importance scores sorted by importance
    """
    # Prepare data
    X = df.drop(columns=[target_col, 'PassengerId', 'Name', 'Ticket', 'Sex', 'Embarked', 'Title', 'AgeGroup', 'FareCategory'], errors='ignore')
    y = df[target_col]
    
    # Select only numerical features
    X = X.select_dtypes(include=[np.number])
    
    # Handle any remaining missing values
    X = X.fillna(X.mean())
    
    # Train Random Forest
    print("\n" + "=" * 70)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 70)
    print("\nTraining Random Forest on engineered features...")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 20 Most Important Features:")
    print("-" * 50)
    for i, row in importance_df.head(20).iterrows():
        bar_length = int(row['Importance'] * 100)
        bar = '█' * bar_length + '░' * (40 - bar_length)
        print(f"{row['Feature']:30} | {bar} {row['Importance']:.4f}")
    
    return importance_df


def identify_redundant_features(df, threshold=0.95):
    """
    Identify highly correlated (redundant) features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with numerical features
    threshold : float
        Correlation threshold (0-1)
    
    Returns
    -------
    list
        List of redundant features to remove
    """
    print("\n" + "=" * 70)
    print("MULTICOLLINEARITY ANALYSIS")
    print("=" * 70)
    print(f"\nSearching for features with correlation > {threshold}...")
    
    # Get numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    corr_matrix = numerical_df.corr().abs()
    
    # Find pairs of highly correlated features
    redundant = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                redundant.append({
                    'Feature1': corr_matrix.columns[i],
                    'Feature2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
    
    if redundant:
        redundant_df = pd.DataFrame(redundant)
        print(f"\nFound {len(redundant_df)} highly correlated pairs:")
        print("\n" + "-" * 50)
        for idx, row in redundant_df.iterrows():
            print(f"{row['Feature1']:30} ↔ {row['Feature2']:30} : {row['Correlation']:.4f}")
    else:
        print("\n✓ No highly correlated feature pairs found")
        redundant_df = pd.DataFrame()
    
    return redundant_df


def select_best_features(df, target_col='Survived', n_features=20):
    """
    Select best features combining multiple criteria.
    
    Criteria:
    1. Feature importance from Random Forest
    2. Correlation with target
    3. Low multicollinearity
    
    Parameters
    ----------
    df : pd.DataFrame
        Engineered dataframe
    target_col : str
        Target column name
    n_features : int
        Number of features to select
    
    Returns
    -------
    list, dict
        List of selected features and selection summary
    """
    print("\n" + "=" * 70)
    print("FEATURE SELECTION SUMMARY")
    print("=" * 70)
    
    # Get correlations
    corr = analyze_correlations(df, target_col)
    
    # Get feature importance
    importance_df = get_feature_importance(df, target_col)
    
    # Find redundant features
    redundant_df = identify_redundant_features(df)
    
    # Select features
    # Strategy: Combine importance and correlation
    selected_features = set()
    
    # Add top features by importance
    top_importance = importance_df.head(n_features)['Feature'].tolist()
    selected_features.update(top_importance)
    
    # Add features with high correlation (but not redundant)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in corr.index and col != target_col:
            if abs(corr[col]) > 0.1:  # Meaningful correlation
                selected_features.add(col)
    
    # Remove redundant features (keep one from each correlated pair)
    if not redundant_df.empty:
        for idx, row in redundant_df.iterrows():
            if row['Feature1'] in selected_features and row['Feature2'] in selected_features:
                # Keep the one with higher importance
                imp1 = importance_df[importance_df['Feature'] == row['Feature1']]['Importance'].values
                imp2 = importance_df[importance_df['Feature'] == row['Feature2']]['Importance'].values
                
                if len(imp1) > 0 and len(imp2) > 0:
                    if imp1[0] > imp2[0]:
                        selected_features.discard(row['Feature2'])
                    else:
                        selected_features.discard(row['Feature1'])
    
    # Convert to list and sort by importance
    selected_features = sorted(list(selected_features), 
                               key=lambda x: importance_df[importance_df['Feature'] == x]['Importance'].values[0] if len(importance_df[importance_df['Feature'] == x]) > 0 else 0,
                               reverse=True)
    
    print("\n" + "=" * 70)
    print(f"FINAL SELECTED FEATURES ({len(selected_features)} features)")
    print("=" * 70)
    print("\n" + "-" * 50)
    for i, feature in enumerate(selected_features, 1):
        imp_val = importance_df[importance_df['Feature'] == feature]['Importance'].values
        corr_val = corr.get(feature, 0)
        if len(imp_val) > 0:
            print(f"{i:2}. {feature:30} | Imp: {imp_val[0]:.4f} | Corr: {corr_val:.4f}")
    
    summary = {
        'total_features_selected': len(selected_features),
        'selected_features': selected_features,
        'redundant_pairs_found': len(redundant_df),
        'criteria_used': ['Feature Importance', 'Correlation Threshold', 'Multicast Tolerance']
    }
    
    return selected_features, summary


def select_features(input_path, output_path, n_features=20):
    """
    Main feature selection pipeline.
    
    Parameters
    ----------
    input_path : str
        Path to engineered dataset
    output_path : str
        Path to save selected features dataset
    n_features : int
        Number of features to select
    
    Returns
    -------
    list
        Selected feature names
    """
    # Load data
    df = pd.read_csv(input_path)
    
    print("\n" + "=" * 70)
    print("TITANIC FEATURE SELECTION PIPELINE")
    print("=" * 70)
    print(f"\nInitial dataset shape: {df.shape}")
    print(f"Features: {len(df.columns)}")
    
    # Select features
    selected_features, summary = select_best_features(df, target_col='Survived', n_features=n_features)
    
    # Create subset with selected features (+ target and ID)
    essential_cols = ['PassengerId', 'Survived']
    output_cols = essential_cols + selected_features
    df_selected = df[[col for col in output_cols if col in df.columns]]
    
    # Save selected features
    df_selected.to_csv(output_path, index=False)
    
    print(f"\n✓ Selected features dataset saved to: {output_path}")
    print(f"  Final shape: {df_selected.shape}")
    print("=" * 70 + "\n")
    
    return selected_features


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    input_file = project_root / "data" / "train_engineered.csv"
    output_file = project_root / "data" / "train_final.csv"
    
    os.makedirs(project_root / "data", exist_ok=True)
    
    selected_features = select_features(str(input_file), str(output_file), n_features=20)
