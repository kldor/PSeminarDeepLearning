"""
Train Random Forest models for landslide prediction.

This script trains separate Random Forest models for each polygon size
and evaluates feature importance across all models.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from pathlib import Path
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def load_and_prepare_data(data_path):
    """Load combined data and prepare for modeling."""
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Create target variable: landslide (MOVEMENT_C != 99) vs no landslide (MOVEMENT_C == 99)
    df['is_landslide'] = (df['MOVEMENT_C'] != 99).astype(int)
    
    print(f"\nClass distribution:")
    print(f"  Landslides: {df['is_landslide'].sum()}")
    print(f"  No landslides: {(df['is_landslide'] == 0).sum()}")
    
    return df


def get_common_features():
    """Get list of common features used across all models."""
    return [
        # Rainfall
        'Rainfall_7d_mm', 'Rainfall_14d_mm', 'Rainfall_21d_mm',
        # Soil moisture
        'SoilMoisture_0.05m', 'SoilMoisture_0.25m', 'SoilMoisture_0.7m', 'SoilMoisture_1.5m',
        # Temperature
        'Temperature_500hPa_K', 'Temperature_700hPa_K', 'Temperature_850hPa_K',
        # Ground temperature
        'GroundTemperature_K'
    ]


def get_slope_features(polygon_size):
    """Get slope features for a specific polygon size."""
    return [
        f'Slope_{polygon_size}_avg',
        f'Slope_{polygon_size}_median',
        f'Slope_{polygon_size}_min',
        f'Slope_{polygon_size}_max'
    ]


def find_common_valid_samples(df, polygon_sizes):
    """
    Find UUIDs that have valid (non-NaN) data for all polygon sizes.
    
    Returns:
        list: UUIDs that have complete data across all configurations
    """
    common_features = get_common_features()
    
    # Start with all UUIDs
    valid_uuids = set(df['UUID'])
    
    print("Finding samples with valid data across all polygon sizes...")
    
    # For each polygon size, remove UUIDs with missing data
    for size in polygon_sizes:
        slope_features = get_slope_features(size)
        all_features = common_features + slope_features
        
        # Find rows with complete data for this size
        size_valid_mask = df[all_features].notna().all(axis=1)
        size_valid_uuids = set(df.loc[size_valid_mask, 'UUID'])
        
        # Keep only UUIDs valid for this size
        valid_uuids = valid_uuids.intersection(size_valid_uuids)
        
        print(f"  {size}: {len(size_valid_uuids)} valid samples, {len(valid_uuids)} common so far")
    
    print(f"\nTotal samples valid across all configurations: {len(valid_uuids)}")
    
    return list(valid_uuids)


def prepare_features_for_size(df, polygon_size):
    """
    Prepare feature set for a specific polygon size.
    
    Returns:
        X: Features DataFrame
        y: Target series
        feature_names: List of feature names
    """
    # Get all features for this size
    common_features = get_common_features()
    slope_features = get_slope_features(polygon_size)
    all_features = common_features + slope_features
    
    # Create feature matrix
    X = df[all_features].copy()
    y = df['is_landslide'].copy()
    
    # Handle missing values - drop rows with any NaN in features
    valid_indices = X.notna().all(axis=1) & y.notna()
    X = X[valid_indices]
    y = y[valid_indices]
    
    print(f"\n  Features for {polygon_size}:")
    print(f"    Total features: {len(all_features)}")
    print(f"    Valid samples: {len(X)} (dropped {(~valid_indices).sum()} due to missing values)")
    
    return X, y, all_features


def train_model(X_train, y_train, random_state=42):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, feature_names, threshold=0.35):
    """
    Evaluate model performance.
    
    Parameters:
        threshold: Prediction threshold (default 0.35 to detect more landslides)
    
    Returns:
        dict: Evaluation metrics
    """
    # Use custom threshold instead of default 0.5
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'threshold': threshold,
        'n_test_samples': len(y_test),
        'n_landslides_test': int(y_test.sum()),
        'n_predicted_landslides': int(y_pred.sum())
    }
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    metrics['feature_importance'] = feature_importance.to_dict('records')
    
    return metrics, feature_importance


def plot_feature_importance(feature_importance, polygon_size, output_dir):
    """Create feature importance plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot top 15 features
    top_features = feature_importance.head(15)
    
    ax.barh(range(len(top_features)), top_features['importance'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top 15 Feature Importance - {polygon_size}')
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f'feature_importance_{polygon_size}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Feature importance plot saved: {plot_path}")


def plot_confusion_matrix(cm, polygon_size, output_dir):
    """Create confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {polygon_size}')
    ax.set_xticklabels(['No Landslide', 'Landslide'])
    ax.set_yticklabels(['No Landslide', 'Landslide'])
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f'confusion_matrix_{polygon_size}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Confusion matrix saved: {plot_path}")


def create_comparison_plot(all_results, output_dir):
    """Create comparison plot of model performance across polygon sizes."""
    sizes = list(all_results.keys())
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        values = [all_results[size][metric] for size in sizes]
        
        axes[idx].bar(range(len(sizes)), values, color='steelblue')
        axes[idx].set_xticks(range(len(sizes)))
        axes[idx].set_xticklabels(sizes, rotation=45, ha='right')
        axes[idx].set_ylabel(metric.replace('_', ' ').title())
        axes[idx].set_title(f'{metric.replace("_", " ").title()} by Polygon Size')
        axes[idx].set_ylim([0, 1])
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'model_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Model comparison plot saved: {plot_path}")


def create_feature_importance_comparison(all_results, output_dir):
    """Create comparison of top features across all models."""
    # Collect all features and their average importance
    feature_importance_sum = {}
    feature_counts = {}
    
    for size, results in all_results.items():
        for item in results['feature_importance']:
            feature = item['feature']
            importance = item['importance']
            
            if feature not in feature_importance_sum:
                feature_importance_sum[feature] = 0
                feature_counts[feature] = 0
            
            feature_importance_sum[feature] += importance
            feature_counts[feature] += 1
    
    # Calculate average importance
    avg_importance = {
        feature: feature_importance_sum[feature] / feature_counts[feature]
        for feature in feature_importance_sum
    }
    
    # Sort by importance
    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Plot top 20 features
    fig, ax = plt.subplots(figsize=(12, 10))
    
    top_n = 20
    features = [f[0] for f in sorted_features[:top_n]]
    importances = [f[1] for f in sorted_features[:top_n]]
    
    ax.barh(range(len(features)), importances, color='steelblue')
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Average Importance Across All Models')
    ax.set_title(f'Top {top_n} Features - Average Across All Polygon Sizes')
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'feature_importance_overall.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Overall feature importance plot saved: {plot_path}")


def main():
    """Main function to train all models."""
    print("=" * 70)
    print("RANDOM FOREST MODEL TRAINING")
    print("=" * 70)
    
    # Setup paths
    base_path = Path(__file__).parent
    data_path = base_path / "input" / "combined_data.csv"
    output_dir = base_path / "output"
    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    
    # Create output directories
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_and_prepare_data(data_path)
    
    # Define polygon sizes
    polygon_sizes = [
        "5x5km", "2x2km", "1x1km", "500x500m",
        "250x250m", "100x100m", "50x50m", "30x30m", "10x10m"
    ]
    
    # Find common valid samples across all polygon sizes
    print("\n" + "=" * 70)
    common_valid_uuids = find_common_valid_samples(df, polygon_sizes)
    
    # Create fixed test set from common valid samples
    test_size = min(60, int(len(common_valid_uuids) * 0.2))  # Use 60 or 20%, whichever is smaller
    
    # Sample test UUIDs with stratification
    df_common = df[df['UUID'].isin(common_valid_uuids)].copy()
    
    # Stratified split to maintain class balance
    from sklearn.model_selection import train_test_split as split_uuids
    train_uuids, test_uuids = split_uuids(
        df_common['UUID'].unique(),
        test_size=test_size,
        random_state=42,
        stratify=df_common.drop_duplicates('UUID')['is_landslide']
    )
    
    print(f"\nFixed test set created: {len(test_uuids)} samples")
    print(f"Training set: {len(train_uuids)} samples")
    print(f"Test set class distribution:")
    test_labels = df_common[df_common['UUID'].isin(test_uuids)]['is_landslide']
    print(f"  Landslides: {test_labels.sum()}")
    print(f"  No landslides: {(test_labels == 0).sum()}")
    print("=" * 70)
    
    all_results = {}
    
    # Train model for each polygon size
    for size in polygon_sizes:
        print("\n" + "=" * 70)
        print(f"Training model for {size}")
        print("=" * 70)
        
        # Prepare features
        X, y, feature_names = prepare_features_for_size(df, size)
        
        if len(X) < 10:
            print(f"  ⚠ Not enough samples for {size}, skipping...")
            continue
        
        # Split using fixed test set
        # Create train/test split based on fixed UUIDs
        test_mask = df['UUID'].isin(test_uuids)
        train_mask = df['UUID'].isin(train_uuids)
        
        # Get valid indices for this size (non-NaN features)
        valid_indices = X.index
        
        # Filter masks to only valid indices
        test_indices = df.index[test_mask & df.index.isin(valid_indices)]
        train_indices = df.index[train_mask & df.index.isin(valid_indices)]
        
        X_train = X.loc[train_indices]
        X_test = X.loc[test_indices]
        y_train = y.loc[train_indices]
        y_test = y.loc[test_indices]
        
        print(f"\n  Train set: {len(X_train)} samples")
        print(f"  Test set: {len(X_test)} samples (fixed across all models)")
        
        # Train model
        print("\n  Training Random Forest...")
        model = train_model(X_train, y_train)
        
        # Evaluate model
        print("  Evaluating model...")
        metrics, feature_importance = evaluate_model(model, X_test, y_test, feature_names)
        
        # Print results
        print(f"\n  Results:")
        print(f"    Accuracy:  {metrics['accuracy']:.3f}")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall:    {metrics['recall']:.3f}")
        print(f"    F1-Score:  {metrics['f1_score']:.3f}")
        
        print(f"\n  Top 5 most important features:")
        for i, row in feature_importance.head(5).iterrows():
            print(f"    {row['feature']:30s} {row['importance']:.4f}")
        
        # Save model
        model_path = models_dir / f'rf_model_{size}.pkl'
        joblib.dump(model, model_path)
        print(f"\n  ✓ Model saved: {model_path}")
        
        # Create plots
        plot_feature_importance(feature_importance, size, plots_dir)
        plot_confusion_matrix(np.array(metrics['confusion_matrix']), size, plots_dir)
        
        # Store results
        all_results[size] = metrics
    
    # Save all results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTrained {len(all_results)} models")
    print(f"Results saved to: {results_path}")
    
    # Create comparison plots
    if all_results:
        create_comparison_plot(all_results, plots_dir)
        create_feature_importance_comparison(all_results, plots_dir)
    
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 70)
    for size, results in all_results.items():
        print(f"\n{size}:")
        print(f"  Accuracy: {results['accuracy']:.3f} | Precision: {results['precision']:.3f} | "
              f"Recall: {results['recall']:.3f} | F1: {results['f1_score']:.3f}")
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
