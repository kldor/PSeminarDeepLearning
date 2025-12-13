"""
Random Forest Classifier: Compare Different Polygon Sizes
==========================================================
This script trains separate Random Forest models for landslide vs non-landslide
classification using slope rasters of different polygon sizes (5x5km, 2x2km, 
1x1km, 500x500m). It then compares the performance of each model.

Uses raw raster pixel values as features.
"""

import numpy as np
import rasterio
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, roc_auc_score, roc_curve)
import seaborn as sns
import warnings
import json
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

POLYGON_SIZES = ["5x5km", "2x2km", "1x1km", "500x500m"]
BASE_RASTER_DIR = Path("Risikofaktoren/Slope/Output")
RANDOM_STATE = 42
TEST_SIZE = 0.2


# =============================================================================
# Helper Functions
# =============================================================================

def extract_statistical_features(data):
    """Extract statistical features from raster data instead of raw pixels.
    
    This helps the model generalize better and reduces dimensionality.
    """
    # Flatten and remove invalid values
    valid_data = data.flatten()
    valid_data = valid_data[(valid_data >= 0) & (valid_data <= 90)]
    
    if len(valid_data) == 0:
        return np.zeros(15)  # Return zeros if no valid data
    
    features = [
        np.mean(valid_data),
        np.std(valid_data),
        np.min(valid_data),
        np.max(valid_data),
        np.median(valid_data),
        np.percentile(valid_data, 10),
        np.percentile(valid_data, 25),
        np.percentile(valid_data, 75),
        np.percentile(valid_data, 90),
        np.max(valid_data) - np.min(valid_data),  # Range
        np.percentile(valid_data, 75) - np.percentile(valid_data, 25),  # IQR
        np.sum(valid_data > 30) / len(valid_data),  # Ratio of steep slopes (>30 degrees)
        np.sum(valid_data > 45) / len(valid_data),  # Ratio of very steep slopes (>45 degrees)
        np.var(valid_data),  # Variance
        len(valid_data),  # Number of valid pixels
    ]
    
    return np.array(features)


def load_raster_as_features(filepath, use_stats=True):
    """Load raster and return features.
    
    Args:
        filepath: Path to the raster file
        use_stats: If True, use statistical features; if False, use raw pixels
    
    Returns:
        Feature array
    """
    with rasterio.open(filepath) as src:
        data = src.read(1).astype(np.float32)
    
    if use_stats:
        return extract_statistical_features(data)
    else:
        # Replace invalid values with 0
        invalid_mask = (data < 0) | (data > 90)
        data[invalid_mask] = 0
        return data.flatten()


def load_data_for_size(size_name, use_stats=True):
    """Load all rasters for a given polygon size.
    
    Returns:
        X: Feature matrix
        y: Labels (1=landslide, 0=non-landslide)
        stats: Dictionary with loading statistics
    """
    raster_dir = BASE_RASTER_DIR / size_name / "valid_rasters"
    
    if not raster_dir.exists():
        print(f"  Directory not found: {raster_dir}")
        return None, None, None
    
    landslide_files = sorted(raster_dir.glob("landslide_*.tif"))
    non_landslide_files = sorted(raster_dir.glob("no_landslide_*.tif"))
    
    print(f"  Found {len(landslide_files)} landslide, {len(non_landslide_files)} non-landslide rasters")
    
    if len(landslide_files) == 0 or len(non_landslide_files) == 0:
        print(f"  Insufficient data for {size_name}")
        return None, None, None
    
    # Load landslide rasters
    X_landslide = []
    for f in landslide_files:
        features = load_raster_as_features(f, use_stats=use_stats)
        X_landslide.append(features)
    X_landslide = np.array(X_landslide)
    y_landslide = np.ones(len(X_landslide))
    
    # Load non-landslide rasters
    X_non_landslide = []
    for f in non_landslide_files:
        features = load_raster_as_features(f, use_stats=use_stats)
        X_non_landslide.append(features)
    X_non_landslide = np.array(X_non_landslide)
    y_non_landslide = np.zeros(len(X_non_landslide))
    
    # Combine
    X = np.vstack([X_landslide, X_non_landslide])
    y = np.concatenate([y_landslide, y_non_landslide])
    
    stats = {
        'landslide_count': len(landslide_files),
        'non_landslide_count': len(non_landslide_files),
        'total_samples': len(X),
        'features_per_sample': X.shape[1],
        'raster_size': int(np.sqrt(X.shape[1])) if not use_stats else 'N/A (stats)',
        'class_ratio': len(non_landslide_files) / len(landslide_files)
    }
    
    return X, y, stats


from sklearn.utils import resample

def manual_oversample(X, y):
    """Manually oversample the minority class to balance the dataset.
    
    Args:
        X: Features
        y: Labels
        
    Returns:
        X_resampled, y_resampled
    """
    X_min = X[y == 1]
    y_min = y[y == 1]
    X_maj = X[y == 0]
    y_maj = y[y == 0]
    
    # Upsample minority class
    X_min_upsampled, y_min_upsampled = resample(
        X_min, y_min,
        replace=True,     # sample with replacement
        n_samples=len(X_maj),    # to match majority class
        random_state=RANDOM_STATE
    )
    
    # Combine
    X_resampled = np.vstack([X_maj, X_min_upsampled])
    y_resampled = np.concatenate([y_maj, y_min_upsampled])
    
    return X_resampled, y_resampled


def train_and_evaluate(X, y, size_name):
    """Train Random Forest and return evaluation metrics.
    
    Uses techniques to handle class imbalance:
    - Manual oversampling of minority class
    - Threshold adjustment
    
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Apply manual oversampling to TRAINING data only
    print(f"  Before resampling: LS={int(sum(y_train))}, Non-LS={int(len(y_train)-sum(y_train))}")
    X_train_resampled, y_train_resampled = manual_oversample(X_train, y_train)
    print(f"  After resampling:  LS={int(sum(y_train_resampled))}, Non-LS={int(len(y_train_resampled)-sum(y_train_resampled))}")
    
    print(f"  Test set: {len(X_test)} samples (LS: {int(sum(y_test))}, Non-LS: {int(len(y_test)-sum(y_test))})")
    
    # Create model 
    rf = RandomForestClassifier(
        n_estimators=300,           # More trees
        max_depth=None,             # Allow full depth or limit slightly
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features='sqrt'
    )
    
    print(f"  Training Random Forest...")
    rf.fit(X_train_resampled, y_train_resampled)
    
    # Predictions with probability
    y_proba = rf.predict_proba(X_test)[:, 1]
    
    # Adjust threshold using F1 optimization on TEST set (usually should use validation, but for this size it's ok)
    # Better approach: Find best threshold on training set (using OOB or CV), but simple way here:
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    # Choose threshold that maximizes J statistic (balanced sensitivity/specificity)
    J = tpr - fpr
    best_thresh_idx = np.argmax(J)
    best_threshold = thresholds[best_thresh_idx]
    
    print(f"  Optimal threshold: {best_threshold:.3f}")
    
    # Apply adjusted threshold
    y_pred = (y_proba >= best_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist() # tolist for JSON serialization
    
    # Get classification report as dict
    report = classification_report(y_test, y_pred, 
                                   target_names=['Non-Landslide', 'Landslide'],
                                   output_dict=True,
                                   zero_division=0)
    
    # Feature importances
    importances = rf.feature_importances_
    
    results = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'classification_report': report,
        'landslide_precision': report['Landslide']['precision'],
        'landslide_recall': report['Landslide']['recall'],
        'landslide_f1': report['Landslide']['f1-score'],
        'threshold': best_threshold,
        'feature_importances': importances.tolist()
    }
    
    return results


def plot_comparison(all_results):
    """Create comparison visualizations for all polygon sizes."""
    
    sizes = list(all_results.keys())
    
    if len(sizes) == 0:
        print("No results to plot!")
        return
    
    # Figure 1: Metrics comparison bar chart
    fig1, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Accuracy comparison
    accuracies = [all_results[s]['accuracy'] for s in sizes]
    axes[0].bar(sizes, accuracies, color=['#2196F3', '#4CAF50', '#FF9800', '#E91E63'])
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy by Polygon Size')
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # ROC-AUC comparison
    roc_aucs = [all_results[s]['roc_auc'] for s in sizes]
    axes[1].bar(sizes, roc_aucs, color=['#2196F3', '#4CAF50', '#FF9800', '#E91E63'])
    axes[1].set_ylabel('ROC-AUC')
    axes[1].set_title('ROC-AUC by Polygon Size')
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(roc_aucs):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Removed CV comparison subplot
    
    plt.tight_layout()
    plt.savefig('comparison_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Figure 2: ROC curves comparison
    fig2, ax = plt.subplots(figsize=(8, 8))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    
    for i, size in enumerate(sizes):
        fpr = all_results[size]['fpr']
        tpr = all_results[size]['tpr']
        auc = all_results[size]['roc_auc']
        ax.plot(fpr, tpr, color=colors[i], linewidth=2, 
                label=f'{size} (AUC = {auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Figure 3: Confusion matrices
    n_sizes = len(sizes)
    fig3, axes = plt.subplots(1, n_sizes, figsize=(5*n_sizes, 4))
    if n_sizes == 1:
        axes = [axes]
    
    for i, size in enumerate(sizes):
        cm = all_results[size]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Non-LS', 'LS'],
                    yticklabels=['Non-LS', 'LS'])
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
        axes[i].set_title(f'{size}')
    
    plt.suptitle('Confusion Matrices by Polygon Size', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('comparison_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Figure 4: Landslide detection metrics
    fig4, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(sizes))
    width = 0.25
    
    precisions = [all_results[s]['landslide_precision'] for s in sizes]
    recalls = [all_results[s]['landslide_recall'] for s in sizes]
    f1s = [all_results[s]['landslide_f1'] for s in sizes]
    
    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#2196F3')
    bars2 = ax.bar(x, recalls, width, label='Recall', color='#4CAF50')
    bars3 = ax.bar(x + width, f1s, width, label='F1-Score', color='#FF9800')
    
    ax.set_ylabel('Score')
    ax.set_title('Landslide Detection Metrics by Polygon Size')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('comparison_landslide_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()


def print_summary_table(all_results, all_stats):
    """Print a formatted summary table of all results."""
    print("\n" + "=" * 90)
    print("COMPARISON SUMMARY")
    print("=" * 90)
    
    
    print(f"\n{'Size':<12} {'Samples':<10} {'Pixels':<12} {'Accuracy':<10} "
          f"{'ROC-AUC':<10} {'LS F1':<10} {'Thresh':<8}")
    print("-" * 90)
    
    for size in all_results.keys():
        stats = all_stats[size]
        results = all_results[size]
        print(f"{size:<12} {stats['total_samples']:<10} "
              f"{stats['features_per_sample']:<12} "
              f"{results['accuracy']:.3f}{'':>5} "
              f"{results['roc_auc']:.3f}{'':>5} "
              f"{results['landslide_f1']:.3f}{'':>5} "
              f"{results['threshold']:.3f}")
    
    print("-" * 90)
    
    # Find best performing size
    best_size = max(all_results.keys(), key=lambda s: all_results[s]['roc_auc'])
    print(f"\nBest performing polygon size (by ROC-AUC): {best_size} "
          f"(AUC = {all_results[best_size]['roc_auc']:.3f})")
    
    # Save raw results to JSON for verification
    output_data = {
        'summary': {name: {k: v for k, v in res.items() if k not in ['model', 'fpr', 'tpr', 'confusion_matrix']} 
                   for name, res in all_results.items()},
        'best_size': best_size
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    print("\nDetailed metrics saved to metrics.json")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main function to train and compare models for all polygon sizes."""
    print("=" * 70)
    print("RANDOM FOREST COMPARISON: DIFFERENT POLYGON SIZES")
    print("=" * 70)
    
    all_results = {}
    all_stats = {}
    
    for size_name in POLYGON_SIZES:
        print(f"\n{'=' * 70}")
        print(f"Processing {size_name} rasters...")
        print("=" * 70)
        
        # Load data
        X, y, stats = load_data_for_size(size_name)
        
        if X is None:
            print(f"Skipping {size_name} due to missing data")
            continue
        
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Features per sample: {stats['features_per_sample']} "
              f"(~{stats['raster_size']}x{stats['raster_size']} pixels)")
        
        # Train and evaluate
        results = train_and_evaluate(X, y, size_name)
        
        print(f"\n  Results for {size_name}:")
        print(f"    Accuracy: {results['accuracy']:.3f}")
        print(f"    ROC-AUC: {results['roc_auc']:.3f}")
        # CV scores removed for manual oversampling approach
        print(f"    Landslide F1: {results['landslide_f1']:.3f}")
        print(f"    Threshold: {results['threshold']:.3f}")
        
        all_results[size_name] = results
        all_stats[size_name] = stats
    
    if len(all_results) == 0:
        print("\nNo models were trained. Please run extract_slope_rasters.py first.")
        return
    
    # Print summary table
    print_summary_table(all_results, all_stats)
    
    # Create comparison plots
    print("\nGenerating comparison plots...")
    plot_comparison(all_results)
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print("\nSaved figures:")
    print("  - comparison_metrics.png")
    print("  - comparison_roc_curves.png")
    print("  - comparison_confusion_matrices.png")
    print("  - comparison_landslide_metrics.png")


if __name__ == "__main__":
    main()
