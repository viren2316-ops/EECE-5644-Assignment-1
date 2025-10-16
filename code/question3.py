#!/usr/bin/env python3
"""
EECE5644 Assignment 1 - Question 3
Gaussian Classification on Real Datasets
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import os
import urllib.request
import zipfile
import warnings
warnings.filterwarnings('ignore')

os.makedirs('data/real_datasets', exist_ok=True)
os.makedirs('figures', exist_ok=True)
np.random.seed(42)

# DATA LOADING

def load_wine_dataset():
    """Load Wine Quality dataset (4,898 samples, 11 features)."""
    print("Loading Wine Quality Dataset...")
    
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
    filepath = 'data/real_datasets/winequality-white.csv'
    
    if not os.path.exists(filepath):
        print("Downloading wine data...")
        urllib.request.urlretrieve(url, filepath)
    
    # Load data
    wine_data = pd.read_csv(filepath, sep=';')
    X = wine_data.drop('quality', axis=1).values
    y = wine_data['quality'].values
    
    print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
    print(f"  Quality scores: {np.unique(y)}")
    
    return X, y

def load_har_dataset():
    """Load HAR dataset (10,299 samples, 561 features, 6 activities)."""
    print("\nLoading Human Activity Recognition Dataset...")
    
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
    zip_path = 'data/real_datasets/HAR_Dataset.zip'
    extract_path = 'data/real_datasets/'
    
    # Download if needed
    if not os.path.exists(f'{extract_path}/UCI HAR Dataset'):
        if not os.path.exists(zip_path):
            print("Downloading HAR data (60MB)...")
            urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    
    # Load data
    base_path = f'{extract_path}/UCI HAR Dataset/'
    X_train = np.loadtxt(f'{base_path}/train/X_train.txt')
    y_train = np.loadtxt(f'{base_path}/train/y_train.txt')
    X_test = np.loadtxt(f'{base_path}/test/X_test.txt')
    y_test = np.loadtxt(f'{base_path}/test/y_test.txt')
    
    # Combine train and test
    X = np.vstack([X_train, X_test])
    y = np.hstack([y_train, y_test]).astype(int) - 1  # 0-5 instead of 1-6
    
    activity_names = ['Walking', 'W-Upstairs', 'W-Downstairs', 
                     'Sitting', 'Standing', 'Laying']
    
    print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
    print(f"  Activities: {activity_names}")
    
    return X, y, activity_names

# GAUSSIAN CLASSIFIER

class GaussianClassifier:
    """Gaussian classifier with regularization."""
    
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.params = {}
        
    def fit(self, X, y):
        """Estimate Gaussian parameters with regularization."""
        self.classes = np.unique(y)
        n_features = X.shape[1]
        
        for c in self.classes:
            X_c = X[y == c]
            
            # Estimate parameters
            mean = np.mean(X_c, axis=0)
            cov = np.cov(X_c.T) if len(X_c) > 1 else np.eye(n_features)
            
            # Regularization: λ = α * trace(C) / rank(C)
            eigenvalues = np.linalg.eigvalsh(cov)
            rank = np.sum(eigenvalues > 1e-10)
            lambda_reg = self.alpha * np.trace(cov) / (rank if rank > 0 else n_features)
            
            # Apply regularization
            cov_reg = cov + lambda_reg * np.eye(n_features)
            
            self.params[c] = {
                'mean': mean,
                'cov': cov_reg,
                'prior': len(X_c) / len(X)
            }
    
    def predict(self, X):
        """MAP classification."""
        n_samples = X.shape[0]
        log_posteriors = np.zeros((n_samples, len(self.classes)))
        
        for i, c in enumerate(self.classes):
            p = self.params[c]
            try:
                rv = multivariate_normal(p['mean'], p['cov'], allow_singular=True)
                log_posteriors[:, i] = rv.logpdf(X) + np.log(p['prior'] + 1e-10)
            except:
                log_posteriors[:, i] = -1e10
        
        return self.classes[np.argmax(log_posteriors, axis=1)]

# EVALUATION AND VISUALIZATION

def evaluate_dataset(X, y, dataset_name, class_names=None):
    """Train and evaluate Gaussian classifier."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {dataset_name}")
    print('='*60)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train classifier
    clf = GaussianClassifier(alpha=0.01)
    clf.fit(X_scaled, y)
    
    # Predict
    predictions = clf.predict(X_scaled)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y, predictions)
    conf_matrix_norm = conf_matrix.astype(float)
    for j in range(len(conf_matrix)):
        col_sum = np.sum(conf_matrix[:, j])
        if col_sum > 0:
            conf_matrix_norm[:, j] /= col_sum
    
    # Results
    accuracy = np.mean(predictions == y)
    error_rate = 1 - accuracy
    
    print(f"Error rate: {error_rate:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print confusion matrix
    print("\nConfusion Matrix (normalized):")
    classes = np.unique(y)
    if class_names:
        labels = class_names if len(class_names) == len(classes) else [str(c) for c in classes]
    else:
        labels = [str(c) for c in classes]
    
    # Print header
    print("     ", end="")
    for label in labels[:6]:  # Limit display width
        print(f"{label[:6]:>7}", end="")
    print()
    
    # Print matrix
    for i in range(min(len(classes), 6)):
        print(f"{labels[i][:5]:5}", end="")
        for j in range(min(len(classes), 6)):
            print(f"{conf_matrix_norm[i, j]:7.3f}", end="")
        print()
    
    return predictions, conf_matrix, error_rate

def visualize_pca(X, y, predictions, dataset_name):
    """PCA visualization of results."""
    print(f"\nCreating PCA visualization for {dataset_name}...")
    
    # Standardize and apply PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_components = 3 if dataset_name == "HAR" else 2
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Print variance explained
    total_var = np.sum(pca.explained_variance_ratio_)
    print(f"  Variance explained by {n_components} PCs: {total_var:.3f}")
    
    # Plot
    fig = plt.figure(figsize=(12, 5))
    
    if n_components == 2:
        # 2D plot for Wine
        ax1 = fig.add_subplot(121)
        scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', 
                            alpha=0.5, s=10)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax1.set_title(f'{dataset_name} - True Labels')
        plt.colorbar(scatter, ax=ax1)
        
        ax2 = fig.add_subplot(122)
        correct = predictions == y
        ax2.scatter(X_pca[correct, 0], X_pca[correct, 1], 
                   c='green', alpha=0.5, s=10, label='Correct')
        ax2.scatter(X_pca[~correct, 0], X_pca[~correct, 1], 
                   c='red', alpha=0.5, s=10, label='Incorrect')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax2.set_title(f'{dataset_name} - Predictions')
        ax2.legend()
    else:
        # 3D plot for HAR
        ax1 = fig.add_subplot(121, projection='3d')
        colors = plt.cm.tab10(np.linspace(0, 0.8, 6))
        for i in range(6):
            mask = y == i
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                       c=[colors[i]], alpha=0.6, s=5)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax1.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
        ax1.set_title(f'{dataset_name} - True Labels')
        
        ax2 = fig.add_subplot(122, projection='3d')
        correct = predictions == y
        ax2.scatter(X_pca[correct, 0], X_pca[correct, 1], X_pca[correct, 2],
                   c='green', alpha=0.6, s=5, label='Correct')
        ax2.scatter(X_pca[~correct, 0], X_pca[~correct, 1], X_pca[~correct, 2],
                   c='red', alpha=0.6, s=5, label='Incorrect')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax2.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
        ax2.set_title(f'{dataset_name} - Predictions')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'figures/q3_{dataset_name.lower()}_pca.png', dpi=150)
    plt.show()

def plot_confusion_matrix(conf_matrix, labels, title, filename):
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    
    # Normalize
    conf_norm = conf_matrix.astype(float)
    for j in range(len(labels)):
        if np.sum(conf_matrix[:, j]) > 0:
            conf_norm[:, j] /= np.sum(conf_matrix[:, j])
    
    plt.imshow(conf_norm, cmap='YlOrRd', vmin=0, vmax=1)
    plt.colorbar()
    
    # Labels
    plt.xticks(range(len(labels)), labels, rotation=45 if len(labels) > 6 else 0)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.title(title)
    
    # Add text
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f'{conf_norm[i,j]:.2f}', ha='center', va='center',
                    color='white' if conf_norm[i,j] > 0.5 else 'black')
    
    plt.tight_layout()
    plt.savefig(f'figures/{filename}', dpi=150)
    plt.show()

# MAIN

def main():
    print("="*60)
    print("EECE5644 Assignment 1 - Question 3")
    print("Gaussian Classification on Real Datasets")
    print("="*60)
    
    # Wine Quality Dataset
    X_wine, y_wine = load_wine_dataset()
    pred_wine, conf_wine, error_wine = evaluate_dataset(
        X_wine, y_wine, "Wine Quality"
    )
    
    quality_labels = [str(q) for q in sorted(np.unique(y_wine))]
    plot_confusion_matrix(conf_wine, quality_labels, 
                         "Wine Quality - Confusion Matrix", 
                         "q3_wine_confusion.png")
    
    visualize_pca(X_wine, y_wine, pred_wine, "Wine")
    
    # HAR Dataset
    X_har, y_har, activity_names = load_har_dataset()
    pred_har, conf_har, error_har = evaluate_dataset(
        X_har, y_har, "HAR", activity_names
    )
    
    activity_labels = ['Walk', 'W-Up', 'W-Dn', 'Sit', 'Stand', 'Lay']
    plot_confusion_matrix(conf_har, activity_labels,
                         "HAR - Confusion Matrix",
                         "q3_har_confusion.png")
    
    visualize_pca(X_har, y_har, pred_har, "HAR")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Wine Quality (4,898 samples, 11 features):")
    print(f"  Error rate: {error_wine:.4f}")
    print(f"  Challenge: Ordinal labels, subjective ratings")
    print(f"\nHAR (10,299 samples, 561 features):")
    print(f"  Error rate: {error_har:.4f}")
    print(f"  Challenge: High dimensionality, regularization critical")
    
    print("\nModel Appropriateness:")
    print("  Wine: Questionable (ordinal data)")
    print("  HAR: Reasonable (sensor data)")
    
    print("\nQuestion 3 Complete!")

if __name__ == "__main__":
    main()
