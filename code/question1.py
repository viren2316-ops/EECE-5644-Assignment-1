#!/usr/bin/env python3
"""
EECE5644 Assignment 1 - Question 1
ERM, Naive Bayes, and Fisher LDA Classification with ROC Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.linalg import eigh
import os

os.makedirs('data/generated', exist_ok=True)
os.makedirs('figures', exist_ok=True)
np.random.seed(42)

# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_data(N=10000, save=True):
    """Generate data from specified distribution."""
    print(f"Generating {N} samples...")
    
    # Class priors
    p0, p1 = 0.65, 0.35
    
    # Generate labels
    u = np.random.rand(N)
    labels = (u >= p0).astype(int)
    N0 = np.sum(labels == 0)
    N1 = np.sum(labels == 1)
    
    print(f"Class 0: {N0} samples ({N0/N*100:.1f}%)")
    print(f"Class 1: {N1} samples ({N1/N*100:.1f}%)")
    
    # Class parameters
    mu0 = np.array([-0.5, -0.5, -0.5])
    Sigma0 = np.array([[1.0, -0.5, 0.3],
                       [-0.5, 1.0, -0.5],
                       [0.3, -0.5, 1.0]])
    
    mu1 = np.array([1.0, 1.0, 1.0])
    Sigma1 = np.array([[1.0, 0.3, -0.2],
                       [0.3, 1.0, 0.3],
                       [-0.2, 0.3, 1.0]])
    
    # Generate samples
    X = np.zeros((N, 3))
    X[labels == 0] = np.random.multivariate_normal(mu0, Sigma0, N0)
    X[labels == 1] = np.random.multivariate_normal(mu1, Sigma1, N1)
    
    # Save data
    if save:
        np.save('data/generated/q1_data.npy', X)
        np.save('data/generated/q1_labels.npy', labels)
        print("Data saved to data/generated/")
    
    return X, labels, (mu0, Sigma0, mu1, Sigma1)

def visualize_data_3d(X, labels):
    """Create 3D scatter plot of data."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    mask0 = labels == 0
    mask1 = labels == 1
    
    ax.scatter(X[mask0, 0], X[mask0, 1], X[mask0, 2], 
              c='blue', marker='o', alpha=0.5, s=10, label='Class 0')
    ax.scatter(X[mask1, 0], X[mask1, 1], X[mask1, 2], 
              c='red', marker='^', alpha=0.5, s=10, label='Class 1')
    
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.set_title('Generated Data Distribution')
    ax.legend()
    plt.savefig('figures/q1_data_3d.png', dpi=150, bbox_inches='tight')
    plt.show()

# PART A: ERM WITH TRUE PDF

def compute_roc_true_pdf(X, labels, params, gammas):
    """Compute ROC curve for true PDF."""
    mu0, Sigma0, mu1, Sigma1 = params
    
    pdf0 = multivariate_normal(mu0, Sigma0)
    pdf1 = multivariate_normal(mu1, Sigma1)
    
    L0 = pdf0.pdf(X)
    L1 = pdf1.pdf(X)
    ratio = L1 / (L0 + 1e-300)
    
    tpr, fpr, errors = [], [], []
    p0, p1 = 0.65, 0.35
    
    for gamma in gammas:
        decisions = (ratio > gamma).astype(int)
        tp = np.sum((decisions == 1) & (labels == 1)) / np.sum(labels == 1)
        fp = np.sum((decisions == 1) & (labels == 0)) / np.sum(labels == 0)
        error = fp * p0 + (1 - tp) * p1
        
        tpr.append(tp)
        fpr.append(fp)
        errors.append(error)
    
    return np.array(tpr), np.array(fpr), np.array(errors)

def part_a_analysis(X, labels, params):
    """Part A: ERM with true PDF knowledge."""
    print("\n" + "="*70)
    print("PART A: ERM Classifier with True PDF Knowledge")
    print("="*70)
    
    gammas = np.logspace(-3, 3, 1000)
    tpr, fpr, errors = compute_roc_true_pdf(X, labels, params, gammas)
    
    min_idx = np.argmin(errors)
    optimal_gamma = gammas[min_idx]
    min_error = errors[min_idx]
    theoretical_gamma = 0.65 / 0.35
    
    print(f"Results:")
    print(f"  Empirical optimal gamma: {optimal_gamma:.4f}")
    print(f"  Theoretical optimal gamma: {theoretical_gamma:.4f}")
    print(f"  Ratio (empirical/theoretical): {optimal_gamma/theoretical_gamma:.4f}")
    print(f"  Minimum P(error): {min_error:.4f}")
    print(f"  TPR at min error: {tpr[min_idx]:.4f}")
    print(f"  FPR at min error: {fpr[min_idx]:.4f}")
    
    # Plot individual ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label='ROC Curve')
    plt.plot(fpr[min_idx], tpr[min_idx], 'ro', markersize=10, 
             label=f'Min P(error) = {min_error:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Part A: ROC Curve - ERM with True PDF')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/q1_part_a_roc.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return tpr, fpr, errors, optimal_gamma, min_error

# PART B: NAIVE BAYES

def part_b_analysis(X, labels):
    """Part B: Naive Bayes with incorrect covariances."""
    print("\n" + "="*70)
    print("PART B: Naive Bayes Classifier (Model Mismatch)")
    print("="*70)
    
    # True means, identity covariances
    mu0 = np.array([-0.5, -0.5, -0.5])
    mu1 = np.array([1.0, 1.0, 1.0])
    params_naive = (mu0, np.eye(3), mu1, np.eye(3))
    
    print("Using incorrect model: Identity covariance matrices")
    
    gammas = np.logspace(-3, 3, 1000)
    tpr, fpr, errors = compute_roc_true_pdf(X, labels, params_naive, gammas)
    
    min_idx = np.argmin(errors)
    optimal_gamma = gammas[min_idx]
    min_error = errors[min_idx]
    
    print(f"Results:")
    print(f"  Empirical optimal gamma: {optimal_gamma:.4f}")
    print(f"  Minimum P(error): {min_error:.4f}")
    print(f"  TPR at min error: {tpr[min_idx]:.4f}")
    print(f"  FPR at min error: {fpr[min_idx]:.4f}")
    
    # Plot individual ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'r-', linewidth=2, label='ROC Curve')
    plt.plot(fpr[min_idx], tpr[min_idx], 'ro', markersize=10,
             label=f'Min P(error) = {min_error:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Part B: ROC Curve - Naive Bayes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/q1_part_b_roc.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return tpr, fpr, errors, optimal_gamma, min_error

# PART C: FISHER LDA

def part_c_analysis(X, labels):
    """Part C: Fisher LDA classifier."""
    print("\n" + "="*70)
    print("PART C: Fisher LDA Classifier")
    print("="*70)
    
    # Separate classes
    X0 = X[labels == 0]
    X1 = X[labels == 1]
    
    # Estimate parameters
    mu0_est = np.mean(X0, axis=0)
    mu1_est = np.mean(X1, axis=0)
    Sigma0_est = np.cov(X0.T)
    Sigma1_est = np.cov(X1.T)
    
    print(f"Estimated mean for class 0: {mu0_est}")
    print(f"Estimated mean for class 1: {mu1_est}")
    
    # Within-class and between-class scatter
    Sw = 0.5 * Sigma0_est + 0.5 * Sigma1_est
    mean_diff = (mu1_est - mu0_est).reshape(-1, 1)
    Sb = mean_diff @ mean_diff.T
    
    # Solve generalized eigenvalue problem
    eigenvalues, eigenvectors = eigh(Sb, Sw)
    w_LDA = eigenvectors[:, np.argmax(eigenvalues)]
    w_LDA = w_LDA / np.linalg.norm(w_LDA)
    
    print(f"LDA projection vector: {w_LDA}")
    
    # Project data
    projections = X @ w_LDA
    proj_0 = X0 @ w_LDA
    proj_1 = X1 @ w_LDA
    
    # Plot projection histogram
    plt.figure(figsize=(10, 6))
    plt.hist(proj_0, bins=50, alpha=0.5, label='Class 0', color='blue', density=True)
    plt.hist(proj_1, bins=50, alpha=0.5, label='Class 1', color='red', density=True)
    plt.xlabel('Projected Value')
    plt.ylabel('Density')
    plt.title('Fisher LDA Projections')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/q1_lda_projections.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Compute ROC curve
    thresholds = np.linspace(np.min(projections)-1, np.max(projections)+1, 1000)
    tpr, fpr, errors = [], [], []
    p0, p1 = 0.65, 0.35
    
    for tau in thresholds:
        decisions = (projections > tau).astype(int)
        tp = np.sum((decisions == 1) & (labels == 1)) / np.sum(labels == 1)
        fp = np.sum((decisions == 1) & (labels == 0)) / np.sum(labels == 0)
        error = fp * p0 + (1 - tp) * p1
        
        tpr.append(tp)
        fpr.append(fp)
        errors.append(error)
    
    tpr, fpr, errors = np.array(tpr), np.array(fpr), np.array(errors)
    
    min_idx = np.argmin(errors)
    optimal_tau = thresholds[min_idx]
    min_error = errors[min_idx]
    
    print(f"Results:")
    print(f"  Optimal threshold tau: {optimal_tau:.4f}")
    print(f"  Minimum P(error): {min_error:.4f}")
    print(f"  TPR at min error: {tpr[min_idx]:.4f}")
    print(f"  FPR at min error: {fpr[min_idx]:.4f}")
    
    # Plot individual ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'g-', linewidth=2, label='ROC Curve')
    plt.plot(fpr[min_idx], tpr[min_idx], 'go', markersize=10,
             label=f'Min P(error) = {min_error:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Part C: ROC Curve - Fisher LDA')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/q1_part_c_roc.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return tpr, fpr, errors, optimal_tau, min_error

# COMPARISON


def compare_all_methods(results_a, results_b, results_c):
    """Create comparison plot of all three methods."""
    print("\n" + "="*70)
    print("COMPARISON OF ALL THREE METHODS")
    print("="*70)
    
    tpr_a, fpr_a, _, _, min_error_a = results_a
    tpr_b, fpr_b, _, _, min_error_b = results_b
    tpr_c, fpr_c, _, _, min_error_c = results_c
    
    # Comparison plot
    plt.figure(figsize=(12, 9))
    plt.plot(fpr_a, tpr_a, 'b-', linewidth=2.5, label=f'True PDF (P_error = {min_error_a:.4f})')
    plt.plot(fpr_b, tpr_b, 'r-', linewidth=2.5, label=f'Naive Bayes (P_error = {min_error_b:.4f})')
    plt.plot(fpr_c, tpr_c, 'g-', linewidth=2.5, label=f'Fisher LDA (P_error = {min_error_c:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison - All Three Methods', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/q1_comparison_roc.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Summary table
    print("\nPerformance Summary:")
    print("-" * 50)
    print(f"{'Method':<20} {'Min P(error)':<15} {'Degradation':<15}")
    print("-" * 50)
    print(f"{'True PDF':<20} {min_error_a:<15.4f} {'(Optimal)':<15}")
    print(f"{'Naive Bayes':<20} {min_error_b:<15.4f} {f'(+{(min_error_b-min_error_a)/min_error_a*100:.1f}%)':<15}")
    print(f"{'Fisher LDA':<20} {min_error_c:<15.4f} {f'(+{(min_error_c-min_error_a)/min_error_a*100:.1f}%)':<15}")
    
    print("\nAnalysis:")
    print("  - True PDF achieves best performance (theoretical optimum)")
    print("  - Naive Bayes suffers from model mismatch (ignores correlations)")
    print("  - Fisher LDA performs well despite being a linear classifier")

# MAIN

def main():
    print("="*70)
    print("EECE5644 Assignment 1 - Question 1")
    print("ERM Classification with ROC Analysis")
    print("="*70)
    
    # Generate or load data
    if os.path.exists('data/generated/q1_data.npy'):
        print("\nLoading existing data...")
        X = np.load('data/generated/q1_data.npy')
        labels = np.load('data/generated/q1_labels.npy')
        mu0 = np.array([-0.5, -0.5, -0.5])
        Sigma0 = np.array([[1, -0.5, 0.3], [-0.5, 1, -0.5], [0.3, -0.5, 1]])
        mu1 = np.array([1, 1, 1])
        Sigma1 = np.array([[1, 0.3, -0.2], [0.3, 1, 0.3], [-0.2, 0.3, 1]])
        params = (mu0, Sigma0, mu1, Sigma1)
    else:
        print("\nGenerating new data...")
        X, labels, params = generate_data(N=10000, save=True)
    
    # Visualize data
    visualize_data_3d(X, labels)
    
    # Run all parts
    results_a = part_a_analysis(X, labels, params)
    results_b = part_b_analysis(X, labels)
    results_c = part_c_analysis(X, labels)
    
    # Compare all methods
    compare_all_methods(results_a, results_b, results_c)
    
    print("\n" + "="*70)
    print("Question 1 Complete!")
    print("All figures saved to figures/ directory")
    print("="*70)

if __name__ == "__main__":
    main()
