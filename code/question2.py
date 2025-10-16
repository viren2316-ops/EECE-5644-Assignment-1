#!/usr/bin/env python3
"""
EECE5644 Assignment 1 - Question 2
4-Class Gaussian Mixture Classification
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
import os

os.makedirs('data/generated', exist_ok=True)
os.makedirs('figures', exist_ok=True)
np.random.seed(42)

# DATA GENERATION

def generate_data_q2(N=10000):
    """Generate 4-class Gaussian mixture data."""
    print("Generating 4-class Gaussian mixture data...")
    
    # Define parameters - Class 4 (index 3) at center for maximum overlap
    means = [
        np.array([-3.0, -3.0]),  # Class 1: Bottom-left
        np.array([3.0, -3.0]),   # Class 2: Bottom-right  
        np.array([-3.0, 3.0]),   # Class 3: Top-left
        np.array([0.0, 0.0])     # Class 4: Center (maximum overlap)
    ]
    
    covs = [
        np.array([[1.0, 0.2], [0.2, 1.0]]),      # Class 1
        np.array([[1.2, -0.3], [-0.3, 1.0]]),    # Class 2
        np.array([[0.8, 0.1], [0.1, 1.2]]),      # Class 3
        np.array([[2.5, 0.4], [0.4, 2.5]])       # Class 4: Large variance
    ]
    
    # Equal priors
    priors = np.array([0.25, 0.25, 0.25, 0.25])
    
    # Generate samples
    labels = np.random.choice(4, size=N, p=priors)
    X = np.zeros((N, 2))
    
    for i in range(4):
        mask = labels == i
        n_samples = np.sum(mask)
        X[mask] = np.random.multivariate_normal(means[i], covs[i], n_samples)
    
    print(f"Generated {N} samples")
    for i in range(4):
        count = np.sum(labels == i)
        print(f"  Class {i+1}: {count} samples ({count/N*100:.1f}%)")
    
    # Save data
    np.save('data/generated/q2_data.npy', X)
    np.save('data/generated/q2_labels.npy', labels)
    np.savez('data/generated/q2_params.npz', means=means, covs=covs, priors=priors)
    
    return X, labels, means, covs, priors

# PART A: MAP CLASSIFIER

def map_classifier(X, means, covs, priors):
    """MAP classifier for minimum probability of error."""
    N = X.shape[0]
    posteriors = np.zeros((N, 4))
    
    for i in range(4):
        likelihood = multivariate_normal.pdf(X, means[i], covs[i])
        posteriors[:, i] = likelihood * priors[i]
    
    decisions = np.argmax(posteriors, axis=1)
    return decisions

def part_a_map(X, labels, means, covs, priors):
    """Part A: MAP classification with 0-1 loss."""
    print("\n" + "="*70)
    print("PART A: MAP Classifier (0-1 Loss)")
    print("="*70)
    
    # Classify
    decisions = map_classifier(X, means, covs, priors)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(labels, decisions, labels=np.arange(4))
    
    # Normalize by columns (true labels)
    conf_matrix_norm = conf_matrix.astype(float)
    for j in range(4):
        total = np.sum(labels == j)
        if total > 0:
            conf_matrix_norm[:, j] /= total
    
    # Print results
    print("\nConfusion Matrix P(D=i|L=j):")
    print("       L=0    L=1    L=2    L=3")
    for i in range(4):
        print(f"D={i}: ", end="")
        for j in range(4):
            print(f"{conf_matrix_norm[i, j]:6.3f} ", end="")
        print()
    
    # Error rate
    error_rate = 1 - np.sum(decisions == labels) / len(labels)
    print(f"\nError rate: {error_rate:.4f}")
    
    # Visualization
    plt.figure(figsize=(10, 8))
    for true_label in range(4):
        mask = labels == true_label
        correct = decisions[mask] == true_label
        
        mask_correct = mask.copy()
        mask_correct[mask] = correct
        mask_incorrect = mask.copy()
        mask_incorrect[mask] = ~correct
        
        if np.any(mask_correct):
            plt.scatter(X[mask_correct, 0], X[mask_correct, 1], 
                       c='green', alpha=0.6, s=30, 
                       marker=['o', '^', 's', 'D'][true_label])
        if np.any(mask_incorrect):
            plt.scatter(X[mask_incorrect, 0], X[mask_incorrect, 1], 
                       c='red', alpha=0.6, s=30,
                       marker=['o', '^', 's', 'D'][true_label])
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('MAP Classification Results (Green=Correct, Red=Incorrect)')
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/q2_map_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return conf_matrix_norm, error_rate

# PART B: ERM CLASSIFIER WITH CUSTOM LOSS

def erm_classifier(X, means, covs, priors, loss_matrix):
    """ERM classifier with custom loss matrix."""
    N = X.shape[0]
    expected_risks = np.zeros((N, 4))
    
    # Compute posteriors first
    posteriors = np.zeros((N, 4))
    for l in range(4):
        likelihood = multivariate_normal.pdf(X, means[l], covs[l])
        posteriors[:, l] = likelihood * priors[l]
    
    # Normalize posteriors
    posteriors = posteriors / (np.sum(posteriors, axis=1, keepdims=True) + 1e-10)
    
    # Compute expected risk for each decision
    for d in range(4):
        for l in range(4):
            expected_risks[:, d] += loss_matrix[d, l] * posteriors[:, l]
    
    decisions = np.argmin(expected_risks, axis=1)
    return decisions

def part_b_erm(X, labels, means, covs, priors):
    """Part B: ERM classification with custom loss matrix."""
    print("\n" + "="*70)
    print("PART B: ERM Classifier with Custom Loss Matrix")
    print("="*70)
    
    # Loss matrix - high penalty for misclassifying Class 4
    loss_matrix = np.array([
        [0,  10, 10, 100],
        [1,  0,  10, 100],
        [1,  1,  0,  100],
        [1,  1,  1,  0]
    ])
    
    print("\nLoss Matrix:")
    print("       L=0    L=1    L=2    L=3")
    for i in range(4):
        print(f"D={i}: ", end="")
        for j in range(4):
            print(f"{loss_matrix[i, j]:6.0f} ", end="")
        print()
    
    # Classify
    decisions = erm_classifier(X, means, covs, priors, loss_matrix)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(labels, decisions, labels=np.arange(4))
    conf_matrix_norm = conf_matrix.astype(float)
    for j in range(4):
        total = np.sum(labels == j)
        if total > 0:
            conf_matrix_norm[:, j] /= total
    
    print("\nConfusion Matrix P(D=i|L=j):")
    print("       L=0    L=1    L=2    L=3")
    for i in range(4):
        print(f"D={i}: ", end="")
        for j in range(4):
            print(f"{conf_matrix_norm[i, j]:6.3f} ", end="")
        print()
    
    # Calculate average risk
    total_risk = 0
    for i in range(4):
        for j in range(4):
            total_risk += loss_matrix[i, j] * conf_matrix[i, j]
    avg_risk = total_risk / len(labels)
    
    print(f"\nAverage empirical risk: {avg_risk:.4f}")
    print(f"Class 4 correct rate: {conf_matrix_norm[3, 3]:.4f}")
    
    # Visualization
    plt.figure(figsize=(10, 8))
    for true_label in range(4):
        mask = labels == true_label
        correct = decisions[mask] == true_label
        
        mask_correct = mask.copy()
        mask_correct[mask] = correct
        mask_incorrect = mask.copy()
        mask_incorrect[mask] = ~correct
        
        if np.any(mask_correct):
            plt.scatter(X[mask_correct, 0], X[mask_correct, 1], 
                       c='green', alpha=0.6, s=30,
                       marker=['o', '^', 's', 'D'][true_label])
        if np.any(mask_incorrect):
            plt.scatter(X[mask_incorrect, 0], X[mask_incorrect, 1], 
                       c='red', alpha=0.6, s=30,
                       marker=['o', '^', 's', 'D'][true_label])
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('ERM Classification Results (Green=Correct, Red=Incorrect)')
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/q2_erm_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return conf_matrix_norm, avg_risk

# VISUALIZE GAUSSIAN CONTOURS

def plot_gaussian_contours(means, covs):
    """Plot contours of the 4 Gaussians."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x = np.linspace(-8, 8, 200)
    y = np.linspace(-8, 8, 200)
    XX, YY = np.meshgrid(x, y)
    pos = np.dstack((XX, YY))
    
    colors = ['blue', 'green', 'orange', 'red']
    
    for i in range(4):
        rv = multivariate_normal(means[i], covs[i])
        contours = ax.contour(XX, YY, rv.pdf(pos), 
                             levels=3, colors=colors[i], alpha=0.7)
        ax.plot(means[i][0], means[i][1], '*', 
               markersize=15, color=colors[i], markeredgecolor='black')
        ax.text(means[i][0], means[i][1]+0.5, f'Class {i+1}', 
               fontsize=12, ha='center')
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('4-Class Gaussian Mixture Contours')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-8, 8])
    ax.set_ylim([-8, 8])
    plt.savefig('figures/q2_gaussian_contours.png', dpi=150, bbox_inches='tight')
    plt.show()

# MAIN

def main():
    print("="*70)
    print("EECE5644 Assignment 1 - Question 2")
    print("4-Class Gaussian Mixture Classification")
    print("="*70)
    
    # Generate or load data
    if os.path.exists('data/generated/q2_data.npy'):
        print("\nLoading existing data...")
        X = np.load('data/generated/q2_data.npy')
        labels = np.load('data/generated/q2_labels.npy')
        params = np.load('data/generated/q2_params.npz')
        means = params['means']
        covs = params['covs']
        priors = params['priors']
    else:
        X, labels, means, covs, priors = generate_data_q2(N=10000)
    
    # Visualize Gaussians
    plot_gaussian_contours(means, covs)
    
    # Part A: MAP Classification
    conf_map, error_map = part_a_map(X, labels, means, covs, priors)
    
    # Part B: ERM Classification
    conf_erm, avg_risk = part_b_erm(X, labels, means, covs, priors)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"MAP Classifier:")
    print(f"  Error rate: {error_map:.4f}")
    print(f"ERM Classifier:")
    print(f"  Average risk: {avg_risk:.4f}")
    print(f"  Class 4 protection: {conf_erm[3, 3]:.4f} vs MAP: {conf_map[3, 3]:.4f}")
    
    print("\nQuestion 2 Complete!")

if __name__ == "__main__":
    main()
