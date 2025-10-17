"""
Simple example demonstrating FBLS classification
"""

import numpy as np
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fuzzy_broad_learning_system import FuzzyBroadLearningSystem, plot_results
import matplotlib.pyplot as plt


def demo_synthetic_data():
    """
    Demonstrate FBLS on synthetic classification data
    """
    print("=" * 50)
    print("FBLS Demo: Synthetic Classification Data")
    print("=" * 50)
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Convert labels to 1-based indexing (to match MATLAB format)
    y = y + 1
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Initialize FBLS
    fbls = FuzzyBroadLearningSystem(
        num_rules=3,
        num_fuzzy=5,
        num_enhance=15,
        shrinkage=0.8
    )
    
    # Train and test
    train_time, train_acc = fbls.fit(X_train, y_train)
    predictions = fbls.predict(X_test)
    
    # Calculate test accuracy
    test_acc = np.mean(predictions == y_test)
    
    print(f"\nResults:")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")
    print(f"Training Time: {train_time:.4f} seconds")
    
    # Plot results
    plot_results(y_test, predictions, "FBLS - Synthetic Data")
    
    return fbls, train_acc, test_acc


def demo_iris_data():
    """
    Demonstrate FBLS on Iris dataset
    """
    print("\n" + "=" * 50)
    print("FBLS Demo: Iris Dataset")
    print("=" * 50)
    
    # Load Iris dataset
    from sklearn.datasets import load_iris
    data = load_iris()
    X, y = data.data, data.target
    
    # Convert to 1-based indexing
    y = y + 1
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Initialize FBLS
    fbls = FuzzyBroadLearningSystem(
        num_rules=2,
        num_fuzzy=4,
        num_enhance=10,
        shrinkage=0.8
    )
    
    # Train and test
    train_time, train_acc = fbls.fit(X_train, y_train)
    predictions = fbls.predict(X_test)
    
    # Calculate test accuracy
    test_acc = np.mean(predictions == y_test)
    
    print(f"\nResults:")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")
    print(f"Training Time: {train_time:.4f} seconds")
    
    # Plot results
    plot_results(y_test, predictions, "FBLS - Iris Dataset")
    
    return fbls, train_acc, test_acc


def parameter_sensitivity_analysis():
    """
    Analyze the sensitivity of FBLS to different parameters
    """
    print("\n" + "=" * 50)
    print("FBLS Parameter Sensitivity Analysis")
    print("=" * 50)
    
    # Generate data
    X, y = make_classification(
        n_samples=500,
        n_features=8,
        n_classes=2,
        random_state=42
    )
    y = y + 1
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Test different parameter combinations
    param_combinations = [
        (2, 4, 10),   # (num_rules, num_fuzzy, num_enhance)
        (3, 6, 15),
        (4, 8, 20),
        (2, 6, 20),
        (3, 4, 15),
    ]
    
    results = []
    
    for num_rules, num_fuzzy, num_enhance in param_combinations:
        print(f"\nTesting: Rules={num_rules}, Fuzzy={num_fuzzy}, Enhance={num_enhance}")
        
        fbls = FuzzyBroadLearningSystem(
            num_rules=num_rules,
            num_fuzzy=num_fuzzy,
            num_enhance=num_enhance,
            shrinkage=0.8
        )
        
        train_time, train_acc = fbls.fit(X_train, y_train)
        predictions = fbls.predict(X_test)
        test_acc = np.mean(predictions == y_test)
        
        results.append({
            'num_rules': num_rules,
            'num_fuzzy': num_fuzzy,
            'num_enhance': num_enhance,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'train_time': train_time
        })
        
        print(f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Time: {train_time:.4f}s")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    x_labels = [f"R{r['num_rules']}F{r['num_fuzzy']}E{r['num_enhance']}" 
                for r in results]
    train_accs = [r['train_acc'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    
    x = np.arange(len(x_labels))
    width = 0.35
    
    ax1.bar(x - width/2, train_accs, width, label='Training Accuracy', alpha=0.8)
    ax1.bar(x + width/2, test_accs, width, label='Testing Accuracy', alpha=0.8)
    ax1.set_xlabel('Parameter Combinations')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('FBLS Accuracy vs Parameters')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Training time comparison
    train_times = [r['train_time'] for r in results]
    ax2.bar(x, train_times, alpha=0.8, color='orange')
    ax2.set_xlabel('Parameter Combinations')
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('FBLS Training Time vs Parameters')
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


if __name__ == "__main__":
    # Run demonstrations
    print("Fuzzy Broad Learning System (FBLS) Classification Examples")
    print("=" * 60)
    
    # Demo 1: Synthetic data
    demo_synthetic_data()
    
    # Demo 2: Iris dataset
    demo_iris_data()
    
    # Demo 3: Parameter sensitivity
    param_results = parameter_sensitivity_analysis()
    
    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60)

