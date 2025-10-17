"""
Fuzzy Broad Learning System (FBLS) for Classification Tasks
Python implementation based on the MATLAB version

This implementation includes:
1. Fuzzy subsystem with membership functions
2. Enhancement layer with activation functions
3. Output layer for classification
4. Training and testing procedures
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')


class FuzzyBroadLearningSystem:
    """
    Fuzzy Broad Learning System for classification tasks
    
    Parameters:
    -----------
    num_rules : int, default=2
        Number of fuzzy rules per fuzzy subsystem
    num_fuzzy : int, default=6
        Number of fuzzy subsystems
    num_enhance : int, default=20
        Number of enhancement nodes
    shrinkage : float, default=0.8
        Shrinkage parameter for enhancement nodes
    regularization : float, default=2**-30
        Regularization parameter for sparse regularization
    std : float, default=1.0
        Standard deviation for Gaussian membership functions
    """
    
    def __init__(self, num_rules=2, num_fuzzy=6, num_enhance=20, 
                 shrinkage=0.8, regularization=2**-30, std=1.0):
        self.num_rules = num_rules
        self.num_fuzzy = num_fuzzy
        self.num_enhance = num_enhance
        self.shrinkage = shrinkage
        self.regularization = regularization
        self.std = std
        
        # Initialize components
        self.alpha = []  # Coefficients for fuzzy rules
        self.centers = []  # Cluster centers for membership functions
        self.scalers = []  # MinMax scalers for each fuzzy subsystem
        self.weight_enhance = None  # Weights for enhancement layer
        self.beta = None  # Output weights
        self.l2 = None  # Scaling factor for enhancement layer
        
    def _gaussian_membership(self, x, centers):
        """
        Compute Gaussian membership functions
        
        Parameters:
        -----------
        x : array-like, shape (n_samples, n_features)
            Input data
        centers : array-like, shape (n_rules, n_features)
            Cluster centers
            
        Returns:
        --------
        membership : array-like, shape (n_samples, n_rules)
            Membership function values
        """
        n_samples, n_features = x.shape
        n_rules = centers.shape[0]
        
        membership = np.zeros((n_samples, n_rules))
        
        for i in range(n_samples):
            for j in range(n_rules):
                # Gaussian membership function
                diff = x[i, :] - centers[j, :]
                membership[i, j] = np.exp(-np.sum(diff**2) / self.std**2)
        
        # Normalize membership functions
        membership = membership / np.sum(membership, axis=1, keepdims=True)
        
        return membership
    
    def _tansig(self, x):
        """
        Hyperbolic tangent sigmoid activation function
        Equivalent to MATLAB's tansig function
        """
        return 2 / (1 + np.exp(-2 * x)) - 1
    
    def _result_transform(self, x):
        """
        Transform continuous output to discrete class labels
        Equivalent to MATLAB's result_tra function
        """
        return np.argmax(x, axis=1) + 1
    
    def fit(self, X_train, y_train):
        """
        Train the Fuzzy Broad Learning System
        
        Parameters:
        -----------
        X_train : array-like, shape (n_samples, n_features)
            Training input data
        y_train : array-like, shape (n_samples,)
            Training target labels
        """
        print(f"Training FBLS with {self.num_rules} rules, {self.num_fuzzy} fuzzy systems, {self.num_enhance} enhancement nodes")
        
        start_time = time.time()
        
        n_samples, n_features = X_train.shape
        
        # Initialize fuzzy subsystems
        fuzzy_outputs = np.zeros((n_samples, self.num_fuzzy * self.num_rules))
        
        for i in range(self.num_fuzzy):
            # Generate random coefficients for fuzzy rules
            alpha_i = np.random.randn(n_features, self.num_rules)
            self.alpha.append(alpha_i)
            
            # Perform K-means clustering to find centers
            kmeans = KMeans(n_clusters=self.num_rules, random_state=42, n_init=10)
            centers_i = kmeans.fit(X_train).cluster_centers_
            self.centers.append(centers_i)
            
            # Compute fuzzy outputs
            fuzzy_output_i = np.zeros((n_samples, self.num_rules))
            
            for j in range(n_samples):
                # Compute membership functions
                membership = self._gaussian_membership(X_train[j:j+1, :], centers_i)
                
                # Compute fuzzy rule outputs
                fuzzy_output_i[j, :] = membership.flatten() * (X_train[j, :] @ alpha_i)
            
            # Normalize fuzzy outputs
            scaler_i = MinMaxScaler()
            fuzzy_output_i_scaled = scaler_i.fit_transform(fuzzy_output_i.T).T
            self.scalers.append(scaler_i)
            
            # Store fuzzy outputs
            fuzzy_outputs[:, self.num_rules*i:self.num_rules*(i+1)] = fuzzy_output_i_scaled
        
        # Prepare input for enhancement layer
        H2 = np.hstack([fuzzy_outputs, 0.1 * np.ones((n_samples, 1))])
        
        # Initialize enhancement layer weights
        self.weight_enhance = np.random.randn(self.num_fuzzy * self.num_rules + 1, self.num_enhance)
        
        # Compute enhancement layer outputs
        T2 = H2 @ self.weight_enhance
        
        # Apply shrinkage
        self.l2 = self.shrinkage / np.max(T2)
        T2 = self._tansig(T2 * self.l2)
        
        # Combine fuzzy and enhancement outputs
        T3 = np.hstack([fuzzy_outputs, T2])
        
        # Compute output weights using ridge regression
        I = np.eye(T3.shape[1])
        self.beta = np.linalg.solve(T3.T @ T3 + self.regularization * I, T3.T @ y_train)
        
        training_time = time.time() - start_time
        
        # Compute training accuracy
        train_pred = T3 @ self.beta
        train_pred_labels = self._result_transform(train_pred)
        train_accuracy = accuracy_score(y_train, train_pred_labels)
        
        print(f"Training completed in {training_time:.4f} seconds")
        print(f"Training accuracy: {train_accuracy:.4f}")
        
        return training_time, train_accuracy
    
    def predict(self, X_test):
        """
        Make predictions on test data
        
        Parameters:
        -----------
        X_test : array-like, shape (n_samples, n_features)
            Test input data
            
        Returns:
        --------
        predictions : array-like, shape (n_samples,)
            Predicted class labels
        """
        n_samples = X_test.shape[0]
        
        # Compute fuzzy outputs for test data
        fuzzy_outputs = np.zeros((n_samples, self.num_fuzzy * self.num_rules))
        
        for i in range(self.num_fuzzy):
            centers_i = self.centers[i]
            alpha_i = self.alpha[i]
            scaler_i = self.scalers[i]
            
            # Compute fuzzy outputs
            fuzzy_output_i = np.zeros((n_samples, self.num_rules))
            
            for j in range(n_samples):
                # Compute membership functions
                membership = self._gaussian_membership(X_test[j:j+1, :], centers_i)
                
                # Compute fuzzy rule outputs
                fuzzy_output_i[j, :] = membership.flatten() * (X_test[j, :] @ alpha_i)
            
            # Apply scaling
            fuzzy_output_i_scaled = scaler_i.transform(fuzzy_output_i.T).T
            
            # Store fuzzy outputs
            fuzzy_outputs[:, self.num_rules*i:self.num_rules*(i+1)] = fuzzy_output_i_scaled
        
        # Prepare input for enhancement layer
        H2 = np.hstack([fuzzy_outputs, 0.1 * np.ones((n_samples, 1))])
        
        # Compute enhancement layer outputs
        T2 = self._tansig(H2 @ self.weight_enhance * self.l2)
        
        # Combine fuzzy and enhancement outputs
        T3 = np.hstack([fuzzy_outputs, T2])
        
        # Make predictions
        test_pred = T3 @ self.beta
        test_pred_labels = self._result_transform(test_pred)
        
        return test_pred_labels


def load_sample_data():
    """
    Load sample classification dataset
    """
    # Load breast cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Convert to binary classification (0, 1) -> (1, 2) to match MATLAB format
    y = y + 1
    
    return X, y


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    start_time = time.time()
    predictions = model.predict(X_test)
    test_time = time.time() - start_time
    
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Testing completed in {test_time:.4f} seconds")
    print(f"Testing accuracy: {accuracy:.4f}")
    
    return accuracy, test_time, predictions


def plot_results(y_true, y_pred, title="Classification Results"):
    """
    Plot confusion matrix and classification report
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.figure.colorbar(im, ax=ax1)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    ax1.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=f'{title} - Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    ax2.axis('tight')
    ax2.axis('off')
    
    # Create table data
    table_data = []
    for class_label in classes:
        if str(class_label) in report:
            metrics = report[str(class_label)]
            table_data.append([f'Class {class_label}', 
                             f"{metrics['precision']:.3f}",
                             f"{metrics['recall']:.3f}",
                             f"{metrics['f1-score']:.3f}"])
    
    table_data.append(['', '', '', ''])
    table_data.append(['Accuracy', '', '', f"{report['accuracy']:.3f}"])
    
    table = ax2.table(cellText=table_data,
                     colLabels=['Class', 'Precision', 'Recall', 'F1-Score'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    ax2.set_title(f'{title} - Classification Report')
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to demonstrate FBLS classification
    """
    print("=" * 60)
    print("Fuzzy Broad Learning System (FBLS) Classification Demo")
    print("=" * 60)
    
    # Load data
    print("Loading dataset...")
    X, y = load_sample_data()
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Initialize and train FBLS
    print("\nInitializing FBLS...")
    fbls = FuzzyBroadLearningSystem(
        num_rules=2,
        num_fuzzy=6,
        num_enhance=20,
        shrinkage=0.8,
        regularization=2**-30
    )
    
    # Train model
    print("\nTraining FBLS...")
    train_time, train_accuracy = fbls.fit(X_train, y_train)
    
    # Test model
    print("\nTesting FBLS...")
    test_accuracy, test_time, predictions = evaluate_model(fbls, X_test, y_test)
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Training Time: {train_time:.4f} seconds")
    print(f"Testing Time: {test_time:.4f} seconds")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    
    # Plot results
    plot_results(y_test, predictions, "FBLS Classification")
    
    return fbls, train_time, test_time, train_accuracy, test_accuracy


if __name__ == "__main__":
    # Run the main function
    model, train_t, test_t, train_acc, test_acc = main()
    
    print("\nFBLS classification demo completed successfully!")

