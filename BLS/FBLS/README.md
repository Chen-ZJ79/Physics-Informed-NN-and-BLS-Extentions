# Fuzzy Broad Learning System (FBLS) - Python Implementation

This directory contains a Python implementation of the Fuzzy Broad Learning System (FBLS) for classification tasks, based on the original MATLAB implementation.

## Files Description

### Original MATLAB Files
- `FBLSclassify.m` - Main classification script
- `bls_train.m` - Training function
- `result_tra.m` - Result transformation function
- `wbc.mat` - Wisconsin Breast Cancer dataset
- `optimal.mat` - Optimal parameters
- `result.mat` - Results storage

### Python Implementation
- `fuzzy_broad_learning_system.py` - Main FBLS class implementation
- `fbls_demo.py` - Demonstration scripts with examples
- `README.md` - This documentation file

## Features

The Python implementation includes:

1. **Fuzzy Subsystem**: 
   - Gaussian membership functions
   - K-means clustering for center initialization
   - Multiple fuzzy subsystems

2. **Enhancement Layer**:
   - Random weight initialization
   - Hyperbolic tangent activation function
   - Shrinkage parameter control

3. **Output Layer**:
   - Ridge regression for weight computation
   - Regularization parameter
   - Multi-class classification support

4. **Evaluation Tools**:
   - Accuracy calculation
   - Confusion matrix visualization
   - Classification report
   - Parameter sensitivity analysis

## Installation Requirements

```bash
pip install numpy matplotlib scikit-learn
```

## Quick Start

### Basic Usage

```python
from fuzzy_broad_learning_system import FuzzyBroadLearningSystem
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
data = load_iris()
X, y = data.data, data.target + 1  # Convert to 1-based indexing

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize FBLS
fbls = FuzzyBroadLearningSystem(
    num_rules=2,      # Number of fuzzy rules per subsystem
    num_fuzzy=4,      # Number of fuzzy subsystems
    num_enhance=10,   # Number of enhancement nodes
    shrinkage=0.8     # Shrinkage parameter
)

# Train model
train_time, train_acc = fbls.fit(X_train, y_train)

# Make predictions
predictions = fbls.predict(X_test)
test_acc = np.mean(predictions == y_test)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy: {test_acc:.4f}")
```

### Run Demonstrations

```bash
# Run the main demo
python fuzzy_broad_learning_system.py

# Run additional examples
python fbls_demo.py
```

## Parameters

### FuzzyBroadLearningSystem Parameters

- `num_rules` (int, default=2): Number of fuzzy rules per fuzzy subsystem
- `num_fuzzy` (int, default=6): Number of fuzzy subsystems
- `num_enhance` (int, default=20): Number of enhancement nodes
- `shrinkage` (float, default=0.8): Shrinkage parameter for enhancement nodes
- `regularization` (float, default=2**-30): Regularization parameter for sparse regularization
- `std` (float, default=1.0): Standard deviation for Gaussian membership functions

### Parameter Guidelines

1. **num_rules**: Start with 2-4 rules. More rules can improve accuracy but increase complexity.
2. **num_fuzzy**: Typically 4-8 subsystems. More subsystems provide more diversity.
3. **num_enhance**: Usually 10-30 nodes. Balance between performance and computational cost.
4. **shrinkage**: Range [0.1, 1.0]. Lower values provide more regularization.

## Algorithm Overview

The FBLS algorithm consists of three main components:

### 1. Fuzzy Subsystem
```
For each fuzzy subsystem i:
    1. Generate random coefficients αᵢ
    2. Perform K-means clustering to find centers
    3. Compute Gaussian membership functions
    4. Calculate fuzzy rule outputs: μ(x) * (xᵀαᵢ)
    5. Normalize outputs using MinMax scaling
```

### 2. Enhancement Layer
```
1. Concatenate fuzzy outputs with bias term
2. Apply random weights: H₂ * W_enhance
3. Apply shrinkage: s/max(T₂)
4. Apply activation function: tanh(T₂ * l₂)
```

### 3. Output Layer
```
1. Concatenate fuzzy and enhancement outputs
2. Solve ridge regression: β = (T₃ᵀT₃ + λI)⁻¹T₃ᵀy
3. Make predictions: ŷ = argmax(T₃ * β)
```

## Performance Comparison

The Python implementation maintains the core functionality of the MATLAB version while providing:

- **Better Integration**: Works seamlessly with scikit-learn
- **Visualization**: Built-in plotting and evaluation tools
- **Flexibility**: Easy parameter tuning and experimentation
- **Documentation**: Comprehensive docstrings and examples

## Examples

### Example 1: Breast Cancer Classification
```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target + 1

# Train FBLS
fbls = FuzzyBroadLearningSystem(num_rules=2, num_fuzzy=6, num_enhance=20)
train_time, train_acc = fbls.fit(X_train, y_train)
predictions = fbls.predict(X_test)
```

### Example 2: Multi-class Classification
```python
from sklearn.datasets import make_classification

X, y = make_classification(n_classes=3, n_features=10, n_samples=1000)
y = y + 1  # Convert to 1-based indexing

# Train with more rules for multi-class
fbls = FuzzyBroadLearningSystem(num_rules=3, num_fuzzy=5, num_enhance=15)
train_time, train_acc = fbls.fit(X_train, y_train)
```

### Example 3: Parameter Tuning
```python
# Test different parameter combinations
param_grid = [
    (2, 4, 10),   # (rules, fuzzy, enhance)
    (3, 6, 15),
    (4, 8, 20),
]

best_acc = 0
best_params = None

for num_rules, num_fuzzy, num_enhance in param_grid:
    fbls = FuzzyBroadLearningSystem(num_rules, num_fuzzy, num_enhance)
    train_time, train_acc = fbls.fit(X_train, y_train)
    predictions = fbls.predict(X_test)
    test_acc = np.mean(predictions == y_test)
    
    if test_acc > best_acc:
        best_acc = test_acc
        best_params = (num_rules, num_fuzzy, num_enhance)
```

## Troubleshooting

### Common Issues

1. **Low Accuracy**: Try increasing `num_rules` or `num_fuzzy`
2. **Overfitting**: Increase `regularization` or decrease `num_enhance`
3. **Slow Training**: Decrease `num_fuzzy` or `num_enhance`
4. **Memory Issues**: Reduce dataset size or parameter values

### Performance Tips

1. **Data Preprocessing**: Normalize input features for better performance
2. **Parameter Selection**: Use cross-validation to find optimal parameters
3. **Random State**: Set random seeds for reproducible results
4. **Batch Processing**: For large datasets, consider batch processing

## References

This implementation is based on the original MATLAB code for Fuzzy Broad Learning Systems. The algorithm combines fuzzy logic with broad learning system architecture for efficient classification tasks.

## License

This implementation is provided for educational and research purposes. Please cite the original work if you use this code in your research.

