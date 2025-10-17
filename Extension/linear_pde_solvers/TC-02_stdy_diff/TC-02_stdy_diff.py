import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import pinv
import os


# -------------------------------------------------------------------------------------------------------------------
#  This code solves the 1D stationary diffusion equation (TC-2) with PIELM and PIBLS
# --------------------------------------------------------------------------------------------------------------------%

def generate_data(N_f, N_b):
    # PDE interior points (0 < x < 1)
    x_f = np.random.uniform(0, 1, N_f)

    # Boundary points (x=0 and x=1)
    x_b1 = np.zeros(N_b)  # x=0
    x_b2 = np.ones(N_b)  # x=1

    x_bc = np.concatenate([x_b1, x_b2])

    return x_f, x_bc


# ====================== TC-2  ======================
def exact_solution(x):
    return np.sin(np.pi * x / 2) * np.cos(2 * np.pi * x) + 1


def source(x):
    # Calculate second derivative of exact solution
    term1 = -np.pi ** 2 * np.sin(np.pi * x / 2) * np.cos(2 * np.pi * x) / 4
    term2 = -np.pi * np.cos(np.pi * x / 2) * 2 * np.pi * np.sin(2 * np.pi * x)
    term3 = -4 * np.pi ** 2 * np.sin(np.pi * x / 2) * np.cos(2 * np.pi * x)
    return term1 + term2 + term3


def plot_results(model_name, x_test, u_pred, exact_solution, save_dir=None):
    exact_sol = exact_solution(x_test)
    error = np.abs(u_pred - exact_sol)

    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Main plot: predicted solution, exact solution and error
    plt.figure(figsize=(12, 8))

    # Predicted solution and exact solution
    plt.subplot(2, 1, 1)
    plt.plot(x_test, u_pred, 'b-', linewidth=2, label=f'{model_name} Prediction')
    plt.plot(x_test, exact_sol, 'r--', linewidth=2, label='Exact Solution')
    plt.title(f'{model_name} Solution for 1D Diffusion Equation (TC-2)', fontsize=16)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Error plot
    plt.subplot(2, 1, 2)
    plt.plot(x_test, error, 'g-', linewidth=2)
    plt.title('Absolute Error', fontsize=16)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir is not None:
        main_path = os.path.join(save_dir, f"{model_name}_solution_and_error.png")
        plt.savefig(main_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved main results to {main_path}")

    plt.show()

    # Save predicted solution and exact solution separately
    plt.figure(figsize=(10, 6))
    plt.plot(x_test, u_pred, 'b-', linewidth=3, label=f'{model_name} Prediction')
    plt.plot(x_test, exact_sol, 'r--', linewidth=2, label='Exact Solution')
    plt.title(f'{model_name} Solution for 1D Diffusion Equation (TC-2)', fontsize=16)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('u(x)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    if save_dir is not None:
        path = os.path.join(save_dir, f"{model_name}_solution_comparison.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved solution comparison to {path}")

    plt.show()

    # Save error plot separately
    plt.figure(figsize=(10, 6))
    plt.plot(x_test, error, 'g-', linewidth=2)
    plt.title(f'{model_name} Absolute Error', fontsize=16)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    if save_dir is not None:
        path = os.path.join(save_dir, f"{model_name}_absolute_error.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved absolute error to {path}")

    plt.show()

    # Calculate relative error
    relative_error = np.linalg.norm(u_pred - exact_sol) / np.linalg.norm(exact_sol)
    print(f"{model_name} Relative Error: {relative_error:.4e}")

    return relative_error


# =====================================================
#                   PIELM (1D扩散方程版本)
# =====================================================

class PIELM:
    """Physics-Informed Extreme Learning Machine (1D diffusion equation version)"""

    def __init__(self, num_neurons, activation='tanh'):
        self.num_neurons = num_neurons
        self.activation_name = activation.lower()
        # Get activation function and its first and second derivatives
        self.activation, self.derivative, self.second_derivative = self._get_activation_functions()
        self.weights = None
        self.bias = None
        self.output_weights = None

    def _get_activation_functions(self):
        """Return activation function and its derivatives"""
        if self.activation_name == 'tanh':
            activation = lambda x: np.tanh(x)
            derivative = lambda x: 1 - np.tanh(x) ** 2
            second_derivative = lambda x: -2 * np.tanh(x) * (1 - np.tanh(x) ** 2)
        elif self.activation_name == 'sigmoid':
            activation = lambda x: 1 / (1 + np.exp(-x))
            derivative = lambda x: activation(x) * (1 - activation(x))
            second_derivative = lambda x: derivative(x) * (1 - 2 * activation(x))
        elif self.activation_name == 'relu':
            activation = lambda x: np.maximum(0, x)
            derivative = lambda x: np.where(x > 0, 1, 0)
            second_derivative = lambda x: np.zeros_like(x)  # Second derivative of ReLU is 0
        else:  # linear
            activation = lambda x: x
            derivative = lambda x: np.ones_like(x)
            second_derivative = lambda x: np.zeros_like(x)
        return activation, derivative, second_derivative

    def initialize_weights(self, input_dim=2):
        self.weights = np.random.randn(input_dim, self.num_neurons)
        self.bias = np.random.randn(1, self.num_neurons)

    def build_system(self, x_f, x_bc):
        if self.weights is None:
            self.initialize_weights(input_dim=2)

        # Add bias term
        X_f = np.column_stack([x_f, np.ones_like(x_f)])
        X_bc = np.column_stack([x_bc, np.ones_like(x_bc)])

        # Forward propagation
        Z_f = X_f @ self.weights + self.bias
        H_f = self.activation(Z_f)
        H_bc = self.activation(X_bc @ self.weights + self.bias)

        # Calculate second derivative terms
        ddH_f = self.second_derivative(Z_f)

        # Calculate second derivative terms
        d2H_dx2 = ddH_f * (self.weights[0, :] ** 2)

        # Diffusion equation: u_xx = R
        A_f = d2H_dx2
        b_f = source(x_f)

        # Boundary conditions
        A_bc = H_bc
        b_bc = exact_solution(x_bc)

        A = np.vstack([A_f, A_bc])
        b = np.concatenate([b_f, b_bc])

        return A, b

    def fit(self, x_f, x_bc):
        A, b = self.build_system(x_f, x_bc)
        self.output_weights = pinv(A) @ b.reshape(-1, 1)
        return self.output_weights

    def predict(self, x):
        X = np.column_stack([x, np.ones_like(x)])
        H = self.activation(X @ self.weights + self.bias)
        return (H @ self.output_weights).flatten()


class ...:
  
    def __init__(self, N1, N2, map_func='tanh', enhance_func='sigmoid'):
        self.N1 = int(N1)
        self.N2 = int(N2)

        # Get activation functions
        self.map_act_name, self.map_activation = self._get_activation(map_func)
        self.enhance_act_name, self.enhance_activation = self._get_activation(enhance_func)

        # Get derivative functions
        self.map_derivative = self._get_derivative(map_func)
        self.map_second_derivative = self._get_second_derivative(map_func)
        self.enhance_derivative = self._get_derivative(enhance_func)
        self.enhance_second_derivative = self._get_second_derivative(enhance_func)

...


# =====================================================
#                       main
# =====================================================
def main():
    np.random.seed(42)

    # Problem parameters
    Xmin, Xmax = 0, 1

    M_pde = 200  # Number of interior points
    M_bc = 50  # Number of points per boundary

    x_f, x_bc = generate_data(M_pde, M_bc)
    x_test = np.linspace(Xmin, Xmax, 500)

    # ====== PIELM ======
    print("\n" + "=" * 50)
    print("Training PIELM Model for 1D Diffusion Equation (TC-2)")
    print("=" * 50)

    pielm_model = PIELM(num_neurons=1000, activation='tanh')
    start_time = time.time()
    pielm_model.fit(x_f, x_bc)
    pielm_time = time.time() - start_time
    pielm_pred = pielm_model.predict(x_test)
    print(f"PIELM Training time: {pielm_time:.4f} seconds")

    pielm_error = plot_results("PIELM", x_test, pielm_pred, exact_solution, "pielm_tc2_results")


    print("\n" + "=" * 50)
    print("Training PIBLS Model for 1D Diffusion Equation (TC-2)")
    print("=" * 50)

    pibls_model = PIBLS(N1=300, N2=300, map_func='tanh', enhance_func='sigmoid')
    start_time = time.time()
    pibls_model.fit(x_f, x_bc)
    pibls_time = time.time() - start_time
    print(f"Training time: {pibls_time:.4f} seconds")

    pibls_pred = pibls_model.predict(x_test)
    pibls_error = plot_results("PIBLS", x_test, pibls_pred, exact_solution, "pibls_tc2_results")



if __name__ == "__main__":
    main()