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


# =====================================================
#                   PIBLS (1D扩散方程版本)
# =====================================================

class PIBLS:
    """Physics-Informed Broad Learning System (1D diffusion equation version)"""

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

        self.W_map = None
        self.B_map = np.random.randn(self.N1)
        self.W_enhance = np.random.randn(self.N1, self.N2)
        self.B_enhance = np.random.randn(self.N2)

        self.beta = None
        self.is_initialized = False

    def _get_activation(self, activation):
        """Get activation function"""
        activations = {
            'relu': ('relu', lambda x: np.maximum(0, x)),
            'tanh': ('tanh', lambda x: np.tanh(x)),
            'sigmoid': ('sigmoid', lambda x: 1 / (1 + np.exp(-x))),
            'linear': ('linear', lambda x: x)
        }
        return activations.get(activation.lower(), activations['tanh'])

    def _get_derivative(self, activation):
        """Get first derivative function"""
        derivatives = {
            'tanh': lambda x: 1 - np.tanh(x) ** 2,
            'relu': lambda x: np.where(x > 0, 1, 0),
            'sigmoid': lambda x: (1 / (1 + np.exp(-x))) * (1 - 1 / (1 + np.exp(-x))),
            'linear': lambda x: np.ones_like(x)
        }
        return derivatives.get(activation.lower(), derivatives['tanh'])

    def _get_second_derivative(self, activation):
        """Get second derivative function"""
        second_derivatives = {
            'tanh': lambda x: -2 * np.tanh(x) * (1 - np.tanh(x) ** 2),
            'relu': lambda x: np.zeros_like(x),
            'sigmoid': lambda x: (1 / (1 + np.exp(-x))) * (1 - 1 / (1 + np.exp(-x))) * (1 - 2 / (1 + np.exp(-x))),
            'linear': lambda x: np.zeros_like(x)
        }
        return second_derivatives.get(activation.lower(), second_derivatives['tanh'])

    def _build_features(self, x):
        # Add bias term
        X_bias = np.column_stack([x, np.ones_like(x)])
        if not self.is_initialized:
            self._initialize_weights(X_bias)

        # 映射层
        Z_map = X_bias @ self.W_map + self.B_map
        H_map = self.map_activation(Z_map)

        # 增强层
        Z_enhance = H_map @ self.W_enhance + self.B_enhance
        H_enhance = self.enhance_activation(Z_enhance)

        return np.hstack([H_map, H_enhance]), (Z_map, Z_enhance)

    def _initialize_weights(self, X_bias):
        init_W = np.random.randn(2, self.N1)
        self.W_map = self.sparse_bls(X_bias, X_bias @ init_W)
        self.is_initialized = True

    def shrinkage(self, a, b):
        return np.sign(a) * np.maximum(np.abs(a) - b, 0)

    def sparse_bls(self, A, b):
        lam = 0.001
        itrs = 50
        AA = A.T.dot(A)
        m = A.shape[1]
        n = b.shape[1]
        x1 = np.zeros([m, n])
        wk = ok = uk = x1
        L1 = np.linalg.inv(AA + np.eye(m))
        L2 = L1.dot(A.T).dot(b)
        for _ in range(itrs):
            ck = L2 + L1.dot(ok - uk)
            ok = self.shrinkage(ck + uk, lam)
            uk += ck - ok
            wk = ok
        return wk

    def _compute_derivatives(self, x, z_values):
        Z_map, Z_enhance = z_values

        # Mapping layer derivatives
        dH_map = self.map_derivative(Z_map)
        ddH_map = self.map_second_derivative(Z_map)

        dH_dx_map = dH_map * self.W_map[0, :]
        d2H_dx2_map = ddH_map * (self.W_map[0, :] ** 2)

        # Enhancement layer derivatives
        dH_enhance = self.enhance_derivative(Z_enhance)
        ddH_enhance = self.enhance_second_derivative(Z_enhance)

        # First derivative (chain rule)
        dH_dx_enhance = dH_enhance * (dH_dx_map @ self.W_enhance)

        # Second derivative (chain rule)
        d2H_dx2_enhance = ddH_enhance * (dH_dx_map @ self.W_enhance) ** 2 + \
                          dH_enhance * (d2H_dx2_map @ self.W_enhance)

        # Combine features
        d2H_dx2 = np.hstack([d2H_dx2_map, d2H_dx2_enhance])

        return d2H_dx2

    def build_system(self, x_f, x_bc):
        # Diffusion equation: u_xx = R
        H_pde, z_pde = self._build_features(x_f)
        d2H_dx2 = self._compute_derivatives(x_f, z_pde)

        A_pde = d2H_dx2
        b_pde = source(x_f)

        # Boundary conditions
        H_bc, _ = self._build_features(x_bc)
        b_bc = exact_solution(x_bc)

        A_matrix = np.vstack([A_pde, H_bc])
        b_vector = np.concatenate([b_pde, b_bc])

        return A_matrix, b_vector

    def fit(self, x_f, x_bc):
        A, b = self.build_system(x_f, x_bc)
        self.beta = pinv(A) @ b.reshape(-1, 1)
        return self.beta

    def predict(self, x):
        if self.beta is None:
            raise ValueError("Model not trained. Call fit() first.")
        H, _ = self._build_features(x)
        return (H @ self.beta).flatten()


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

    # ====== PIBLS ======
    print("\n" + "=" * 50)
    print("Training PIBLS Model for 1D Diffusion Equation (TC-2)")
    print("=" * 50)

    pibls_model = PIBLS(N1=300, N2=300, map_func='tanh', enhance_func='sigmoid')
    start_time = time.time()
    pibls_model.fit(x_f, x_bc)
    pibls_time = time.time() - start_time
    print(f"PIBLS Training time: {pibls_time:.4f} seconds")

    pibls_pred = pibls_model.predict(x_test)
    pibls_error = plot_results("PIBLS", x_test, pibls_pred, exact_solution, "pibls_tc2_results")

    # ====== Model Comparison ======
    print("\n" + "=" * 50)
    print("Performance Comparison for TC-2")
    print("=" * 50)
    print(f"PIELM Training Time: {pielm_time:.6f} sec")
    print(f"PIBLS Training Time: {pibls_time:.6f} sec")
    print(f"PIELM Relative Error: {pielm_error:.6e}")
    print(f"PIBLS Relative Error: {pibls_error:.6e}")

    # # Create comparison plot
    # plt.figure(figsize=(12, 8))
    #
    # # Solution comparison
    # plt.subplot(2, 1, 1)
    # plt.plot(x_test, pielm_pred, 'b-', linewidth=2, label='PIELM Prediction')
    # plt.plot(x_test, pibls_pred, 'g-', linewidth=2, label='PIBLS Prediction')
    # plt.plot(x_test, exact_solution(x_test), 'r--', linewidth=2, label='Exact Solution')
    # plt.title('Model Comparison for TC-2', fontsize=16)
    # plt.xlabel('x', fontsize=12)
    # plt.ylabel('u(x)', fontsize=12)
    # plt.legend(fontsize=12)
    # plt.grid(True, alpha=0.3)

    # # 误差对比
    # plt.subplot(2, 1, 2)
    # plt.plot(x_test, np.abs(pielm_pred - exact_solution(x_test)), 'b-', linewidth=2, label='PIELM Error')
    # plt.plot(x_test, np.abs(pibls_pred - exact_solution(x_test)), 'g-', linewidth=2, label='PIBLS Error')
    # plt.title('Error Comparison', fontsize=16)
    # plt.xlabel('x', fontsize=12)
    # plt.ylabel('Absolute Error', fontsize=12)
    # plt.yscale('log')
    # plt.legend(fontsize=12)
    # plt.grid(True, alpha=0.3)
    #
    # plt.tight_layout()
    # plt.savefig('tc2_comparison.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # # Error statistics comparison
    # plt.figure(figsize=(10, 6))
    # models = ['PIELM', 'PIBLS']
    # errors = [pielm_error, pibls_error]
    # times = [pielm_time, pibls_time]
    #
    # # 误差对比
    # plt.subplot(1, 2, 1)
    # plt.bar(models, errors, color=['blue', 'green'])
    # plt.yscale('log')
    # plt.ylabel('Relative Error (log scale)')
    # plt.title('Error Comparison')
    # plt.grid(axis='y', alpha=0.3)
    #
    # # 时间对比
    # plt.subplot(1, 2, 2)
    # plt.bar(models, times, color=['orange', 'red'])
    # plt.ylabel('Training Time (seconds)')
    # plt.title('Training Time Comparison')
    # plt.grid(axis='y', alpha=0.3)
    #
    # plt.tight_layout()
    # plt.savefig('tc2_error_time_comparison.png', dpi=300, bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    main()