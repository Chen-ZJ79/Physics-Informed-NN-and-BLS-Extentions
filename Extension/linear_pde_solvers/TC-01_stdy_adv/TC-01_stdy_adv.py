import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import pinv
import os


# --------------------------------------------
#   1D Steady Advection Equation (TC-1)
# ---------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
#  This code solves the 1D steady advection equation with PIELM 
# --------------------------------------------------------------------------------------------------------------------%

def generate_data_1d(N_f, N_b):
    # Interior points (0 < x ≤ 1)
    x_f = np.random.uniform(0, 1, N_f)

    # Boundary points (x=0)
    x_bc = np.zeros(N_b)

    return x_f, x_bc


# ====================== TC-1 ======================
def exact_solution(x):
    return np.sin(2 * np.pi * x) * np.cos(4 * np.pi * x) + 1


def source(x):
    return 2 * np.pi * np.cos(2 * np.pi * x) * np.cos(4 * np.pi * x) - \
        4 * np.pi * np.sin(2 * np.pi * x) * np.sin(4 * np.pi * x)


def plot_results_1d(model_name, x_test, u_pred, exact_solution, save_dir=None):
    exact_sol = exact_solution(x_test)
    error = np.abs(u_pred - exact_sol)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(x_test, exact_sol, 'b-', linewidth=3, label='Exact Solution')
    plt.plot(x_test, u_pred, 'r--', linewidth=2, label=f'{model_name} Prediction')
    plt.title(f'{model_name} Solution for 1D Advection Equation (TC-1)', fontsize=16)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(x_test, error, 'g-', linewidth=2)
    plt.title('Absolute Error', fontsize=16)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        path = os.path.join(save_dir, f"{model_name}_tc1_results.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved results to {path}")

    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x_test, error, 'g-', linewidth=2)
    plt.title(f'{model_name} Absolute Error (TC-1)', fontsize=16)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    if save_dir is not None:
        path = os.path.join(save_dir, f"{model_name}_tc1_error.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')

    plt.close()

    relative_error = np.linalg.norm(u_pred - exact_sol) / np.linalg.norm(exact_sol)
    print(f"{model_name} Relative Error: {relative_error:.4e}")

    return relative_error


# =====================================================
#                   PIELM
# =====================================================

class PIELM:
    def __init__(self, num_neurons, activation='tanh'):
        self.num_neurons = num_neurons
        self.activation_name = activation.lower()
        self.activation, self.derivative = self._get_activation_functions()
        self.weights = None
        self.bias = None
        self.output_weights = None

    def _get_activation_functions(self):
        if self.activation_name == 'tanh':
            activation = lambda x: np.tanh(x)
            derivative = lambda x: 1 - np.tanh(x) ** 2
        elif self.activation_name == 'sigmoid':
            activation = lambda x: 1 / (1 + np.exp(-x))
            derivative = lambda x: activation(x) * (1 - activation(x))
        elif self.activation_name == 'relu':
            activation = lambda x: np.maximum(0, x)
            derivative = lambda x: np.where(x > 0, 1, 0)
        else:  # linear
            activation = lambda x: x
            derivative = lambda x: np.ones_like(x)
        return activation, derivative

    def initialize_weights(self, input_dim=2):
        self.weights = np.random.randn(input_dim, self.num_neurons)
        self.bias = np.random.randn(1, self.num_neurons)

    def build_system(self, pde_data, bc_data):
        x_f = pde_data
        x_bc = bc_data

        if self.weights is None:
            self.initialize_weights(input_dim=2)

        X_f = np.column_stack([x_f, np.ones_like(x_f)])
        X_bc = np.column_stack([x_bc, np.ones_like(x_bc)])
        Z_f = X_f @ self.weights + self.bias
        H_f = self.activation(Z_f)
        H_bc = self.activation(X_bc @ self.weights + self.bias)

        dH_f = self.derivative(Z_f)
        dH_dx = dH_f * self.weights[0, :]

        # u_x = R
        A_f = dH_dx
        b_f = source(x_f)

        # BC
        A_bc = H_bc
        b_bc = exact_solution(x_bc)

        A = np.vstack([A_f, A_bc])
        b = np.concatenate([b_f, b_bc])

        return A, b

    def fit(self, pde_data, bc_data):
        A, b = self.build_system(pde_data, bc_data)
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
        self.map_act_name, self.map_activation = self._get_activation(map_func)
        self.enhance_act_name, self.enhance_activation = self._get_activation(enhance_func)
        self.map_derivative = self._get_derivative(map_func)
        self.enhance_derivative = self._get_derivative(enhance_func)

        self.W_map = None
        self.B_map = np.random.randn(self.N1)
        self.W_enhance = np.random.randn(self.N1, self.N2)
        self.B_enhance = np.random.randn(self.N2)

        self.beta = None
        self.is_initialized = False

    def _get_activation(self, activation):
        activations = {
            'relu': ('relu', lambda x: np.maximum(0, x)),
            'tanh': ('tanh', lambda x: np.tanh(x)),
            'sigmoid': ('sigmoid', lambda x: 1 / (1 + np.exp(-x))),
            'linear': ('linear', lambda x: x)
        }
        return activations.get(activation.lower(), activations['tanh'])
...

# =====================================================
#                       main
# =====================================================
def main():
    np.random.seed(42)

    Xmin, Xmax = 0, 1

    M_pde = 100  # Interior points
    M_bc = 2  # Boundary points

    x_f, x_bc = generate_data_1d(M_pde, M_bc)
    x_test = np.linspace(Xmin, Xmax, 200)

    # ====== PIELM ======
    print("\n" + "=" * 60)
    print("Training PIELM Model for 1D Advection Equation (TC-1)")
    print("=" * 60)

    pielm_model = PIELM(num_neurons=102, activation='tanh')
    start_time = time.time()
    pielm_model.fit(x_f, x_bc)
    pielm_time = time.time() - start_time
    pielm_pred = pielm_model.predict(x_test)
    print(f"PIELM Training time: {pielm_time:.2f} seconds")

    pielm_error = plot_results_1d("PIELM", x_test, pielm_pred, exact_solution, "pielm_tc1_results")


...


    # Create comparison plot
    plt.figure(figsize=(12, 8))

    # Solution comparison
    plt.subplot(2, 1, 1)
    plt.plot(x_test, exact_solution(x_test), 'k-', linewidth=3, label='Exact Solution')
    plt.plot(x_test, pielm_pred, 'b--', linewidth=2, label='PIELM Prediction')
    plt.title('Solution Comparison for TC-1', fontsize=16)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # # Error comparison
    # plt.subplot(2, 1, 2)
    # plt.plot(x_test, np.abs(pielm_pred - exact_solution(x_test)), 'b-', linewidth=2, label='PIELM Error')
    # plt.title('Error Comparison', fontsize=16)
    # plt.xlabel('x', fontsize=12)
    # plt.ylabel('Absolute Error', fontsize=12)
    # plt.yscale('log')
    # plt.legend(fontsize=12)
    # plt.grid(True, alpha=0.3)
    #
    # plt.tight_layout()
    # plt.savefig('tc1_comparison.png', dpi=300, bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    main()