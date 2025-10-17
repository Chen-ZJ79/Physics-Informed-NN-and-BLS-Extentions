import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import pinv
import os


# --------------------------------------------
#   1D Steady Advection Equation (TC-1)
# ---------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
#  This code solves the 1D steady advection equation with PIELM and PIBLS
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


# =====================================================
#                   PIBLS
# =====================================================

class PIBLS:

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

    def _get_derivative(self, activation):
        derivatives = {
            'tanh': lambda x: 1 - np.tanh(x) ** 2,
            'relu': lambda x: np.where(x > 0, 1, 0),
            'sigmoid': lambda x: (1 / (1 + np.exp(-x))) * (1 - 1 / (1 + np.exp(-x))),
            'linear': lambda x: np.ones_like(x)
        }
        return derivatives.get(activation.lower(), derivatives['tanh'])

    def _build_features(self, x):
        X_bias = np.column_stack([x, np.ones_like(x)])
        if not self.is_initialized:
            self._initialize_weights(X_bias)

        Z_map = X_bias @ self.W_map + self.B_map
        H_map = self.map_activation(Z_map)
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

        dH_map = self.map_derivative(Z_map)
        dH_dx_map = dH_map * self.W_map[0, :]
        dH_enhance = self.enhance_derivative(Z_enhance)
        dH_dx_enhance = dH_enhance * (dH_dx_map @ self.W_enhance)

        dH_dx = np.hstack([dH_dx_map, dH_dx_enhance])

        return dH_dx

    def build_system(self, pde_data, bc_data):
        x_pde = pde_data
        x_bc = bc_data

        # u_x = R
        H_pde, z_pde = self._build_features(x_pde)
        dH_dx = self._compute_derivatives(x_pde, z_pde)

        A_pde = dH_dx
        b_pde = source(x_pde)

        # BC
        H_bc, _ = self._build_features(x_bc)
        b_bc = exact_solution(x_bc)

        A_matrix = np.vstack([A_pde, H_bc])
        b_vector = np.concatenate([b_pde, b_bc])

        return A_matrix, b_vector

    def fit(self, pde_data, bc_data):
        A, b = self.build_system(pde_data, bc_data)
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

    # ====== PIBLS ======
    print("\n" + "=" * 60)
    print("Training PIBLS Model for 1D Advection Equation (TC-1)")
    print("=" * 60)

    pibls_model = PIBLS(N1=20, N2=20, map_func='tanh', enhance_func='sigmoid')
    start_time = time.time()
    pibls_model.fit(x_f, x_bc)
    pibls_time = time.time() - start_time
    print(f"PIBLS Training time: {pibls_time:.2f} seconds")

    pibls_pred = pibls_model.predict(x_test)
    pibls_error = plot_results_1d("PIBLS", x_test, pibls_pred, exact_solution, "pibls_tc1_results")

    # ====== Model Comparison ======
    print("\n" + "=" * 60)
    print("Performance Comparison for TC-1")
    print("=" * 60)
    print(f"PIELM Training Time: {pielm_time:.4f} sec")
    print(f"PIBLS Training Time: {pibls_time:.4f} sec")
    print(f"PIELM Relative Error: {pielm_error:.4e}")
    print(f"PIBLS Relative Error: {pibls_error:.4e}")

    # Create comparison plot
    plt.figure(figsize=(12, 8))

    # Solution comparison
    plt.subplot(2, 1, 1)
    plt.plot(x_test, exact_solution(x_test), 'k-', linewidth=3, label='Exact Solution')
    plt.plot(x_test, pielm_pred, 'b--', linewidth=2, label='PIELM Prediction')
    plt.plot(x_test, pibls_pred, 'r-.', linewidth=2, label='PIBLS Prediction')
    plt.title('Solution Comparison for TC-1', fontsize=16)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # # Error comparison
    # plt.subplot(2, 1, 2)
    # plt.plot(x_test, np.abs(pielm_pred - exact_solution(x_test)), 'b-', linewidth=2, label='PIELM Error')
    # plt.plot(x_test, np.abs(pibls_pred - exact_solution(x_test)), 'r-', linewidth=2, label='PIBLS Error')
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