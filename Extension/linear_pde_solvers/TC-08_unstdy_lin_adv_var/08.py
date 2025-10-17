import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import pinv
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

#--------------------------------------------
#    Description of the Problem
#---------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
#  This code shows how to solve an unsteady linear advection equation with Spatially varying coefficient with PIELM
# --------------------------------------------------------------------------------------------------------------------%

def generate_data(M_pde,M_ic,M_bc):
    # PDE 残差点 (内部点)
    x_pde = np.random.uniform(-1, 1, M_pde)
    t_pde = np.random.uniform(0, 0.5, M_pde)

    # 初始条件点 (t=0)
    x_ic = np.random.uniform(-1, 1, M_ic)
    t_ic = np.zeros(M_ic)

    # 边界条件点
    t_bc = np.random.uniform(0, 0.5, M_bc)

    # 流入边界: u(-1,t) = 0
    x_bc = np.full(M_bc, -1.0)

    return (x_pde, t_pde), (x_ic, t_ic), (x_bc, t_bc)

def plot_results(model_name, xx, tt, u_pred, exact_solution, save_path=None):

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'{model_name} Solution for Advection Equation', fontsize=20)

    # 子图1：预测解
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf = ax1.plot_surface(xx, tt, u_pred, cmap='viridis', alpha=0.8)
    ax1.set_title(f'{model_name} Predicted Solution', fontsize=16)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('t', fontsize=12)
    ax1.set_zlabel('u(x,t)', fontsize=12)
    fig.colorbar(surf, ax=ax1, shrink=0.5)

    # 子图2：精确解
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    exact_sol = exact_solution(xx, tt)
    surf_exact = ax2.plot_surface(xx, tt, exact_sol, cmap='plasma', alpha=0.8)
    ax2.set_title('Exact Solution', fontsize=16)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('t', fontsize=12)
    ax2.set_zlabel('u(x,t)', fontsize=12)
    fig.colorbar(surf_exact, ax=ax2, shrink=0.5)

    # 子图3：不同时刻的解对比
    ax3 = fig.add_subplot(2, 2, (3, 4))
    time_indices = [0, tt.shape[0] // 2, tt.shape[0] - 1]
    colors = ['r', 'g', 'b']

    for i, idx in enumerate(time_indices):
        t_val = tt[idx, 0]

        pred_slice = u_pred[idx, :]
        exact_slice = exact_solution(xx[idx, :], t_val)

        ax3.plot(xx[idx, :], pred_slice, '--', color=colors[i], linewidth=2.5,
                 label=f'{model_name} t={t_val:.2f}')
        ax3.plot(xx[idx, :], exact_slice, '-', color=colors[i], linewidth=1.5,
                 label=f'Exact t={t_val:.2f}')

    ax3.set_title('Solution Comparison at Different Times', fontsize=16)
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('u(x,t)', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 保存整体图
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        base_dir = save_path.rsplit('.', 1)[0]  # 去掉文件后缀

        # 单独保存预测解子图
        fig_pred = plt.figure(figsize=(8, 6))
        ax_pred = fig_pred.add_subplot(111, projection='3d')
        surf_pred = ax_pred.plot_surface(xx, tt, u_pred, cmap='viridis', alpha=0.8)
        ax_pred.set_title(f'{model_name} Predicted Solution', fontsize=16)
        ax_pred.set_xlabel('x', fontsize=12)
        ax_pred.set_ylabel('t', fontsize=12)
        ax_pred.set_zlabel('u(x,t)', fontsize=12)
        fig_pred.colorbar(surf_pred, ax=ax_pred, shrink=0.5)
        fig_pred.savefig(f"{base_dir}_predicted.png", dpi=300, bbox_inches='tight')
        plt.close(fig_pred)

        # 单独保存精确解子图
        fig_exact = plt.figure(figsize=(8, 6))
        ax_exact = fig_exact.add_subplot(111, projection='3d')
        surf_exact2 = ax_exact.plot_surface(xx, tt, exact_sol, cmap='plasma', alpha=0.8)
        ax_exact.set_title('Exact Solution', fontsize=16)
        ax_exact.set_xlabel('x', fontsize=12)
        ax_exact.set_ylabel('t', fontsize=12)
        ax_exact.set_zlabel('u(x,t)', fontsize=12)
        fig_exact.colorbar(surf_exact2, ax=ax_exact, shrink=0.5)
        fig_exact.savefig(f"{base_dir}_exact.png", dpi=300, bbox_inches='tight')
        plt.close(fig_exact)

        # 单独保存不同时刻对比子图
        fig_comp = plt.figure(figsize=(8, 6))
        ax_comp = fig_comp.add_subplot(111)
        for i, idx in enumerate(time_indices):
            t_val = tt[idx, 0]
            pred_slice = u_pred[idx, :]
            exact_slice = exact_solution(xx[idx, :], t_val)
            ax_comp.plot(xx[idx, :], pred_slice, '--', color=colors[i], linewidth=2.5,
                         label=f'{model_name} t={t_val:.2f}')
            ax_comp.plot(xx[idx, :], exact_slice, '-', color=colors[i], linewidth=1.5,
                         label=f'Exact t={t_val:.2f}')
        ax_comp.set_title('Solution Comparison at Different Times', fontsize=16)
        ax_comp.set_xlabel('x', fontsize=12)
        ax_comp.set_ylabel('u(x,t)', fontsize=12)
        ax_comp.legend(fontsize=10)
        ax_comp.grid(True, alpha=0.3)
        fig_comp.savefig(f"{base_dir}_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig_comp)

    plt.show()

    relative_error = np.linalg.norm(u_pred - exact_sol) / np.linalg.norm(exact_sol)
    print(f"{model_name} Relative Error: {relative_error:.4e}")

    return relative_error


# =====================================================
#                   PIELM
# =====================================================

class PIELM:
    """物理信息极限学习机"""

    def __init__(self, num_neurons, activation='tanh'):
        self.num_neurons = num_neurons
        self.activation_name = activation.lower()
        self.activation, self.derivative = self._get_activation_functions()
        self.weights = None
        self.bias = None
        self.output_weights = None

    def _get_activation_functions(self):
        activations = {
            'tanh': (lambda x: np.tanh(x), lambda x: 1 - np.tanh(x) ** 2),
            'sigmoid': (lambda x: 1 / (1 + np.exp(-x)),
                        lambda x: (1 / (1 + np.exp(-x))) * (1 - 1 / (1 + np.exp(-x)))),
            'relu': (lambda x: np.maximum(0, x), lambda x: np.where(x > 0, 1, 0)),
            'linear': (lambda x: x, lambda x: np.ones_like(x))
        }
        return activations.get(self.activation_name, activations['tanh'])

    def initialize_weights(self, input_dim=3):
        self.weights = np.random.randn(input_dim, self.num_neurons)
        self.bias = np.random.randn(1, self.num_neurons)

    def build_system(self, pde_data, ic_data, bc_data):
        x_pde, t_pde = pde_data
        x_ic, t_ic = ic_data
        x_bc, t_bc = bc_data

        if self.weights is None:
            self.initialize_weights()

        X_pde = np.column_stack([x_pde, t_pde, np.ones_like(x_pde)])
        X_ic = np.column_stack([x_ic, t_ic, np.ones_like(x_ic)])
        X_lft = np.column_stack([np.full(len(t_bc) // 2, -1.0), t_bc[:len(t_bc) // 2], np.ones(len(t_bc) // 2)])
        X_ryt = np.column_stack([np.full(len(t_bc) // 2, 1.0), t_bc[len(t_bc) // 2:], np.ones(len(t_bc) // 2)])

        H_pde = self.activation(X_pde @ self.weights + self.bias)
        H_ic = self.activation(X_ic @ self.weights + self.bias)
        H_lft = self.activation(X_lft @ self.weights + self.bias)
        H_ryt = self.activation(X_ryt @ self.weights + self.bias)

        Z_pde = X_pde @ self.weights + self.bias
        dH_pde = self.derivative(Z_pde)
        dH_dx_pde = dH_pde * self.weights[0, :]
        dH_dt_pde = dH_pde * self.weights[1, :]

        # PDE：u_t + a * u_x = 0
        a_x = 1 + x_pde.reshape(-1, 1)  # a(x)
        A_pde = dH_dt_pde + a_x * dH_dx_pde

        # IC：u(x,0) = sin(πx)
        A_ic = H_ic
        b_ic = np.sin(np.pi * x_ic)

        # BC：u(-1,t) = 0
        X_bc = np.zeros_like(np.column_stack([x_bc, t_bc, np.ones_like(x_bc)]))
        b_bc = np.zeros_like(x_bc)
        A_bc = self.activation(X_bc @ self.weights + self.bias)

        A_matrix = np.vstack([A_pde, A_ic, A_bc])
        b_vector = np.concatenate([np.zeros(len(x_pde)), b_ic, b_bc])

        return A_matrix, b_vector

    def fit(self, pde_data, ic_data, bc_data):
        A, b = self.build_system(pde_data, ic_data, bc_data)
        self.output_weights = pinv(A) @ b.reshape(-1, 1)
        return self.output_weights

    def predict(self, x, t):
        if self.output_weights is None:
            raise ValueError("Model not trained. Call fit() first.")

        X = np.column_stack([x, t, np.ones_like(x)])
        H = self.activation(X @ self.weights + self.bias)

        return (H @ self.output_weights).flatten()


class model:

    def __init__(self, N1, N2, map_func='tanh', enhance_func='tanh'):
        self.N1 = int(N1)
        self.N2 = int(N2)

        self.map_act_name, self.map_activation = self._get_activation(map_func)
        self.enhance_act_name, self.enhance_activation = self._get_activation(enhance_func)

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

    def _activation_derivative(self, activation_name, z):
        derivatives = {
            'tanh': lambda x: 1 - np.tanh(x) ** 2,
            'relu': lambda x: np.where(x > 0, 1, 0),
            'sigmoid': lambda x: (1 / (1 + np.exp(-x))) * (1 - 1 / (1 + np.exp(-x))),
            'linear': lambda x: np.ones_like(x)
        }
        return derivatives[activation_name](z)

    def _build_features(self, x, t):
        X_bias = np.column_stack([x, t, np.ones_like(x)])
        if not self.is_initialized:
            self._initialize_weights(X_bias)

        Z_map = X_bias @ self.W_map + self.B_map
        H_map = self.map_activation(Z_map)
        Z_enhance = H_map @ self.W_enhance + self.B_enhance
        H_enhance = self.enhance_activation(Z_enhance)

        return np.hstack([H_map, H_enhance]), (Z_map, Z_enhance)

    def _initialize_weights(self, X_bias):
        # self.W_map = np.random.randn(3, self.N1)
        init_W = np.random.randn(3, self.N1)
        self.W_map = self.sparse_bls(X_bias, X_bias @ init_W)
        self.is_initialized = True

    def shrinkage(self, a, b):
        return np.sign(a) * np.maximum(np.abs(a) - b, 0)

    def sparse_bls(self, A, b):
        lam = 0.001  # 稀疏化系数
        itrs = 50  # 迭代次数
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

    def _compute_derivatives(self, x, t, z_values):
        Z_map, Z_enhance = z_values

        dH_map = self._activation_derivative(self.map_act_name, Z_map)
        dH_dx_map = dH_map * self.W_map[0, :]
        dH_dt_map = dH_map * self.W_map[1, :]

        dH_enhance = self._activation_derivative(self.enhance_act_name, Z_enhance)
        dH_dx_enhance = dH_enhance * (dH_dx_map @ self.W_enhance)
        dH_dt_enhance = dH_enhance * (dH_dt_map @ self.W_enhance)

        return (
            np.hstack([dH_dx_map, dH_dx_enhance]),
            np.hstack([dH_dt_map, dH_dt_enhance])
        )

    def build_system(self, pde_data, ic_data, bc_data):
        x_pde, t_pde = pde_data
        x_ic, t_ic = ic_data
        x_bc, t_bc = bc_data

        # PDE：u_t + a * u_x = 0
        H_pde, z_pde = self._build_features(x_pde, t_pde)
        dH_dx_pde, dH_dt_pde = self._compute_derivatives(x_pde, t_pde, z_pde)
        a_x = 1 + x_pde.reshape(-1, 1)  # a(x)
        A_pde = dH_dt_pde + a_x * dH_dx_pde


        # IC：u(x,0) = sin(πx)
        H_ic, _ = self._build_features(x_ic, np.zeros_like(x_ic))
        b_ic = np.sin(np.pi * x_ic)

        # BC: u(-1, t) = 0
        H_bc, _ = self._build_features(np.full(len(t_bc), -1.0), t_bc)
        A_bc = H_bc
        b_bc = np.zeros(len(t_bc))

        A_matrix = np.vstack([A_pde, H_ic, A_bc])
        b_vector = np.concatenate([np.zeros(len(x_pde)), b_ic, b_bc])

        return A_matrix, b_vector

    def fit(self, pde_data, ic_data, bc_data):
        A, b = self.build_system(pde_data, ic_data, bc_data)
        self.beta = pinv(A) @ b.reshape(-1, 1)
        return self.beta

    def predict(self, x, t):
        if self.beta is None:
            raise ValueError("Model not trained. Call fit() first.")

        H, _ = self._build_features(x, t)
        return (H @ self.beta).flatten()


# =====================================================
#                       main
# =====================================================

def main():
    np.random.seed(42)

    Xmin, Xmax = -1, 1
    Tmin, Tmax = 0, 0.5

    M_pde = 420  # PDE 残差点数
    M_ic = 21  # 初始条件点数
    M_bc = 10  # 边界条件点数

    exact_solution = lambda x, t: np.sin(np.pi * ((1 + x) * np.exp(-t) - 1))

    pde_data, ic_data, bc_data = generate_data(M_pde, M_ic, M_bc)
    x_test = np.linspace(Xmin, Xmax, 100)
    t_test = np.linspace(Tmin, Tmax, 20)
    xx, tt = np.meshgrid(x_test, t_test)

    # ================ PIELM  ================
    print("\n" + "=" * 40)
    print("Training PIELM Model")
    print("=" * 40)

    pielm_model = PIELM(num_neurons=440, activation='tanh')
    start_time = time.time()
    pielm_model.fit(pde_data, ic_data, bc_data)
    pielm_time = time.time() - start_time
    print(f"PIELM Training time: {pielm_time:.2f} seconds")

    pielm_pred = pielm_model.predict(xx.ravel(), tt.ravel()).reshape(xx.shape)
    plot_results("PIELM", xx, tt, pielm_pred, exact_solution, "pielm_solution.png")

    print("\n" + "=" * 40)
    print("Training Model")
    print("=" * 40)

    Our_model = model(N1=200, N2=100, map_func='tanh', enhance_func='sigmoid')
    start_time = time.time()
    Our_model.fit(pde_data, ic_data, bc_data)
    Our_model_time = time.time() - start_time
    print(f"Our_model Training time: {Our_model_time:.2f} seconds")

    Our_model_pred = Our_model.predict(xx.ravel(), tt.ravel()).reshape(xx.shape)
    plot_results("Our_model", xx, tt, Our_model_pred, exact_solution, "Our_model_solution.png")

    # ================ Results comparison ================
    print("\n" + "=" * 40)
    print("Performance Comparison")
    print("=" * 40)
    print(f"PIELM Training Time: {pielm_time:.4f} sec")
    print(f"Our_model Training Time: {Our_model_time:.4f} sec")

    exact_sol = exact_solution(xx, tt)
    pielm_error = np.linalg.norm(pielm_pred - exact_sol) / np.linalg.norm(exact_sol)
    Our_model_error = np.linalg.norm(Our_model_pred - exact_sol) / np.linalg.norm(exact_sol)

    print(f"\nPIELM Relative Error: {pielm_error:.4e}")
    print(f"Our_model Relative Error: {Our_model_error:.4e}")

    # plt.figure(figsize=(10, 6))
    # plt.bar(['PIELM', 'Our_model'], [pielm_error, Our_model_error], color=['blue', 'orange'])
    # plt.yscale('log')
    # plt.ylabel('Relative Error (log scale)')
    # plt.title('Model Performance Comparison')
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.savefig('error_comparison.png', dpi=300, bbox_inches='tight')
    # plt.show()
    #
    # plt.figure(figsize=(10, 6))
    # plt.bar(['PIELM', 'Our_model'], [pielm_time, Our_model_time], color=['green', 'red'])
    # plt.ylabel('Training Time (seconds)')
    # plt.title('Training Time Comparison')
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.savefig('time_comparison.png', dpi=300, bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    main()