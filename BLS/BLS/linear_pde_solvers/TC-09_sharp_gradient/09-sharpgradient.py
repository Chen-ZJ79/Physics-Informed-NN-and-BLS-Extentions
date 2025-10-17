import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import pinv
import os
import warnings

# 忽略特定警告
warnings.filterwarnings('ignore', category=np.RankWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


# --------------------------------------------
#    TC-09: 1D Composite Function with Discontinuities and Sharp Gradients
# ---------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
#  This code uses PIELM and PIBLS to approximate a composite 1D function that contains
#  both discontinuous functions and functions with sharp gradients
# --------------------------------------------------------------------------------------------------------------------%

# ====================== 复合函数定义 ======================
def composite_function(x):

    # 方波 (-1 < x < -0.5)
    mask1 = (x >= -1) & (x < -0.5)
    y1 = np.where(mask1, 1.0, 0.0)

    # 高斯脉冲 (-0.5 <= x < 0)
    mask2 = (x >= -0.5) & (x < 0)
    y2 = np.exp(-500 * (x + 0.75) ** 2) * mask2

    # 分段线性函数 (0 <= x < 0.5)
    mask3 = (x >= 0) & (x < 0.5)
    y3 = np.where(x < 0.25, 2 * x, 2 - 2 * x) * mask3

    # 高频正弦波 (0.5 <= x <= 1)
    mask4 = (x >= 0.5) & (x <= 1)
    y4 = np.sin(40 * np.pi * x) * np.exp(-10 * (x - 0.75) ** 2) * mask4

    return y1 + y2 + y3 + y4


def generate_data(N_train, N_test, noise_level=0.0):

    # 训练数据 (在关键区域增加采样密度)
    x_train1 = np.linspace(-1, -0.5, N_train // 4)  # 方波区域
    x_train2 = np.linspace(-0.5, -0.2, N_train // 8)  # 高斯脉冲左侧
    x_train3 = np.linspace(-0.2, 0, N_train // 8)  # 高斯脉冲右侧
    x_train4 = np.linspace(0, 0.5, N_train // 4)  # 分段线性区域
    x_train5 = np.linspace(0.5, 0.7, N_train // 8)  # 高频正弦左侧
    x_train6 = np.linspace(0.7, 1, N_train // 8)  # 高频正弦右侧

    x_train = np.sort(np.concatenate([
        x_train1, x_train2, x_train3, x_train4, x_train5, x_train6
    ]))


    # 测试数据 (均匀分布)
    x_test = np.linspace(-1, 1, N_test)

    y_train = composite_function(x_train)
    y_test = composite_function(x_test)

    return x_train, y_train, x_test, y_test


def plot_results(model_name, x_train, y_train, x_test, y_test, y_pred, save_dir=None):
    plt.figure(figsize=(14, 10))
    plt.suptitle(f'{model_name} Approximation of Composite Function with Discontinuities and Sharp Gradients',
                 fontsize=16)

    plt.subplot(2, 2, 1)
    plt.plot(x_test, y_test, 'b-', linewidth=2.5, label='True Function')
    plt.plot(x_test, y_pred, 'r--', linewidth=1.8, label=f'{model_name} Prediction')
    plt.scatter(x_train, y_train, s=15, c='g', alpha=0.6, label='Training Points')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Function Approximation')
    plt.legend()
    plt.grid(True, alpha=0.3)

    error = np.abs(y_pred - y_test)
    plt.subplot(2, 2, 2)
    plt.plot(x_test, error, 'm-', linewidth=1.8)
    plt.xlabel('x')
    plt.ylabel('Absolute Error')
    plt.title('Approximation Error')
    plt.grid(True, alpha=0.3)

    # 方波和高斯脉冲
    plt.subplot(2, 2, 3)
    plt.plot(x_test, y_test, 'b-', linewidth=2.5, label='True Function')
    plt.plot(x_test, y_pred, 'r--', linewidth=1.8, label=f'{model_name} Prediction')
    plt.scatter(x_train, y_train, s=15, c='g', alpha=0.6, label='Training Points')
    plt.xlim([-1.0, 0.0])
    plt.ylim([-0.2, 1.2])
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Zoom: Square Wave and Gaussian Pulse')
    plt.grid(True, alpha=0.3)

    # 分段线性和高频正弦
    plt.subplot(2, 2, 4)
    plt.plot(x_test, y_test, 'b-', linewidth=2.5, label='True Function')
    plt.plot(x_test, y_pred, 'r--', linewidth=1.8, label=f'{model_name} Prediction')
    plt.scatter(x_train, y_train, s=15, c='g', alpha=0.6, label='Training Points')
    plt.xlim([0.0, 1.0])
    plt.ylim([-1.2, 1.2])
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Zoom: Piecewise Linear and High-Frequency Sine')
    plt.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        path = os.path.join(save_dir, f"{model_name}_results.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved results to {path}")
    plt.show()

    mse = np.mean((y_pred - y_test) ** 2)
    max_error = np.max(np.abs(y_pred - y_test))
    print(f"{model_name} Mean Squared Error: {mse:.4e}")
    print(f"{model_name} Maximum Error: {max_error:.4e}")

    plt.figure(figsize=(10, 6))
    plt.plot(x_test, error, 'm-', linewidth=1.8)
    plt.xlabel('x')
    plt.ylabel('Absolute Error')
    plt.title(f'{model_name} Approximation Error')
    plt.grid(True, alpha=0.3)

    if save_dir is not None:
        path = os.path.join(save_dir, f"{model_name}_error_plot.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

    return mse, max_error


# =====================================================
#                   PIELM
# =====================================================

class PIELM:
    def __init__(self, num_neurons, activation='tanh', regularization=None, alpha=1e-6):
        self.num_neurons = num_neurons
        self.activation_name = activation.lower()
        self.regularization = regularization
        self.alpha = alpha
        self.activation, _ = self._get_activation_functions()
        self.weights = None
        self.bias = None
        self.output_weights = None

    def _get_activation_functions(self):
        if self.activation_name == 'tanh':
            activation = lambda x: np.tanh(x)
        elif self.activation_name == 'sigmoid':
            activation = lambda x: 1 / (1 + np.exp(-x))
        elif self.activation_name == 'relu':
            activation = lambda x: np.maximum(0, x)
        elif self.activation_name == 'elu':
            activation = lambda x: np.where(x > 0, x, np.exp(x) - 1)
        elif self.activation_name == 'softplus':
            activation = lambda x: np.log(1 + np.exp(x))
        elif self.activation_name == 'gaussian':
            activation = lambda x: np.exp(-x ** 2)
        else:
            activation = lambda x: x
        return activation, None

    def initialize_weights(self, input_dim=1):
        self.weights = np.random.randn(input_dim, self.num_neurons)
        self.bias = np.random.randn(1, self.num_neurons)

    def build_system(self, X, y):
        if self.weights is None:
            self.initialize_weights(input_dim=X.shape[1])

        X_bias = np.column_stack([X, np.ones((X.shape[0], 1))])
        Z = X_bias @ np.vstack([self.weights, self.bias])
        H = self.activation(Z)

        return H, y

    def fit(self, X, y):
        H, y = self.build_system(X, y)

        if self.regularization == 'l2':
            # 岭回归
            A = H.T @ H + self.alpha * np.eye(H.shape[1])
            b = H.T @ y.reshape(-1, 1)
            self.output_weights = np.linalg.solve(A, b)
        elif self.regularization == 'l1':
            # LASSO
            from sklearn.linear_model import Lasso
            lasso = Lasso(alpha=self.alpha, max_iter=10000, tol=1e-4)
            lasso.fit(H, y)
            self.output_weights = lasso.coef_.reshape(-1, 1)
        else:
            # 伪逆
            self.output_weights = pinv(H) @ y.reshape(-1, 1)

        return self.output_weights

    def predict(self, X):
        if self.output_weights is None:
            raise ValueError("Model not trained. Call fit() first.")

        X_bias = np.column_stack([X, np.ones((X.shape[0], 1))])
        Z = X_bias @ np.vstack([self.weights, self.bias])
        H = self.activation(Z)

        return (H @ self.output_weights).flatten()


# =====================================================
#                   PIBLS
# =====================================================

class PIBLS:

    def __init__(self, N1, N2, map_func='tanh', enhance_func='sigmoid',
                 regularization=None, alpha=1e-6, use_sparse_init=True):
        self.N1 = int(N1)
        self.N2 = int(N2)
        self.regularization = regularization
        self.alpha = alpha
        self.use_sparse_init = use_sparse_init

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
            'elu': ('elu', lambda x: np.where(x > 0, x, np.exp(x) - 1)),
            'softplus': ('softplus', lambda x: np.log(1 + np.exp(x))),
            'gaussian': ('gaussian', lambda x: np.exp(-x ** 2)),
            'linear': ('linear', lambda x: x)
        }
        return activations.get(activation.lower(), activations['tanh'])

    def _build_features(self, X):
        """构建特征"""
        if not self.is_initialized:
            self._initialize_weights(X)

        X_bias = np.column_stack([X, np.ones((X.shape[0], 1))])

        Z_map = X_bias @ self.W_map + self.B_map
        H_map = self.map_activation(Z_map)
        Z_enhance = H_map @ self.W_enhance + self.B_enhance
        H_enhance = self.enhance_activation(Z_enhance)

        return np.hstack([H_map, H_enhance])

    def _initialize_weights(self, X):
        X_bias = np.column_stack([X, np.ones((X.shape[0], 1))])

        if self.use_sparse_init:
            init_W = np.random.randn(X_bias.shape[1], self.N1)
            self.W_map = self.sparse_bls(X_bias, X_bias @ init_W)
        else:
            self.W_map = np.random.randn(X_bias.shape[1], self.N1)

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

    def fit(self, X, y):
        H = self._build_features(X)

        if self.regularization == 'l2':
            A = H.T @ H + self.alpha * np.eye(H.shape[1])
            b = H.T @ y.reshape(-1, 1)
            self.beta = np.linalg.solve(A, b)
        elif self.regularization == 'l1':
            from sklearn.linear_model import Lasso
            lasso = Lasso(alpha=self.alpha, max_iter=10000, tol=1e-4)
            lasso.fit(H, y)
            self.beta = lasso.coef_.reshape(-1, 1)
        else:
            self.beta = pinv(H) @ y.reshape(-1, 1)

        return self.beta

    def predict(self, X):
        if self.beta is None:
            raise ValueError("Model not trained. Call fit() first.")

        H = self._build_features(X)
        return (H @ self.beta).flatten()

#
# # =====================================================
# #                   Baseline Models
# # =====================================================
#
# class PolynomialRegression:
#     """多项式回归模型"""
#
#     def __init__(self, degree=5):
#         self.degree = degree
#         self.coef_ = None
#
#     def fit(self, X, y):
#         # 创建多项式特征
#         X_poly = np.vander(X.flatten(), self.degree + 1, increasing=True)
#
#         # 最小二乘拟合
#         self.coef_ = np.linalg.lstsq(X_poly, y, rcond=None)[0]
#
#     def predict(self, X):
#         if self.coef_ is None:
#             raise ValueError("Model not trained. Call fit() first.")
#
#         # 创建多项式特征
#         X_poly = np.vander(X.flatten(), self.degree + 1, increasing=True)
#         return X_poly @ self.coef_


# class FourierRegression:
#     """傅里叶回归模型"""
#
#     def __init__(self, n_terms=20):
#         self.n_terms = n_terms
#         self.coef_ = None
#
#     def fit(self, X, y):
#         # 创建傅里叶特征
#         X_fourier = np.ones((len(X), 2 * self.n_terms + 1))
#
#         for k in range(1, self.n_terms + 1):
#             X_fourier[:, 2 * k - 1] = np.sin(2 * np.pi * k * X.flatten())
#             X_fourier[:, 2 * k] = np.cos(2 * np.pi * k * X.flatten())
#
#         # 最小二乘拟合
#         self.coef_ = np.linalg.lstsq(X_fourier, y, rcond=None)[0]
#
#     def predict(self, X):
#         if self.coef_ is None:
#             raise ValueError("Model not trained. Call fit() first.")
#
#         # 创建傅里叶特征
#         X_fourier = np.ones((len(X), 2 * self.n_terms + 1))
#
#         for k in range(1, self.n_terms + 1):
#             X_fourier[:, 2 * k - 1] = np.sin(2 * np.pi * k * X.flatten())
#             X_fourier[:, 2 * k] = np.cos(2 * np.pi * k * X.flatten())
#
#         return X_fourier @ self.coef_


# =====================================================
#                       main
# =====================================================
def main():
    np.random.seed(42)

    N_train = 1000
    N_test = 2000
    noise_level = 0.01

    x_train, y_train, x_test, y_test = generate_data(N_train, N_test, noise_level)

    x_train = x_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)

    print(f"Generated {len(x_train)} training points and {len(x_test)} test points")
    print(f"Function contains discontinuities and sharp gradients")

    # ====== PIELM ======
    print("\n" + "=" * 70)
    print("Training PIELM Model for Composite Function Approximation (TC-09)")
    print("=" * 70)

    pielm_model = PIELM(
        num_neurons=500,
        activation='tanh',
        regularization='l2',
        alpha=1e-4
    )

    start_time = time.time()
    pielm_model.fit(x_train, y_train)
    pielm_time = time.time() - start_time
    pielm_pred = pielm_model.predict(x_test)
    print(f"PIELM Training time: {pielm_time:.4f} seconds")

    pielm_mse, pielm_max_error = plot_results(
        "PIELM",
        x_train, y_train,
        x_test, y_test,
        pielm_pred,
        "pielm_tc09_results"
    )

    # ====== PIBLS ======
    print("\n" + "=" * 70)
    print("Training PIBLS Model for Composite Function Approximation (TC-09)")
    print("=" * 70)

    pibls_model = PIBLS(
        N1=200,
        N2=300,
        map_func='tanh',
        enhance_func='gaussian',  # 高斯激活函数适合尖锐梯度
        regularization='l2',
        alpha=1e-4,
        use_sparse_init=True
    )

    start_time = time.time()
    pibls_model.fit(x_train, y_train)
    pibls_time = time.time() - start_time
    pibls_pred = pibls_model.predict(x_test)
    print(f"PIBLS Training time: {pibls_time:.4f} seconds")

    pibls_mse, pibls_max_error = plot_results(
        "PIBLS",
        x_train, y_train,
        x_test, y_test,
        pibls_pred,
        "pibls_tc09_results"
    )

    # # ====== 多项式回归 (基线) ======
    # print("\n" + "=" * 70)
    # print("Training Polynomial Regression Model (Baseline)")
    # print("=" * 70)
    #
    # poly_model = PolynomialRegression(degree=15)
    # start_time = time.time()
    # poly_model.fit(x_train, y_train)
    # poly_time = time.time() - start_time
    # poly_pred = poly_model.predict(x_test)
    # print(f"Polynomial Regression Training time: {poly_time:.4f} seconds")
    #
    # poly_mse, poly_max_error = plot_results(
    #     "Polynomial",
    #     x_train, y_train,
    #     x_test, y_test,
    #     poly_pred,
    #     "poly_tc09_results"
    # )

    # # ====== 傅里叶回归 (基线) ======
    # print("\n" + "=" * 70)
    # print("Training Fourier Regression Model (Baseline)")
    # print("=" * 70)
    #
    # fourier_model = FourierRegression(n_terms=50)
    # start_time = time.time()
    # fourier_model.fit(x_train, y_train)
    # fourier_time = time.time() - start_time
    # fourier_pred = fourier_model.predict(x_test)
    # print(f"Fourier Regression Training time: {fourier_time:.4f} seconds")
    #
    # fourier_mse, fourier_max_error = plot_results(
    #     "Fourier",
    #     x_train, y_train,
    #     x_test, y_test,
    #     fourier_pred,
    #     "fourier_tc09_results"
    # )

    # ====== 模型对比 ======
    print("\n" + "=" * 70)
    print("Performance Comparison for TC-09")
    print("=" * 70)
    print(f"{'Model':<20} {'Training Time (s)':<20} {'MSE':<20} {'Max Error':<20}")
    print(f"{'-' * 70}")
    print(f"{'PIELM':<20} {pielm_time:<20.6f} {pielm_mse:<20.4e} {pielm_max_error:<20.4e}")
    print(f"{'PIBLS':<20} {pibls_time:<20.6f} {pibls_mse:<20.4e} {pibls_max_error:<20.4e}")
    # print(f"{'Polynomial':<20} {poly_time:<20.6f} {poly_mse:<20.4e} {poly_max_error:<20.4e}")
    # print(f"{'Fourier':<20} {fourier_time:<20.6f} {fourier_mse:<20.4e} {fourier_max_error:<20.4e}")

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    plt.plot(x_test, y_test, 'k-', linewidth=3, label='True Function')
    plt.plot(x_test, pielm_pred, 'b-', linewidth=1.5, alpha=0.8, label='PIELM')
    plt.plot(x_test, pibls_pred, 'r-', linewidth=1.5, alpha=0.8, label='PIBLS')
    # plt.plot(x_test, poly_pred, 'g--', linewidth=1.2, alpha=0.7, label='Polynomial')
    # plt.plot(x_test, fourier_pred, 'm-.', linewidth=1.2, alpha=0.7, label='Fourier')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Function Approximation Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # # 误差比较
    # plt.subplot(2, 1, 2)
    # plt.semilogy(x_test, np.abs(pielm_pred - y_test), 'b-', label='PIELM Error')
    # plt.semilogy(x_test, np.abs(pibls_pred - y_test), 'r-', label='PIBLS Error')
    # plt.semilogy(x_test, np.abs(poly_pred - y_test), 'g--', label='Polynomial Error')
    # plt.semilogy(x_test, np.abs(fourier_pred - y_test), 'm-.', label='Fourier Error')
    # plt.xlabel('x')
    # plt.ylabel('Absolute Error (log scale)')
    # plt.title('Error Comparison (Log Scale)')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    #
    # plt.tight_layout()
    # plt.savefig('tc09_comparison.png', dpi=300, bbox_inches='tight')
    # plt.show()
    #
    # # 模型指标对比
    # models = ['PIELM', 'PIBLS', 'Polynomial', 'Fourier']
    # mse_values = [pielm_mse, pibls_mse, poly_mse, fourier_mse]
    # max_errors = [pielm_max_error, pibls_max_error, poly_max_error, fourier_max_error]
    # times = [pielm_time, pibls_time, poly_time, fourier_time]
    #
    # fig, ax1 = plt.subplots(figsize=(12, 8))

    # # MSE 和 Max Error
    # ax1.set_xlabel('Model')
    # ax1.set_ylabel('Error (log scale)')
    # ax1.set_yscale('log')
    # ax1.bar(np.arange(len(models)) - 0.2, mse_values, 0.4, color='b', label='MSE')
    # ax1.bar(np.arange(len(models)) + 0.2, max_errors, 0.4, color='r', label='Max Error')
    # ax1.set_xticks(np.arange(len(models)))
    # ax1.set_xticklabels(models)
    # ax1.legend(loc='upper left')
    #
    # # 训练时间
    # ax2 = ax1.twinx()
    # ax2.plot(np.arange(len(models)), times, 'go-', linewidth=2, markersize=8, label='Training Time')
    # ax2.set_ylabel('Training Time (s)')
    # ax2.legend(loc='upper right')
    #
    # plt.title('Model Performance Comparison for TC-09')
    # plt.tight_layout()
    # plt.savefig('tc09_performance_comparison.png', dpi=300, bbox_inches='tight')
    # plt.show()
    #

if __name__ == "__main__":
    main()