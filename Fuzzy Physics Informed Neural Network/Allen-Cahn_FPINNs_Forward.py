import os
import random
import time
import torch
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
# import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
# from plotting import newfig, savefig
from FuzzyLayers import FuzzyLayer


# seed
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed = 1111
seed_torch(seed)

# CUDA support 
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# the deep neural network
class FPINN(torch.nn.Module):
    # def __init__(self, layers):
    def __init__(self, layers, fuzzy, linear):
        super(FPINN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layerDict = OrderedDict(layer_list)

        print(layerDict)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

        self.fuzzylayer = FuzzyLayer(2, fuzzy)
        self.layer1 = torch.nn.Linear(fuzzy + linear, 1)

    def forward(self, x):
        print(x.shape)
        out1 = self.layers(x)
        out2 = self.fuzzylayer(x)
        # print('out2:',out2.shape)
        print('shape-check',out1.shape,out2.shape)
        out = torch.cat([out1, out2], dim=1)
        out = self.layer1(out)

        return out


# the physics-guided neural network
class PhysicsInformedNN():
    def __init__(self, X, u, layers, fuzzy, linear, lb, ub):

        # boundary conditions
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        # data
        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        self.t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)


        # deep neural networks
        self.dnn = FPINN(layers, fuzzy, linear).to(device)

        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(), 
            lr=1.0,
            max_iter=200,
            max_eval=50000,
            history_size=50,  
            tolerance_grad=1e-5,  
            tolerance_change=1.0 * np.finfo(float).eps,  
            line_search_fn="strong_wolfe" 
        )

        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters())  #
        self.iter = 0

    def net_u(self, x, t):
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_f(self, x, t):
        """ The pytorch autograd version of calculating residual """
        u = self.net_u(x, t)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        f = u_t - 0.0001 * u_xx + 5 * (u ** 3) - 5 * u 
        return f

    def loss_func(self):
        start_time = time.time()

        u_pred = self.net_u(self.x, self.t)
        f_pred = self.net_f(self.x, self.t)



        loss = torch.mean((self.u - u_pred) ** 2) + torch.mean(f_pred ** 2)




        LossList.append(loss)

        self.optimizer.zero_grad()
        loss.backward()

        self.iter += 1
        if self.iter % 1 == 0:
            elapsed = time.time() - start_time

            print(
                'It: %d, Loss: %e, Time: %.2f' %
                (
                    self.iter,
                    loss.item(),
                    elapsed
                )
            )
        return loss

    def train(self, nIter):
        self.dnn.train()
        start_time = time.time()

        for epoch in range(nIter):
            u_pred = self.net_u(self.x, self.t)
            f_pred = self.net_f(self.x, self.t)
            loss = torch.mean((self.u - u_pred) ** 2) + torch.mean(f_pred ** 2)

            LossList.append(loss)

            self.optimizer_Adam.zero_grad()
            loss.backward()
            self.optimizer_Adam.step()

            if epoch % 1 == 0:
                elapsed = time.time() - start_time

                print(
                    'It: %d, Loss: %.3e, Time: %.2f' %
                    (
                        epoch,
                        loss.item(),
                        elapsed
                    )
                )
                start_time = time.time()

        # self.optimizer.step(self.loss_func)

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        u = self.net_u(x, t)
        f = self.net_f(x, t)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f



class PhysicsInformedNN_pinv():
    def __init__(self, X, u, layers, fuzzy, linear, lb, ub):
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        self.t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)

        self.dnn = FPINN(layers, fuzzy, linear).to(device)
        self.beta = None  # 解出来的系数参数

    def compute_derivatives(self, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)

        H = self.dnn(torch.cat([x, t], dim=1))  # shape: (N, D)
        H_t = torch.autograd.grad(H, t, torch.ones_like(H), create_graph=True, retain_graph=True)[0]
        H_x = torch.autograd.grad(H, x, torch.ones_like(H), create_graph=True, retain_graph=True)[0]
        H_xx = torch.autograd.grad(H_x, x, torch.ones_like(H_x), create_graph=True, retain_graph=True)[0]
        return H, H_t, H_xx

    def train(self):
        eps = 0.0001  # Allen-Cahn epsilon
        H, H_t, H_xx = self.compute_derivatives(self.x, self.t)
        J = H_t - eps * H_xx  # residual: f(x,t) = J(x,t) · β

        # 组成联合线性系统
        H_data = H               # u = H @ β
        y_data = self.u

        H_phys = J               # f = J @ β ≈ 0
        y_phys = torch.zeros_like(self.u)

        λ = 1.0  # 权重参数，可调
        H_all = torch.cat([H_data, λ * H_phys], dim=0)
        y_all = torch.cat([y_data, λ * y_phys], dim=0)

        # 最小二乘解
        self.beta = torch.linalg.pinv(H_all) @ y_all

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        H = self.dnn(torch.cat([x, t], dim=1))
        u = H @ self.beta
        print('!!!',H.shape,self.beta.shape,u.shape)
        return u.detach().cpu().numpy()





if __name__ == "__main__":

    N_u = 8000  # 初边值点的数量
    data = scipy.io.loadmat('dataset/burgers_shock.mat')
    t = data['t'].flatten()[:, None]  # shape:201*1
    print("t: ", t.shape)
    x = data['x'].flatten()[:, None]  # shape:512*1
    print("x: ", x.shape)
    Exact = np.real(data['usol']).T  # shape:201*512
    print("Exact: ", Exact.shape)
    X, T = np.meshgrid(x, t)  # 生成网格采样点矩阵  shape: X:201*512 T:201*512
    print("X: ", X.shape)
    print("T: ", T.shape)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    print("X_star: ", X_star.shape)
    u_star = Exact.flatten()[:, None]
    print("u_star: ", u_star.shape)
    # data = scipy.io.loadmat('dataset/AC.mat')

    # t = data['tt'].flatten()[:, None]  # shape:201*1
    # print("t: ", t.shape)
    # x = data['x'].flatten()[:, None]  # shape:512*1
    # print("x: ", x.shape)
    # Exact = np.real(data['uu']).T  # shape:201*512
    # print("Exact: ", Exact.shape)
    # X, T = np.meshgrid(x, t)  # 生成网格采样点矩阵  shape: X:201*512 T:201*512
    # print("X: ", X.shape)
    # print("T: ", T.shape)
    # X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    # print("X_star: ", X_star.shape)
    # u_star = Exact.flatten()[:, None]
    # print("u_star: ", u_star.shape)

    # Doman bounds
    lb = X_star.min(0)  
    ub = X_star.max(0)

    ####################### Training on Non-noisy Data #######################

    # time

    # noise = 0.0

    # # create training set
    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx, :]  
    print("X_u_train: ", X_u_train.shape)
    u_train = u_star[idx, :]  # label
    print("u_train: ", u_train.shape)

    ################## Training  on Noisy Data ####################

    error = []

    fuzzy = 4
    linear = 10
    LossList = []

    # training
    layers = [2, 200, 200, 200, 200, linear]  
    model = PhysicsInformedNN(X_u_train, u_train, layers, fuzzy, linear, lb, ub)
    time1 = time.time()
    model.train(1)
    time2 = time.time() - time1
    print('Total Time: ', time2)

    # evaluations
    u_pred, f_pred = model.predict(X_star)
    # u_pred = model.predict(X_star)

    print("u_star shape:", u_star.shape)  # 应为 (N,1) 或 (N,)
    print("u_pred shape:", u_pred.shape)  # 应为相同形状
    print("u_star sample:", u_star[:5])  # 查看前5个值
    print("u_pred sample:", u_pred[:5])  # 查看前5个值

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    U_star = griddata(X_star, u_star.flatten(), (X, T), method='cubic')
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

    print('Error u: %e' % (error_u))
    error.append(str(fuzzy) + ' ' + str(linear) + ' ' + str(error_u))
    # root1 = './loss/AC/Forward/FPINN' + '_' + str(fuzzy) + '_' + str(linear) + '.txt'
    # root2 = './predict/AC/Forward/FPINN' + '_' + str(fuzzy) + '_' + str(linear) + '.csv'
    # with open(root1, 'w') as f:
    #     for j in LossList:
    #         j = str(j.tolist())
    #         f.write(j + '\n')
    #
    # np.savetxt(root2, u_pred)
    #
    # with open('Allen-Cahn_FPINNs_Forward_Result.txt', 'w') as f:
    #     for i in error:
    #         f.write(i + '\n')
    #         # f.write('\r\n')
    import matplotlib.pyplot as plt

    # 设置画布
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # 真实解
    cf1 = ax[0].imshow(U_star.T, interpolation='nearest', cmap='rainbow',
                       extent=[t.min(), t.max(), x.min(), x.max()],
                       origin='lower', aspect='auto')
    ax[0].set_title('Exact $u(x,t)$')
    fig.colorbar(cf1, ax=ax[0])

    # 预测解
    cf2 = ax[1].imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
                       extent=[t.min(), t.max(), x.min(), x.max()],
                       origin='lower', aspect='auto')
    ax[1].set_title('Predicted $u(x,t)$')
    fig.colorbar(cf2, ax=ax[1])

    plt.tight_layout()
    plt.show()
