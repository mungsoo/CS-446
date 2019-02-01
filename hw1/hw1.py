import numpy as np
import hw1_utils as utils
import matplotlib.pyplot as plt
import torch
# Problem 2
def linear_gd(X,Y,lrate=0.1,num_iter=1000):
    w = np.zeros(X.shape[1]+1)
    X = np.insert(X, 0, 1, 1)
    for i in range(num_iter):
        dw = (X.transpose() @ (X @ w - Y)) / X.shape[0]
        w -= lrate*dw
    # return parameters as numpy array
    return w

def linear_normal(X,Y):
    X = np.insert(X, 0, 1, 1)
    # return parameters as numpy array
    return np.linalg.pinv(X.transpose() @ X) @ X.transpose() @ Y

def plot_linear():
    # return plot
    X, Y = utils.load_reg_data()
    w = linear_normal(X, Y)
    x_min, x_max = X.min(), X.max()
    x = np.linspace(x_min, x_max, num=10000)
    x = np.insert(x, 0, 1, 1)
    fig = plt.figure()
    plt.plot(x[:1], x @ w)
    plt.scatter(X, Y, c='r')
    return fig

# Problem 4
def poly_gd(X,Y,lrate=0.01,num_iter=3000):
    # return parameters as numpy array
    length = X.shape[1]
    for i in range(length):
        for j in range(i, length):
            X = np.insert(X, X.shape[1], X[:,i]*X[:,j], 1)

    Y = Y.reshape(Y.shape[0])
    w = linear_gd(X, Y, lrate, num_iter)
    return w

def poly_normal(X,Y):
    # return parameters as numpy array
    length = X.shape[1]
    for i in range(length):
        for j in range(i, length):
            X = np.insert(X, X.shape[1], X[:,i]*X[:,j], 1)

    Y = Y.reshape(Y.shape[0])
    w = linear_normal(X, Y)
    return w

def plot_poly():
    # return plot
    X, Y = utils.load_reg_data()
    length = X.shape[1]
    X_p = X.copy()
    for i in range(length):
        for j in range(i, length):
            X_p = np.insert(X_p, X_p.shape[1], X_p[:,i]*X_p[:,j], 1)
    
    w = linear_normal(X_p, Y)
    
    x_min, x_max = X.min(), X.max()
    x = np.linspace(x_min, x_max, num=10000).reshape((-1,1))
    print(x.shape)
    for i in range(1):
        for j in range(i, 1):
            x = np.insert(x, 1, x[:,i]*x[:,j], 1)
    x = np.insert(x, 0, 1, 1)
    fig = plt.figure()
    print(x.shape)
    plt.plot(x[:,1], x @ w)
    plt.scatter(X, Y, c='r')
    return fig

def poly_xor():
    # return labels for XOR from linear,polynomal models
    X, Y = utils.load_xor_data
    w_p = poly_normal(X, Y)
    w_l = linear_normal(X, Y)
    
    X_l = np.insert(X, 0, 1, 1)
    y_linear = X_l @ w_l
    
    length = X.shape[1]
    for i in range(length):
        for j in range(i, length):
            X = np.insert(X, X.shape[1], X[:,i]*X[:,j], 1)
    X = np.insert(X, 0, 1, 1)   
    y_poly = X @ w_p

    return y_linear,y_poly
    
    
def get_linear_xor(grid):
    width = grid.shape[0]
    grid = grid.reshape((-1, 2))
    X, Y = utils.load_xor_data()
    w_l = linear_normal(X, Y)
    grid = np.insert(grid, 0, 1, 1)
    return (grid @ w_l).reshape((width, -1))

def get_poly_xor(grid):
    width = grid.shape[0]
    grid = grid.reshape((-1, 2))
    X, Y = utils.load_xor_data()
    w_p = poly_normal(X, Y)
    length = grid.shape[1]
    for i in range(length):
        for j in range(i, length):
            grid = np.insert(grid, grid.shape[1], grid[:,i]*grid[:,j], 1)
    grid = np.insert(grid, 0, 1, 1)
    return (grid @ w_p).reshape((width, -1))
    
    
# Problem 5
def nn(X,Y,X_test):
    # return labels for X_test as numpy array
    Y_test = []
    for x_test in X_test:
        idx = np.square(X - x_test).sum(axis=1).argmin()
        Y_test.append(Y[idx])
    return np.array(Y_test)

def nn_iris():
    X, Y = utils.load_iris_data()
    split_idx = int(X.shape[0]*0.3)
    X_test, Y_test = X[:split_idx], Y[:split_idx]
    X_train, Y_train = X[split_idx:], Y[split_idx:]
    Y_label = nn(X_train, Y_train, X_test)
    return Y_label[Y_label==Y_test].shape[0] / Y_label.shape[0]

# Problem 6

def loss(X, Y, w):
    return torch.log(1 + torch.exp(-Y * (X @ w)))
def logistic(X,Y,lrate=1,num_iter=3000):
    # return parameters as numpy array
    w = torch.zeros(X.shape[1], dtype=torch.double, requires_grad=True)
    X_t, Y_t = torch.from_numpy(X), torch.from_numpy(Y)
    
    for i in range(num_iter):
        l = loss(X_t, Y_t, w).mean()
        l.backward()
        
        with torch.no_grad():
            w -= lrate * w.grad
            w.grad.zero_()
            
    return w.detach().numpy()

def logistic_vs_ols():
    # return plot
    
    X, Y = utils.load_logistic_data()
    idx = np.where(Y==1)
    plt.scatter(X[idx,0], X[idx,1], c='r')
    idx = np.where(Y==-1)
    plt.scatter(X[idx,0], X[idx,1], c='b')
    
    x_min, x_max = X.min(), X.max()
    x = np.linspace(x_min, x_max, 1000)
    
    w_log = logistic(X, Y)
    plt.plot(x, (-w_log[0]/w_log[1])*x, label='logistic', c='gray')
    
    w_lin = linear_gd(X, Y)
    bias = w_lin[0]
    plt.plot(x, (-w_lin[1]/w_lin[2])*x-bias/w_lin[2], label='linear', c = 'g')
    
    plt.legend()
    plt.show()
    
logistic_vs_ols()