from hw6_utils import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class VAE(torch.nn.Module):
    def __init__(self, lam,lrate,latent_dim,loss_fn):
        """
        Initialize the layers of your neural network

        @param lam: Hyperparameter to scale KL-divergence penalty
        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param x - an (N,D) tensor
            @param y - an (N,D) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param latent_dim: The dimension of the latent space

        The network should have the following architecture (in terms of hidden units):
        Encoder Network:
        2 -> 50 -> ReLU -> 50 -> ReLU -> 50 -> ReLU -> (6,6) (mu_layer,logstd2_layer)

        Decoder Network:
        6 -> 50 -> ReLU -> 50 -> ReLU -> 2 -> Sigmoid

        See set_parameters() function for the exact shapes for each weight
        """
        super(VAE, self).__init__()

        self.lrate = lrate
        self.lam = lam
        self.loss_fn = loss_fn
        self.latent_dim = latent_dim
        
        self.fc_e1 = nn.Linear(2, 50)
        self.fc_e2 = nn.Linear(50, 50)
        self.fc_e3 = nn.Linear(50, 50)
        
        self.fc_mu = nn.Linear(50, 6)
        self.fc_std = nn.Linear(50, 6)
        
        self.fc_d1 = nn.Linear(6, 50)
        self.fc_d2 = nn.Linear(50, 50)
        self.fc_d3 = nn.Linear(50, 2)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lrate)

    def set_parameters(self, We1,be1, We2, be2, We3, be3, Wmu, bmu, Wstd, bstd, Wd1, bd1, Wd2, bd2, Wd3, bd3):
        """ Set the parameters of your network

        # Encoder weights:
        @param We1: an (50,2) torch tensor
        @param be1: an (50,) torch tensor
        @param We2: an (50,50) torch tensor
        @param be2: an (50,) torch tensor
        @param We3: an (50,50) torch tensor
        @param be3: an (50,) torch tensor
        @param Wmu: an (6,50) torch tensor
        @param bmu: an (6,) torch tensor
        @param Wstd: an (6,50) torch tensor
        @param bstd: an (6,) torch tensor

        # Decoder weights:
        @param Wd1: an (50,6) torch tensor
        @param bd1: an (50,) torch tensor
        @param Wd2: an (50,50) torch tensor
        @param bd2: an (50,) torch tensor
        @param Wd3: an (2,50) torch tensor
        @param bd3: an (2,) torch tensor

        """
        with torch.no_grad():
            self.fc_e1.weight = nn.Parameter(We1)
            self.fc_e1.bias = nn.Parameter(be1)
            self.fc_e2.weight = nn.Parameter(We2)
            self.fc_e2.bias = nn.Parameter(be2)
            self.fc_e3.weight = nn.Parameter(We3)
            self.fc_e3.bias = nn.Parameter(be3)
            self.fc_mu.weight = nn.Parameter(Wmu)
            self.fc_mu.bias = nn.Parameter(bmu)
            self.fc_std.weight = nn.Parameter(Wstd)
            self.fc_std.bias = nn.Parameter(bstd)
            self.fc_d1.weight = nn.Parameter(Wd1)
            self.fc_d1.bias = nn.Parameter(bd1)
            self.fc_d2.weight = nn.Parameter(Wd2)
            self.fc_d2.bias = nn.Parameter(bd2)
            self.fc_d3.weight = nn.Parameter(Wd3)
            self.fc_d3.bias = nn.Parameter(bd3)

    def forward(self, x):
        """ A forward pass of your autoencoder

        @param x: an (N, 2) torch tensor

        # return the following in this order from left to right:
        @return y: an (N, 50) torch tensor of output from the encoder network
        @return mean: an (N,latent_dim) torch tensor of output mu layer
        @return stddev_p: an (N,latent_dim) torch tensor of output stddev layer
        @return z: an (N,latent_dim) torch tensor sampled from N(mean,exp(stddev_p/2)
        @return xhat: an (N,D) torch tensor of outputs from f_dec(z)
        """
        N = x.shape[0]
        y = F.relu(self.fc_e3(F.relu(self.fc_e2(F.relu(self.fc_e1(x))))))
        mean = self.fc_mu(y)
        stddev_p = self.fc_std(y)
        z = torch.randn(N, self.latent_dim) * torch.exp(stddev_p / 2)+ mean 
        xhat = torch.sigmoid(self.fc_d3(F.relu(self.fc_d2(F.relu(self.fc_d1(z))))))
        
        
        return y, mean, stddev_p, z, xhat

    def decode(self, N):
        """ A forward pass of only the decode network
        @param N: number of output samples
        @return gen_samples: an (N, D) torch tensor of output
        """
        x = torch.randn(N, self.latent_dim)
        with torch.no_grad():
            gen_samples = torch.sigmoid(self.fc_d3(F.relu(self.fc_d2(F.relu(self.fc_d1(x))))))
        return gen_samples
        
    def step(self, x):
        """
        Performs one gradient step through a batch of data x
        @param x: an (N, 2) torch tensor

        # return the following in this order from left to right:
        @return L_rec: float containing the reconstruction loss at this time step
        @return L_kl: kl divergence penalty at this time step
        @return L: total loss at this time step
        """
        N = x.shape[0]
        y, mean, stddev_p, z, xhat = self.forward(x)
        
        self.optimizer.zero_grad()
        L_rec = self.loss_fn(x, xhat)
        L_kl = self.latent_dim + torch.sum(stddev_p - mean**2 - torch.exp(stddev_p), dim=1)
        assert(L_kl.shape[0] == N)
        L_kl = -torch.sum(L_kl) / 2 / N
        L = L_rec + self.lam * L_kl        
        L.backward()
        self.optimizer.step()
        # print(L_kl.shape)
        
        return L_rec, L_kl, L

def fit(net,X,n_iter):
    """ Fit a VAE.  Use the full batch size.
    @param net: the VAE
    @param X: an (N, D) torch tensor
    @param n_iter: int, the number of iterations of training

    # return all of these from left to right:

    @return losses_rec: Array of reconstruction losses at the beginning and after each iteration. Ensure len(losses_rec) == n_iter
    @return losses_kl: Array of KL loss penalties at the beginning and after each iteration. Ensure len(losses_kl) == n_iter
    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return Xhat: an (N,D) NumPy array of approximations to X
    @return gen_samples: an (N,D) NumPy array of N samples generated by the VAE
    """
    N = X.shape[0]
    losses_rec = []
    losses_kl = []
    losses = []
    
    for _ in range(n_iter):
        L_rec, L_kl, L = net.step(X)
        losses_rec.append(L_rec)
        losses_kl.append(L_kl)
        losses.append(L)
    assert(len(losses) == n_iter)
    with torch.no_grad():
        _, _, _, _, Xhat = net(X)
        gen_samples = net.decode(N)
    torch.save(net.cpu().state_dict(), "vae.pb")
    return losses_rec, losses_kl, losses, Xhat, gen_samples
    
    
    
def mse(x, y):
    return (x-y).pow(2).sum(1).mean()
    
def l1_loss(x, y):
    return torch.abs(x-y).sum(1).mean()
    
if __name__ == "__main__":
    data = generate_data()
    lam = 1
    lr = 0.001
    h = 6
    train_iter = 8000
    loss_fn = l1_loss
    
    net = VAE(lam, lr, h, loss_fn)
    _, _, losses, Xhat, gen_samples = fit(net, data, train_iter)
    
    t = np.linspace(0, train_iter-1, train_iter)
    
    plt.figure("Risk")
    plt.plot(t, losses)
    
    plt.figure("Data points and reconstruction")
    plt.scatter(data[:, 0], data[:, 1], marker="x", c="r")
    plt.scatter(Xhat[:, 0], Xhat[:, 1], marker="o", c="b")
    
    plt.figure("Data points and gen_samples")
    plt.scatter(data[:, 0], data[:, 1], marker="x", c="r")
    plt.scatter(gen_samples[:, 0], gen_samples[:, 1], marker="o", c="b")
    
    plt.show()
    
    