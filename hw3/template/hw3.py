import hw3_utils
from hw3_utils import loss_batch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def svm_solver(x_train, y_train, lr, num_iters,
               kernel=hw3_utils.poly(degree=1), c=None):
    """An SVM solver.

    Arguments:
        x_train: a 2d tensor with shape (n, d).
        y_train: a 1d tensor with shape (n,), whose elememnts are +1 or -1.
        lr: the learning rate.
        num_iters: the number of gradient descent steps.
        kernel: the kernel function.
           The default kernel function is 1 + <x, y>.
        c: the trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Return:
        alpha: a 1d tensor with shape (n,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step. 
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    """
    alpha = nn.Linear(x_train.size(0), 1, bias=False)
    N = x_train.size(0)
    Q = torch.empty((N, N))
    with torch.no_grad():
        for i in range(N):
            for j in range(N):
                Q[i][j] = kernel(x_train[i], x_train[j]) * y_train[i] * y_train[j]
         
    def loss_func(alpha):
        return 0.5 * alpha(alpha(Q).transpose(1, 0)) - alpha.weight.sum()
    
    sgd = torch.optim.SGD(alpha.parameters(), lr=lr)
    
    for iter in range(num_iters):
        loss = loss_func(alpha)
        loss.backward()
        sgd.step()
        sgd.zero_grad()
    print(alpha.weight.shape)
    return alpha.weight


def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=hw3_utils.poly(degree=1)):
    """An SVM predictor.

    Arguments:
        alpha: a 1d tensor with shape (n,), denoting an optimal dual solution.
        x_train: a 2d tensor with shape (n, d), denoting the training set.
        y_train: a 1d tensor with shape (n,), whose elememnts are +1 or -1.
        x_test: a 2d tensor with shape (m, d), denoting the test set.
        kernel: the kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (m,), the outputs of SVM on the test set.
    """
    return torch.zeros(x_test.size(0))


def svm_contour(alpha, x_train, y_train, kernel,
                xmin=-5, xmax=5, ymin=-5, ymax=5, ngrid = 33):
    """Plot the contour lines of the svm predictor. """
    with torch.no_grad():
        xgrid = torch.linspace(xmin, xmax, ngrid)
        ygrid = torch.linspace(ymin, ymax, ngrid)
        (xx, yy) = torch.meshgrid((xgrid, ygrid))
        x_test = torch.cat(
            (xx.view(ngrid, ngrid, 1), yy.view(ngrid, ngrid, 1)),
            dim = 2).view(-1, 2)
        zz = svm_predictor(alpha, x_train, y_train, x_test, kernel)
        zz = zz.view(ngrid, ngrid)
        cs = plt.contour(xx.cpu().numpy(), yy.cpu().numpy(), zz.cpu().numpy(),
                        cmap = 'RdYlBu')
        plt.clabel(cs)
        plt.show()


class Block(nn.Module):
    """A basic block used to build ResNet."""

    def __init__(self, num_channels):
        """Initialize a building block for ResNet.

        Argument:
            num_channels: the number of channels of the input to Block, and
                          the number of channels of conv layers of Block.
        """
        super(Block, self).__init__()
        
        self.C = num_channels
        self.conv1 = nn.Conv2d(self.C, self.C, 3, padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(self.C)
        self.conv2 = nn.Conv2d(self.C, self.C, 3, padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(self.C)
        
    def forward(self, x):
        """
        The input will have shape (N, num_channels, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have the same shape as input.
        """

        z = F.relu(self.bn1(self.conv1(x)))
        z = F.relu(self.bn2(self.conv2(z)) + x)
        return z

    def set_param(self, kernel_1, bn1_weight, bn1_bias,
                  kernel_2, bn2_weight, bn2_bias):
        """Set the parameters of self using given arguments.

        Parameters of a Conv2d, BatchNorm2d, and Linear 
        are all given by attributes weight and bias.
        Note that you should wrap the arguments in nn.Parameter.

        Arguments (C denotes number of channels):
            kernel_1: a (C, C, 3, 3) tensor, kernels of the first conv layer.
            bn1_weight: a (C,) tensor.
            bn1_bias: a (C,) tensor.
            kernel_2: a (C, C, 3, 3) tensor, kernels of the second conv layer.
            bn2_weight: a (C,) tensor.
            bn2_bias: a (C,) tensor.
        """
        with torch.no_grad():
            self.conv1.weight = nn.Parameter(kernel_1)
            self.bn1.weight = nn.Parameter(bn1_weight)
            self.bn1.bias = nn.Parameter(bn1_bias)
            self.conv2.weight = nn.Parameter(kernel_2)
            self.bn2.weight = nn.Parameter(bn2_weight)
            self.bn2.bias = nn.Parameter(bn2_bias)


class ResNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self, num_channels, num_classes=10):
        """Initialize a shallow ResNet.

        Arguments:
            num_channels: the number of output channels of the conv layer
                          before the building block, and also 
                          the number of channels of the building block.
            num_classes: the number of output units.
        """
        super(ResNet, self).__init__()
        
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.conv0 = nn.Conv2d(1, self.num_channels, 3, stride=2, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(self.num_channels)
        self.max_pool = nn.MaxPool2d(2)
        self.block = Block(self.num_channels)
        self.adapt_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.num_channels, self.num_classes)
        
    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """
        N = x.shape[0]
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.adapt_avg_pool(self.block(self.max_pool(x))).view(N, self.num_channels)
        x = self.fc(x)
        return x

    def set_param(self, kernel_0, bn0_weight, bn0_bias,
                  kernel_1, bn1_weight, bn1_bias,
                  kernel_2, bn2_weight, bn2_bias,
                  fc_weight, fc_bias):
        """Set the parameters of self using given arguments.

        Parameters of a Conv2d, BatchNorm2d, and Linear 
        are all given by attributes weight and bias.
        Note that you should wrap the arguments in nn.Parameter.

        Arguments (C denotes number of channels):
            kernel_0: a (C, 1, 3, 3) tensor, kernels of the conv layer
                      before the building block.
            bn0_weight: a (C,) tensor, weight of the batch norm layer
                        before the building block.
            bn0_bias: a (C,) tensor, bias of the batch norm layer
                      before the building block.
            fc_weight: a (10, C) tensor
            fc_bias: a (10,) tensor
        See the docstring of Block.set_param() for the description
        of other arguments.
        """
        
        
        with torch.no_grad():
            self.conv0.weight = nn.Parameter(kernel_0)
            self.bn0.weight = nn.Parameter(bn0_weight)
            self.bn0.bias = nn.Parameter(bn0_bias)
            self.block.set_param(kernel_1, bn1_weight, bn1_bias, kernel_2, bn2_weight, bn2_bias)
            self.fc.weight = nn.Parameter(fc_weight)
            self.fc.bias = nn.Parameter(fc_bias)
            
            
def fit_and_validate(net, optimizer, loss_func, train, val, n_epochs, batch_size=1):
    """
    @param net: the neural network
    @param optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
    @param train: a torch.utils.data.Dataset
    @param val: a torch.utils.data.Dataset
    @param n_epochs: the number of epochs over which to do gradient descent
    @param batch_size: the number of samples to use in each batch of gradient descent
    @return train_epoch_loss, validation_epoch_loss: two arrays of length n_epochs+1, containing the mean loss at the beginning of training and after each epoch
    """
    net.eval() #put the net in evaluation mode
    train_dl = torch.utils.data.DataLoader(train, batch_size)
    val_dl = torch.utils.data.DataLoader(val)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)
    with torch.no_grad():
        # compute the mean loss on the training set at the beginning of iteration
        losses, nums = zip(*[loss_batch(net, loss_func, X, Y) for X, Y in train_dl])
        train_epoch_loss = [np.sum(np.multiply(losses, nums)) / np.sum(nums)]
        # TODO compute the validation loss and store it in a list
        losses, nums = zip(*[loss_batch(net, loss_func, X, Y) for X, Y in val_dl])
        validation_epoch_loss = [np.sum(np.multiply(losses, nums)) / np.sum(nums)]
        
    for _ in range(n_epochs):
        net.train() #put the net in train mode
        # TODO 
        for X, Y in train_dl:
            loss_batch(net, loss_func, X, Y, optimizer)
        with torch.no_grad():
            net.eval() #put the net in evaluation mode
            # TODO compute the train and validation losses and store it in a list
            losses, nums = zip(*[loss_batch(net, loss_func, X, Y) for X, Y in train_dl])
            train_epoch_loss.append(np.sum(np.multiply(losses, nums)) / np.sum(nums))
            losses, nums = zip(*[loss_batch(net, loss_func, X, Y) for X, Y in val_dl])
            validation_epoch_loss.append(np.sum(np.multiply(losses, nums)) / np.sum(nums))
        scheduler.step()
    return train_epoch_loss, validation_epoch_loss

    
# P3
# net = ResNet(16)
# loss_func = nn.CrossEntropyLoss()
# sgd = torch.optim.SGD(net.parameters(), lr=0.005)
# train, val = hw3_utils.torch_digits()
# train_epoch_loss, validation_epoch_loss = fit_and_validate(net, sgd, loss_func, train, val, n_epochs=30, batch_size=16)
# plt.plot(range(len(train_epoch_loss)), train_epoch_loss, label='train_loss')
# plt.plot(range(len(validation_epoch_loss)), validation_epoch_loss, label='val_loss')
# plt.legend()
# plt.show()