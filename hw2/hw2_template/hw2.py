import hw2_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from hw2_utils import loss_batch
import scipy
import time


class XORNet(nn.Module):
    def __init__(self):
        """
        Initialize the layers of your neural network

        You should use nn.Linear
        """
        super(XORNet, self).__init__()
        self.layer_1 = nn.Linear(2,2)
        self.layer_2 = nn.Linear(2,1)
        
    
    def set_l1(self, w, b):
        """
        Set the weights and bias of your first layer
        @param w: (2,2) torch tensor
        @param b: (2,) torch tensor
        """
        with torch.no_grad():
            self.layer_1.weight = nn.Parameter(w)
            self.layer_1.bias = nn.Parameter(b)
    
    def set_l2(self, w, b):
        """
        Set the weights and bias of your second layer
        @param w: (1,2) torch tensor
        @param b: (1,) torch tensor
        """
        with torch.no_grad():
            self.layer_2.weight = nn.Parameter(w)
            self.layer_2.bias = nn.Parameter(b)
        
    def forward(self, xb):
        """
        Compute a forward pass in your network.  Note that the nonlinearity should be F.relu.
        @param xb: The (n, 2) torch tensor input to your model
        @return: an (n, 1) torch tensor
        """
        l1_out = F.relu(self.layer_1(xb))
        l2_out = self.layer_2(l1_out)
        return l2_out
class DigitsConvNet(nn.Module):
    def __init__(self):
        """ Initialize the layers of your neural network

        You should use nn.Conv2d, nn.MaxPool2D, and nn.Linear
        The layers of your neural network (in order) should be
        1) a 2D convolutional layer with 1 input channel and 8 outputs, with a kernel size of 3, followed by 
        2) a 2D maximimum pooling layer, with kernel size 2
        3) a 2D convolutional layer with 8 input channels and 4 output channels, with a kernel size of 3
        4) a fully connected (Linear) layer with 4 inputs and 10 outputs
        """
        super(DigitsConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 4, 3)
        self.fc1 = nn.Linear(4, 10)
        
    def set_parameters(self, kern1, bias1, kern2, bias2, fc_weight, fc_bias):
        """ Set the parameters of your network

        @param kern1: an (8, 1, 3, 3) torch tensor
        @param bias1: an (8,) torch tensor
        @param kern2: an (4, 8, 3, 3) torch tensor
        @param bias2: an (4,) torch tensor
        @param fc_weight: an (10, 4) torch tensor
        @param fc_bias: an (10,) torch tensor
        """
        with torch.no_grad:
            self.conv1.weight = nn.Parameter(kern1)
            self.conv1.bias = nn.Parameter(bias1)
            self.conv2.weight = nn.Parameter(kern2)
            self.conv2.bias = nn.Parameter(bias2)
            self.fc1.weight = nn.Parameter(fc_weight)
            self.fc1.bias = nn.Parameter(fc_bias)

    def intermediate(self, xb):
        """ Return the feature representation your network lerans

        Note that the nonlinearity between each layer should be F.relu.  You
        may need to use a tensor's view() method to reshape outputs. Hint: this
        should be very similar to your forward method
        @param xb: an (N, 8, 8) torch tensor
        @return: an (N, 4) torch tensor
        """
        
        conv1 = F.relu(self.conv1(xb.view(xb.shape[0], 1, xb.shape[1], xb.shape[2])))
        max_pool1 = self.max_pool1(conv1)
        conv2 = F.relu(self.conv2(max_pool1)).view(-1, 4)
        return conv2

    def forward(self, xb):
        """ A forward pass of your neural network

        Note that the nonlinearity between each layer should be F.relu.  You
        may need to use a tensor's view() method to reshape outputs
        @param xb: an (N, 8, 8) torch tensor
        @return: an (N, 10) torch tensor
        """
        conv1 = F.relu(self.conv1(xb.view(xb.shape[0], 1, xb.shape[1], xb.shape[2])))
        max_pool1 = self.max_pool1(conv1)
        conv2 = F.relu(self.conv2(max_pool1))
        fc1 = self.fc1(conv2.view(xb.shape[0], -1))
        return fc1
        # return torch.ones([xb.shape[0], 10])

        

def fit(net, optimizer,  X, Y, n_epochs):
    """ Fit a net with BCEWithLogitsLoss.  Use the full batch size.
    @param net: the neural network
    @param optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
    @param X: an (N, D) torch tensor
    @param Y: an (N, 1) torch tensor
    @param n_epochs: int, the number of epochs of training
    @return epoch_loss: Array of losses at the beginning and after each epoch. Ensure len(epoch_loss) == n_epochs+1
    """
    loss_fn = nn.BCEWithLogitsLoss() #note: input to loss function needs to be of shape (N, 1) and (N, 1)
    with torch.no_grad():
        epoch_loss = [loss_fn(net(X), Y)]

    for _ in range(n_epochs):
        loss = loss_fn(net(X), Y)
        epoch_loss.append(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # TODO: compute the loss for X, Y, it's gradient, and optimize
        # TODO: append the current loss to epoch_loss
    return epoch_loss


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


def reconstruct_SVD(img, k, best=True):
    """ Compute the thin SVD for each channel of an image, keep only k singular values, and reconstruct a lossy image

    You should use numpy.linalg.svd, np.diag, and matrix multiplication
    @param img: a (M, N, 3) numpy ndarray 
    @param k: the number of singular value to keep
    @param best: Keep the k largest singular values if True.  Otherwise keep the k smallest singular values
    @return new_img: the (M, N, 3) reconstructed image
    """

    u, s, vh = np.linalg.svd(img.transpose(2, 0, 1), full_matrices=False)
    r = s.shape[1]
    # plt.plot(range(r), np.log(s[0] + 1))
    # plt.show()
    k = min(k, s.shape[1])
    if best:
        s = np.array(list(map(np.diag, s[:,:k])))
        u = u[:,:,:k]
        vh = vh[:,:k,:]
    else:
        s = np.array(list(map(np.diag, s[:,r-k:])))
        u = u[:,:,r-k:]
        vh = vh[:,r-k:,:]
    img = (u @ s @ vh).transpose(1, 2, 0)
    # img[img>1] = 1
    # img[img<0] = 0
    return img
#################################################################
# XORNet
# net = XORNet()
# X, y = hw2_utils.XOR_data()
# print(y)
# optimizer = torch.optim.SGD(net.parameters(), lr=1)
# epoch_loss = fit(net, optimizer, X, y, 5000)
# plt.plot(range(len(epoch_loss)), epoch_loss)
# plt.show()
# print(net.forward(torch.Tensor([[-1., -1.],
         # [ 1., -1.],
         # [-1.,  1.],
         # [ 1.,  1.]])))
# hw2_utils.contour_torch(-1.5, 1.5, -1.5, 1.5, net)
#################################################################

#################################################################
# Part c
#
# net = DigitsConvNet()
# loss_func = torch.nn.CrossEntropyLoss()
# sgd = torch.optim.SGD(net.parameters(), lr=0.005)
# train, val = hw2_utils.torch_digits()
# start = time.process_time()
# train_epoch_loss, validation_epoch_loss = fit_and_validate(net, sgd, loss_func, train, val, 30)
# elapsed = time.process_time() - start
# print("Training time:", elapsed)
# plt.plot(range(len(train_epoch_loss)), train_epoch_loss, label='train_loss')
# plt.plot(range(len(validation_epoch_loss)), validation_epoch_loss, label='val_loss')
# plt.legend()
# plt.show()
#
#
#################################################################


#################################################################
# Part d
#
# net = DigitsConvNet()
# loss_func = torch.nn.CrossEntropyLoss()
# sgd = torch.optim.SGD(net.parameters(), lr=0.005)
# train, val = hw2_utils.torch_digits()
# start = time.process_time()
# train_epoch_loss, validation_epoch_loss = fit_and_validate(net, sgd, loss_func, train, val, 30)
# elapsed = time.process_time() - start
# print("Training time:", elapsed)
# plt.plot(range(len(train_epoch_loss)), train_epoch_loss, label='train_loss')
# plt.plot(range(len(validation_epoch_loss)), validation_epoch_loss, label='val_loss')
# plt.legend()
# plt.show()
# torch.save(net.cpu().state_dict(), "conv.pb")
#################################################################


#################################################################
# Part e
#
# net = DigitsConvNet()
# loss_func = torch.nn.CrossEntropyLoss()
# sgd = torch.optim.SGD(net.parameters(), lr=0.005)
# train, val = hw2_utils.torch_digits()
# train_epoch_loss, validation_epoch_loss = fit_and_validate(net, sgd, loss_func, train, val, 30, batch_size=16)
# plt.plot(range(len(train_epoch_loss)), train_epoch_loss, label='train_loss')
# plt.plot(range(len(validation_epoch_loss)), validation_epoch_loss, label='val_loss')
# plt.show()
#################################################################

#################################################################
# Part f
#
# train, _ = hw2_utils.torch_digits()
# train_dl = torch.utils.data.DataLoader(train, 1617)
# net = DigitsConvNet()
# net.load_state_dict(torch.load("conv.pb"))
# for X, y in train_dl:
    # features = np.array(net.intermediate(X).detach())
    # y = np.array(y)
    # hw2_utils.plot_PCA(features, y)
#################################################################


#################################################################
# Part g

# train, val = hw2_utils.torch_digits()
# train = torch.utils.data.DataLoader(train, 1617)
# val = torch.utils.data.DataLoader(val, 180)
# net = DigitsConvNet()
# net.load_state_dict(torch.load("conv.pb"))

# train_features, train_y = None, None
# val_features, val_y = None, None
# for X, y in train:
    # train_features = np.array(net.intermediate(X).detach())
    # train_y = np.array(y)
# for X, y in val:
    # val_features = np.array(net.intermediate(X).detach())
    # val_y = np.array(y)
    
# kd_tree = scipy.spatial.KDTree(train_features)
# _, nn = kd_tree.query(val_features, 5)
# nn_labels = train_y[nn]
# nn_labels.sort()
# predict_labels = np.median(nn_labels, 1)
# acc = np.sum(predict_labels==val_y)/len(val_y)
# print(acc)
#################################################################

