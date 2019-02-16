import hw2_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class XORNet(nn.Module):
    def __init__(self):
        """
        Initialize the layers of your neural network

        You should use nn.Linear
        """
        super(XORNet, self).__init__()
    
    def set_l1(self, w, b):
        """
        Set the weights and bias of your first layer
        @param w: (2,2) torch tensor
        @param b: (2,) torch tensor
        """
        pass
    
    def set_l2(self, w, b):
        """
        Set the weights and bias of your second layer
        @param w: (1,2) torch tensor
        @param b: (1,) torch tensor
        """
        pass
    
    def forward(self, xb):
        """
        Compute a forward pass in your network.  Note that the nonlinearity should be F.relu.
        @param xb: The (n, 2) torch tensor input to your model
        @return: an (n, 1) torch tensor
        """
        return torch.ones(xb.shape[0], 1)

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

    def set_parameters(self, kern1, bias1, kern2, bias2, fc_weight, fc_bias):
        """ Set the parameters of your network

        @param kern1: an (8, 1, 3, 3) torch tensor
        @param bias1: an (8,) torch tensor
        @param kern2: an (4, 8, 3, 3) torch tensor
        @param bias2: an (4,) torch tensor
        @param fc_weight: an (10, 4) torch tensor
        @param fc_bias: an (10,) torch tensor
        """
        pass

    def intermediate(self, xb):
        """ Return the feature representation your network lerans

        Note that the nonlinearity between each layer should be F.relu.  You
        may need to use a tensor's view() method to reshape outputs. Hint: this
        should be very similar to your forward method
        @param xb: an (N, 8, 8) torch tensor
        @return: an (N, 4) torch tensor
        """
        pass

    def forward(self, xb):
        """ A forward pass of your neural network

        Note that the nonlinearity between each layer should be F.relu.  You
        may need to use a tensor's view() method to reshape outputs
        @param xb: an (N, 8, 8) torch tensor
        @return: an (N, 10) torch tensor
        """
        return torch.ones([xb.shape[0], 10])
        

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
        #TODO: compute the loss for X, Y, it's gradient, and optimize
        #TODO: append the current loss to epoch_loss
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
    with torch.no_grad():
        # compute the mean loss on the training set at the beginning of iteration
        losses, nums = zip(*[loss_batch(net, loss_func, X, Y) for X, Y in train_dl])
        train_epoch_loss = [np.sum(np.multiply(losses, nums)) / np.sum(nums)]
        # TODO compute the validation loss and store it in a list
    for _ in range(n_epochs):
        net.train() #put the net in train mode
        # TODO 
        with torch.no_grad():
            net.eval() #put the net in evaluation mode
            # TODO compute the train and validation losses and store it in a list
    return train_el, val_el


def reconstruct_SVD(img, k, best=True):
    """ Compute the thin SVD for each channel of an image, keep only k singular values, and reconstruct a lossy image

    You should use numpy.linalg.svd, np.diag, and matrix multiplication
    @param img: a (M, N, 3) numpy ndarray 
    @param k: the number of singular value to keep
    @param best: Keep the k largest singular values if True.  Otherwise keep the k smallest singular values
    @return new_img: the (M, N, 3) reconstructed image
    """
    return np.zeros(img.shape)

