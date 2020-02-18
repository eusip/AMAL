# standard libraries
import random

# third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Function
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


class Context():
    def __init__(self):
        self._saved_tensors = ()

    def save_for_backward(self, *args):
        self._saved_tensors = args

    @property
    def saved_tensors(self):
        return self._saved_tensors


class Linear(Function):
    """linear regression equation"""
    @staticmethod
    def forward(ctx, x, w, b):  # input/data, weights/parameters, bias
        ctx.save_for_backward(x, w, b)
        return torch.mm(x, w) + b

    @staticmethod
    def backward(ctx, grad_output):
        x, w, b = ctx.saved_tensors
        return None, None, None


class MSELoss(Function):
    """mean-squared error loss function"""
    @staticmethod
    def forward(ctx, y_pred, y):  # input/pred, target/label
        ctx.save_for_backward(y_pred, y)
        return torch.mean((y - y_pred) ** 2)

    @staticmethod
    def backward(ctx, grad_output):
        y_pred, y = ctx.saved_tensors
        return -2 * (y - y_pred), None


class TwoLayerLR(nn.Module):
    """two-layer linear regression model"""

    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerLR, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        fc1 = self.fc1(x)
        tanh = self.tanh(fc1)
        fc2 = self.fc2(tanh)
        y_pred = fc2
        # y_pred = self.fc2(self.tanh(self.fc1(x)))
        return y_pred


class TwoLayerLRSeq(nn.Module):
    """two-layer linear regression model using a container"""

    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerLRSeq, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_size),
                                   nn.Tanh(), nn.Linear(hidden_size, output_size))

    def forward(self, x):
        y_pred = self.model(x)
        return y_pred


################ automatic differention using autograd #################

# def build_train_test_step(input_size, output_size, ctx, linear, criterion, lr):
# 	"""build a function that performs a step in the training-testing loop"""
# 	def train_test_step(w, b, x, y, mode):
# 		if mode == 'train':
# 			# training step
# 			# forward pass (predict y)
# 			y_pred = linear.forward(ctx, x, w, b)
# 			# compute loss
# 			loss = criterion.forward(ctx, y_pred, y)
# 			loss.retain_grad
# 			# backward pass (compute gradients)
# 			loss.backward()
# 			print('Gradient of w: {}'.format(w.grad))
# 			print('Gradient of b: {}'.format(b.grad))
# 			# manually update parameters using gradient descent
# 			with torch.no_grad():
# 				w -= lr * w.grad
# 				b -= lr * b.grad
# 			# manually zero gradients
# 			w.grad.zero_()
# 			b.grad.zero_()

# 		if mode == 'test':
# 			# testing step
# 			y_pred = linear.forward(ctx, x, w, b)
# 			loss = criterion.forward(ctx, y_pred, y)

# 		return loss

# 	return train_test_step

################## two-layer linear regression module ##################

def build_train_test_step(model, criterion, optimizer):
    """build a function that performs a step in the training-tesing loop"""
    def train_test_step(x, y, mode):
        if mode == 'train':
            # set model to train
            model.train()
            # forward pass (predict y)
            y_pred = model(x)
            # compute loss
            loss = criterion(y_pred, y)
            # zero gradients
            optimizer.zero_grad()
            # backward pass (compute gradients)
            loss.backward()
            # update parameters
            optimizer.step()

        if mode == 'test':
            # set model to evaluate
            model.eval()
            # supress gradient calculations
            with torch.no_grad():
                y_pred = model(x)
                loss = criterion(y_pred, y)

        return loss

    return train_test_step

########################################################################


def batch_size(x, y, mode):
    """batch data; entire dataset or mini-batches"""
    if mode == 'batch':
        return x, y
    elif mode == 'mini-batch':
        ind = sorted(random.sample(range(len(x)), 100))
        x_mini = torch.index_select(x, 0, torch.tensor(ind))
        y_mini = torch.index_select(y, 0, torch.tensor(ind))
        return x_mini, y_mini
    else:
        print('Invalid batch mode. Please selection another.')
        None


def plot_losses(train, test, mode):
    """plot training and validation MSE losses for each epoch"""
    plt.figure()
    plt.plot(range(len(train)), train, 'r', label='Training')
    plt.plot(range(len(test)), test, 'b', label='Testing')
    plt.title('MSE Loss (batch type: ' + mode + ')')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(34)

    # tensorboard writer
    writer = SummaryWriter()

    # hyper-parameters
    input_size = 13
    hidden_size = 13
    output_size = 1
    n_epochs = 100  # number of epochs
    lr = 1e-6  # learning rate for manually updating gradients

    # load data
    data = np.loadtxt('../data/housing/housing.data')
    train = data[:int(len(data) * 0.8)]  # the first 80% of the dataset
    test = data[int(len(data) * 0.8):]  # the last 20% of the dataset

    ############### automatic differention using autograd ##############

    # ctx = Context()
    # linear = Linear()  # custom linear prediction function
    # criterion = MSELoss()  # custom MSE loss function
    # train_test_step = build_train_test_step(
    # 	13, 1, ctx, linear, criterion, lr)

    # # data
    # x_train = torch.from_numpy(train[:, :-1]) # all features except MEDV
    # y_train = torch.from_numpy(train[:, -1]).reshape(-1, 1)  # MEDV
    # x_test = torch.from_numpy(test[:, :-1])   # all features except MEDV
    # y_test = torch.from_numpy(test[:, -1]).reshape(-1, 1)  # MEDV

    # # weight and bias
    # w = torch.randn(input_size, output_size,
    #                 requires_grad=True, dtype=torch.float64)
    # b = torch.randn(output_size, requires_grad=True, dtype=torch.float64)

    # for mode in ('batch','mini-batch'):
    # 	# lists of training/testing losses for each epoch
    # 	train_losses, test_losses = [], []
    # 	for epoch in range(n_epochs):
    # 		# model training
    # 		x_batch, y_batch = batch_size(x_train, y_train, mode)
    # 		loss = train_test_step(w, b, x_batch, y_batch, mode='train')
    # 		train_losses.append(loss)
    # 		# model testing
    # 		x_batch, y_batch = batch_size(x_test, y_test, mode)
    # 		test_loss = train_test_step(w, b, x_batch, y_batch,
    # 			mode='test')
    # 		test_losses.append(test_loss)
    # 		writer.add_scalar('Loss/tp2/train', loss, epoch)
    # 		writer.add_scalar('Loss/tp2/test', test_loss, epoch)
    # 	writer.close()
    # 	plot_losses(train_losses, test_losses, mode)

    ################# two-layer linear regression module ###############

    # data
    x_train = torch.FloatTensor(train[:, :-1])  # all features except MEDV
    y_train = torch.FloatTensor(train[:, -1]).reshape(-1, 1)  # MEDV
    x_test = torch.FloatTensor(test[:, :-1])   # all features except MEDV
    y_test = torch.FloatTensor(test[:, -1]).reshape(-1, 1)  # MEDV

    # weight and bias
    w = nn.Parameter(torch.randn(hidden_size, output_size,
                                 requires_grad=True, dtype=torch.float64))
    b = nn.Parameter(torch.randn(output_size, requires_grad=True,
                                 dtype=torch.float64))

    # two-layer linear regression model
    # model = TwoLayerLR(input_size, hidden_size, output_size)
    # two-layer linear regression model (seq.)
    model = TwoLayerLRSeq(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(params=[w, b], lr=lr)
    train_test_step = build_train_test_step(model, criterion, optimizer)

    for mode in ('batch', 'mini-batch'):
            # lists of training/testing losses for each epoch
        train_losses, test_losses = [], []
        for epoch in range(n_epochs):
                # model training
            x_batch, y_batch = batch_size(x_train, y_train, mode)
            loss = train_test_step(x_batch, y_batch, mode='train')
            train_losses.append(loss)
            # model testing
            x_batch, y_batch = batch_size(x_test, y_test, mode)
            test_loss = train_test_step(x_batch, y_batch, mode='test')
            test_losses.append(test_loss)
            writer.add_scalar('Loss/tp2/train', loss, epoch)
            writer.add_scalar('Loss/tp2/test', test_loss, epoch)
		writer.add_graph(net, images)
        writer.close()
        plot_losses(train_losses, test_losses, mode)
