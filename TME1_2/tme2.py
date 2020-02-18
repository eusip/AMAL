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


torch.manual_seed(34)

# hyper-parameters
INPUT_SIZE = 13
HIDDEN_SIZE = 13
OUTPUT_SIZE = 1
N_EPOCHS = 100
LR = 1E-6

# load data
data = np.loadtxt('housing/housing.data')
train = data[:int(len(data) * 0.8)]  # the first 80% of the dataset
test = data[int(len(data) * 0.8):]  # the last 20% of the dataset

# tensorboard writer
global writer
writer = SummaryWriter()


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
    """two-layer linear regression model using a sequential container"""

    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerLRSeq, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_size),
                                   nn.Tanh(), nn.Linear(hidden_size, output_size))

    def forward(self, x):
        y_pred = self.model(x)
        return y_pred


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
    filename = 'results/' + mode + '.png'
    plt.figure()
    plt.plot(range(len(train)), train, 'r', label='Training')
    plt.plot(range(len(test)), test, 'b', label='Testing')
    plt.title('MSE Loss (batch type: ' + mode + ')')
    plt.legend()
    plt.savefig(filename)
    # plt.show()


def build_auto_diff_step(input_size, output_size, ctx, linear, criterion, lr):
	"""build a function that performs a step in the training-testing loop"""
	def train_test_step(w, b, x, y, mode):
		if mode == 'train':
			# training step
			# forward pass (predict y)
			y_pred = linear.forward(ctx, x, w, b)
			# compute loss
			loss = criterion.forward(ctx, y_pred, y)
			loss.retain_grad
			# backward pass (compute gradients)
			loss.backward()
			print('Gradient of w: {}'.format(w.grad))
			print('Gradient of b: {}'.format(b.grad))
			# manually update parameters using gradient descent
			with torch.no_grad():
				w -= lr * w.grad
				b -= lr * b.grad
			# manually zero gradients
			w.grad.zero_()
			b.grad.zero_()

		if mode == 'test':
			# testing step
			y_pred = linear.forward(ctx, x, w, b)
			loss = criterion.forward(ctx, y_pred, y)

		return loss

	return train_test_step


def build_regression_step(model, criterion, optimizer):
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


def auto_diff(train, test, ctx, linear, criterion, lr):
    """implementation of automatic differentiation using Housing data"""
    # data
    x_train = torch.from_numpy(train[:, :-1]) # all features except MEDV
    y_train = torch.from_numpy(train[:, -1]).reshape(-1, 1)  # MEDV
    x_test = torch.from_numpy(test[:, :-1])   # all features except MEDV
    y_test = torch.from_numpy(test[:, -1]).reshape(-1, 1)  # MEDV

    # weight and bias
    w = torch.randn(INPUT_SIZE, OUTPUT_SIZE,
                    requires_grad=True, dtype=torch.float64)
    b = torch.randn(OUTPUT_SIZE, requires_grad=True, dtype=torch.float64)

    # build training step
    train_test_step = build_auto_diff_step(
    	INPUT_SIZE, OUTPUT_SIZE, ctx=ctx, linear=linear, criterion=criterion, lr=lr)

    for mode in ('batch','mini-batch'):
    	# train_losses, test_losses = [], []
    	for epoch in range(N_EPOCHS):
    		# model training
    		x_batch, y_batch = batch_size(x_train, y_train, mode)
    		loss = train_test_step(w, b, x_batch, y_batch, mode='train')
    		# train_losses.append(loss)
    		# model testing
    		x_batch, y_batch = batch_size(x_test, y_test, mode)
    		test_loss = train_test_step(w, b, x_batch, y_batch,
    			mode='test')
    		# test_losses.append(test_loss)
    		writer.add_scalar('auto_diff/train', loss, epoch)
    		writer.add_scalar('auto_diff/test', test_loss, epoch)
    	writer.close()
    	# plot_losses(train_losses, test_losses, mode)

def regression(train, test, sequential=False):
    # data
    x_train = torch.FloatTensor(train[:, :-1])  # all features except MEDV
    y_train = torch.FloatTensor(train[:, -1]).reshape(-1, 1)  # MEDV
    x_test = torch.FloatTensor(test[:, :-1])   # all features except MEDV
    y_test = torch.FloatTensor(test[:, -1]).reshape(-1, 1)  # MEDV

    # weight and bias
    w = nn.Parameter(torch.randn(HIDDEN_SIZE, OUTPUT_SIZE,
                                 requires_grad=True, dtype=torch.float64))
    b = nn.Parameter(torch.randn(OUTPUT_SIZE, requires_grad=True,
                                 dtype=torch.float64))

    if sequential == True:
        model = TwoLayerLRSeq(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        train_path = 'reg/seq/train'
        test_path = 'reg/seq/test'

    else:
        model = TwoLayerLR(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        train_path = 'reg/non_seq/train'
        test_path = 'reg/non_seq/test'

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(params=[w, b], lr=LR)
    train_test_step = build_regression_step(model, criterion, optimizer)

    for mode in ('batch', 'mini-batch'):
        # lists of training/testing losses for each epoch
        # train_losses, test_losses = [], []
        for epoch in range(N_EPOCHS):
            # model training
            x_batch, y_batch = batch_size(x_train, y_train, mode)
            loss = train_test_step(x_batch, y_batch, mode='train')
            # train_losses.append(loss)
            # model testing
            x_batch, y_batch = batch_size(x_test, y_test, mode)
            test_loss = train_test_step(x_batch, y_batch, mode='test')
            # test_losses.append(test_loss)
            writer.add_scalar(train_path, loss, epoch)
            writer.add_scalar(test_path, test_loss, epoch)
        writer.close()
        # plot_losses(train_losses, test_losses, mode)

if __name__ == '__main__':
    ctx = Context()
    linear = Linear()
    criterion = MSELoss()
    auto_diff(train, test, ctx, linear, criterion, lr=LR)
    regression(train, test)
    regression(train, test, sequential=True)
