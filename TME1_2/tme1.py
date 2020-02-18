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
OUTPUT_SIZE = 1
N_EPOCHS = 100
LR = 1E-6

# tensorboard writer
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


def validate_functions():
    """validate the gradients of the Linear and MSELoss fuctions"""
    # hyper-parameters
    input_size = 10
    output_size = 5
    x = torch.randn((input_size, output_size),
                    requires_grad=True, dtype=torch.float64)
    w = torch.randn(output_size, 1, requires_grad=True, dtype=torch.float64)
    b = torch.randn(input_size, 1, requires_grad=True, dtype=torch.float64)
    y_pred = torch.randn(
        input_size, 1, requires_grad=True, dtype=torch.float64)
    y = torch.randn(input_size, 1, requires_grad=True, dtype=torch.float64)

    # validate the gradient of the linear function -- Jacobian mismatch is expected
    linear_chk = Linear.apply
    torch.autograd.gradcheck(linear_chk, (x, w, b,),
                             eps=1e-2, atol=1e-2, raise_exception=True)

    # validate the gradient of the loss function -- Jacobian mismatch is expected
    MSELoss_chk = MSELoss.apply
    torch.autograd.gradcheck(MSELoss_chk, (y_pred, y),
                             eps=1e-2, atol=1e-2, raise_exception=True)


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
    """plot training and testing MSE losses for each epoch"""
    filename = 'results/'+ mode + '.png'
    plt.figure()
    plt.plot(range(len(train)), train, 'r', label='Training')
    plt.plot(range(len(test)), test, 'b', label='Testing')
    plt.title('MSE Loss (batch type: ' + mode + ')')
    plt.legend()
    plt.savefig(filename)
    # plt.show()


def build_train_test_step(input_size, output_size, ctx, linear, criterion, lr):
    """build a function that performs a step in the training-testing loop"""
    def train_test_step(w, b, x, y, mode):
        if mode == 'train':
            # training step
            # forward pass (predict y)
            y_pred = linear.forward(ctx, x, w, b)
            # compute loss
            loss = criterion.forward(ctx, y_pred, y)
            # backward pass (compute gradients)
            delta = criterion.backward(ctx, 1)[0]
            grad_w = torch.mean(x * delta, dim=0).reshape(w.size())
            grad_b = torch.mean(delta, dim=0).reshape(b.size())
            # manually update parameters using gradient descent
            w -= lr * grad_w
            b -= lr * grad_b
            # manually zero gradients
            grad_w = torch.zeros(input_size, output_size)
            grad_b = torch.zeros(output_size)

        if mode == 'test':
            # testing step
            y_pred = linear.forward(ctx, x, w, b)
            loss = criterion.forward(ctx, y_pred, y)

        return loss

    return train_test_step


def main():
    # validate_functions()  # Jacobian mismatch is expected

    # load data
    data = np.loadtxt('housing/housing.data')
    train = data[:int(len(data) * 0.8)]  # the first 80% of the dataset
    test = data[int(len(data) * 0.8):]  # the last 20% of the dataset
    x_train = torch.from_numpy(train[:, :-1])  # all features except MEDV
    y_train = torch.from_numpy(train[:, -1]).reshape(-1, 1)  # MEDV
    x_test = torch.from_numpy(test[:, :-1])   # all features except MEDV
    y_test = torch.from_numpy(test[:, -1]).reshape(-1, 1)  # MEDV

    ctx = Context()
    linear = Linear()
    criterion = MSELoss()
    train_test_step = build_train_test_step(
        INPUT_SIZE, OUTPUT_SIZE, ctx, linear, criterion, lr=LR)

    # weight and bias
    w = torch.randn(INPUT_SIZE, OUTPUT_SIZE, dtype=torch.float64)
    b = torch.randn(OUTPUT_SIZE, dtype=torch.float64)

    for mode in ('batch', 'mini-batch'):
        # train_losses, test_losses = [], []
        for epoch in range(N_EPOCHS):
            # model training
            x_batch, y_batch = batch_size(x_train, y_train, mode)
            loss = train_test_step(w, b, x_batch, y_batch, mode='train')
            # train_losses.append(loss)
            # model testing
            x_batch, y_batch = batch_size(x_test, y_test, mode)
            test_loss = train_test_step(w, b, x_batch, y_batch, mode='test')
            # test_losses.append(test_loss)
            writer.add_scalar('tme1/train', loss, epoch)
            writer.add_scalar('tme1/test', test_loss, epoch)
        writer.close()
        # plot_losses(train_losses, test_losses, mode)


if __name__ == '__main__':
    main()
