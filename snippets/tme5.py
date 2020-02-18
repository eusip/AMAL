import torch
import torch. nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import torch.nn.parameter as P


"""
x = torch.randn(10, requires_grad = True)
y = torch.randn(10, requires_grad = True)

s = ((x-y)**2).sum()

s.backward()
"""

class context(object):

    def __init__(self):
        self.x = None
        self.w = None

    def input(self, x, w):
        self.x = x
        self.w = w

    def getx(self):
        return self.x

    def getw(self):
        return self.w

class Linear(torch.autograd.Function):

    @staticmethod
    def forward(self, ctx, x, w, b):
        res = torch.mm(x,w) + b
        ctx.input(x, w)
        return res

    @staticmethod
    def backward(self, ctx, grad_output):
        grad_w = torch.mm(torch.t(ctx.getx()), grad_output)
        grad_b = grad_output

        return grad_w, grad_b

class Sigmoide(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        return torch.sigmoid(x)

    @staticmethod
    def backward(self, delta, x):
        return torch.mul((torch.sigmoid(x)*(1 - torch.sigmoid(x))),delta)

class MSE(torch.autograd.Function):
    @staticmethod
    def forward(self, y, ypred):
        return ((y - ypred) ** 2).mean()

    @staticmethod
    def backward(self, delta, y, ypred):
        return torch.mul((-2 * (y - ypred)), delta)



def train():
    torch.manual_seed(666)

    ###########
    # load data
    ###########
    epoch = 5
    data = pd.read_csv("housing.data", sep='\s+')
    data = np.array(data)
    train = data[:int(len(data) * 0.8)]
    test = data[int(len(data) * 0.8):]
    xtrain = torch.FloatTensor(train[:, :-1])
    xtrain = torch.tensor(xtrain, requires_grad=True)
    ytrain = torch.FloatTensor(train[:, -1]).reshape(-1, 1)
    xtest = torch.FloatTensor(test[:, :-1])
    ytest = torch.FloatTensor(test[:, -1]).reshape(-1, 1)

    #########################
    # declaration des modules
    #########################

    linear = Linear.apply
    sigmoide = Sigmoide.apply
    mse = MSE.apply

    ctx = context()

    ############################
    # Declaration des paramètres
    ############################
    w1 = torch.nn.Parameter(torch.randn(13, 1))
    b1 = torch.nn.Parameter(torch.randn(1))
    #w2 = P(torch.randn(10, 1))
    #b2 = P(torch.randn(1))
    opt = optim.SGD([w1, b1], lr=1e-2)

    ########
    # Train
    ########

    for i in range(epoch):
        opt.zero_grad()
        forward = linear(ctx, xtrain, w1, b1)
        erreur = mse(forward, ytrain)
        print(erreur)

        erreur.backward()
        opt.step()

train()


