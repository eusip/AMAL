import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
from torchvision import datasets, transforms

class Linear2(object):
    @staticmethod
    def forward(x,w,b):
        return torch.mm(x,w) + b

    @staticmethod
    def backward(delta, x, grad_w, grad_b):
        grad_w += torch.mm(torch.t(delta), x).view(grad_w.size())
        grad_b += delta.view(grad_b.size())
        grad_x = None

        return grad_w, grad_b, grad_x

class Linear(object):
    @staticmethod
    def forward(x,w,b):
        return torch.mm(x,w) + b

    @staticmethod
    def backward(delta, x, w, b):
        print(torch.mm(torch.t(delta),x).size())
        grad_w = torch.mean(torch.mm(torch.t(delta),x), dim=0).reshape(w.size())
        grad_b = torch.mean(delta, dim=0).reshape(b.size())
        # grad_x = w * delta

        return grad_w, grad_b # , grad_x

class Sigmoide(object):
    @staticmethod
    def forward(x):
        return 1/1+torch.exp(-x)

    @staticmethod
    def backward(delta, x):
        return torch.mul((Sigmoide.forward(x)*(1 - Sigmoide.forward(x))),delta)

class Tanh(object):
    @staticmethod
    def forward(x):
        return torch.tanh(x)

    @staticmethod
    def backward(delta, x):
        return torch.mul(delta.view(1,-1), (torch.FloatTensor(x.size()).fill_(1.) - torch.tanh(x)**2))

class Soft_max(object):
    @staticmethod
    def forward(x):
        xmax = x - torch.max(x)
        exp = torch.exp(xmax)
        return exp / torch.sum(exp)

    @staticmethod
    def backward(delta, x):
        ypred = Soft_max.forward(x)
        res = torch.zeros(len(ypred), len(x))
        for i in range(len(ypred)):
            for j in range(len(x)):
                if i == j:
                    res[i, j] = ypred[i] * (1-ypred[i])
                else:
                    res[i, j] = ypred[i] * ypred[j]
        res = res.sum(axis=1)

        return torch.mul(res,delta)

class MSE(object):
    @staticmethod
    def forward(y, ypred):
        return ((y - ypred) ** 2).mean()

    @staticmethod
    def backward(delta, y, ypred):
        return torch.mul((-2 * (y - ypred)), delta)

class Hinge(object):
    @staticmethod
    def forward(y, ypred):
        return torch.max(torch.zeros(y.size()), -1*torch.mul(y,ypred)).mean()

    @staticmethod
    def backward(delta, y, ypred):
        tmp = torch.mul(y,ypred)
        return torch.where(tmp < torch.zeros(tmp.size()), -y, torch.zeros(tmp.size())) * delta

def Cross_entropy(object):
    @staticmethod
    def forward(y, ypred):
        return -torch.sum(torch.mul(y, torch.log(ypred)).sum(axis=0))

    @staticmethod
    def backward(delta, y, ypred):
        pass


def train_test_MNIST():
    ###########
    # load data
    ###########
    batch_size = 1000
    nb_digits = 10
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True,
                                                              transform=transforms.Compose([transforms.ToTensor(),
                                                                                            transforms.Normalize(
                                                                                                (0.1307,),
                                                                                                (0.3081,))])),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, download=True,
                                                             transform=transforms.Compose([transforms.ToTensor(),
                                                                                           transforms.Normalize(
                                                                                               (0.1307,),
                                                                                               (0.3081,))])),
                                              batch_size=batch_size, shuffle=True, drop_last=True)
    ################
    # initialisation
    ################
    w = torch.randn((28 * 28, 1), dtype=torch.float)
    b = torch.randn(1, dtype=torch.float)
    epsilon = 0.000001
    epoch = 20
    ltrain = []
    ltest = []
    for i in range(epoch):
        for i, (data, target) in enumerate(train_loader):
            y_onehot = - torch.ones((batch_size, nb_digits))
            y_onehot[np.arange(batch_size), target] = 1
            y_onehot = y_onehot[:, 0].reshape(-1, 1)
            data = data.reshape(batch_size, -1)

            # forward
            forward1 = Linear.forward(data, w, b)
            act1 = Soft_max.forward(forward1)
            print(act1.sum())
            print(np.where(act1 == 1))
            exit()
            erreur = Hinge.forward(y_onehot, ypred)
            print('train', erreur)
            ltrain.append(erreur)

            # backward
            delta = Hinge.backward(1, y_onehot, ypred)
            grad_w, grad_b = Linear.backward(delta, data, w, b)

            # maj param
            w -= epsilon * grad_w
            b -= epsilon * grad_b

            if i % 10 == 1:
                for i, (datat, targett) in enumerate(test_loader):
                    y_onehott = - torch.ones((batch_size, nb_digits))
                    y_onehott[np.arange(batch_size), targett] = 1
                    y_onehott = y_onehott[:, 0].reshape(-1, 1)
                    datat = datat.reshape(batch_size, -1)

                    # forward
                    ypredt = Linear.forward(datat, w, b)
                    erreurtest = Hinge.forward(y_onehott, ypredt)
                    print('test', erreurtest)
                    ltest.append(erreurtest)

    return ltrain, ltest


ltrain, ltest = train_test_MNIST()
print_loss(ltrain, ltest)