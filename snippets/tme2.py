import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
from torchvision import datasets, transforms

class Linear(object):
    @staticmethod
    def forward(x,w,b):
        return torch.mm(x, w) + b

    @staticmethod
    def backward(delta, x, w, b):
        grad_w = torch.mean(x * delta, dim=0).reshape(w.size())
        grad_b = torch.mean(delta, dim=0).reshape(b.size())
        #grad_x = w * delta

        return grad_w, grad_b#, grad_x


class MSE(object):
    @staticmethod
    def forward(y, ypred):
        return torch.mean((y - ypred) ** 2)

    @staticmethod
    def backward(delta, y, ypred):
        return -2 * (y - ypred) * delta

class Hinge(object):
    @staticmethod
    def forward(y, ypred):
        return torch.mean(F.relu(-y * ypred))

    @staticmethod
    def backward(delta, y, ypred):
        return torch.FloatTensor(np.where(y * ypred > 0, 0, -y)) * delta

#############
# size batch
#############
def size_batch(x, y, mode):
    if mode == 'batch':
        return x, y
    elif mode == 'mini-batch':
        ind = sorted(random.sample(range(len(x)), 100))
        xbis = torch.index_select(x, 0, torch.tensor(ind))
        ybis = torch.index_select(y, 0, torch.tensor(ind))
        return xbis, ybis
    elif mode == 'stoch':
        i = random.randint(0, len(x) - 1)
        return x[i].unsqueeze(0), y[i].unsqueeze(0)
    else:
        print('erreur')
        return None

#####################
# descente de gradient
######################
def train_test():
    ###########
    # load data
    ###########
    data = pd.read_csv("housing.data", sep='\s+')
    data = np.array(data)
    train = data[:int(len(data) * 0.8)]
    test = data[int(len(data) * 0.8):]
    xtrain = torch.FloatTensor(train[:, :-1])
    ytrain = torch.FloatTensor(train[:, -1]).reshape(-1, 1)
    xtest = torch.FloatTensor(test[:, :-1])
    ytest = torch.FloatTensor(test[:, -1]).reshape(-1, 1)

    ###########
    # initialisation
    ###########
    w = torch.randn((13, 1), dtype=torch.float)
    b = torch.randn(1, dtype=torch.float)
    epsilon = 0.000001
    epoch = 1000
    ltrain = []
    ltest = []

    for i in range(epoch):
        x,y = size_batch(xtrain, ytrain, 'stoch')
        #forward
        ypred = Linear.forward(x,w,b)
        erreur = Hinge.forward(y, ypred)
        # print(erreur)
        if i > 2:
            ltrain.append(erreur)

        #backward
        delta = Hinge.backward(1, y, ypred)
        grad_w, grad_b = Linear.backward(delta, x, w, b)

        #maj param
        w -= epsilon * grad_w
        b -= epsilon * grad_b

        #test
        ypred = Linear.forward(xtest,w,b)
        erreurtest = Hinge.forward(ytest, ypred)
        print(erreur)
        if i > 2:
            ltest.append(erreurtest)

    return ltrain, ltest

# ltrain,ltest = train_test()

###########
# affichage
###########
def print_loss(ltrain, ltest):
    plt.figure()
    plt.plot(range(len(ltrain)), ltrain, 'b', label='train')
    plt.plot(range(len(ltest)), ltest, 'r', label='test')
    plt.title('Erreur= ' + 'MSE' + ', Mode du gradient= ' + 'batch')
    plt.legend()
    plt.show()


def train_test_MNIST():
    ###########
    # load data
    ###########
    batch_size = 64
    nb_digits = 10
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True,
                                                              transform=transforms.Compose([transforms.ToTensor(),
                                                                                            transforms.Normalize(
                                                                                                (0.1307,),
                                                                                                (0.3081,))])),
                                               batch_size=batch_size, shuffle=True)

    ###########
    # initialisation
    ###########
    w = torch.randn((28*28, 10), dtype=torch.float)
    b = torch.randn(10, dtype=torch.float)
    epsilon = 0.000001
    epoch = 1000
    ltrain = []
    ltest = []

    for i, (data, target) in enumerate(train_loader):
        y_onehot = torch.FloatTensor(batch_size, nb_digits)
        y_onehot.zero_()
        y_onehot[np.arange(batch_size), target] = 1
        data = data.reshape(batch_size, -1)

        #forward
        ypred = Linear.forward(data,w,b)
        erreur = Hinge.forward(y_onehot, ypred)
        print(erreur)
        if i > 2:
            ltrain.append(erreur)

        #backward
        delta = Hinge.backward(1, y_onehot, ypred)
        grad_w, grad_b = Linear.backward(delta, data, w, b)

        #maj param
        w -= epsilon * grad_w
        b -= epsilon * grad_b
        """
        #test
        ypred = Linear.forward(xtest,w,b)
        erreurtest = Hinge.forward(ytest, ypred)
        print(erreur)
        if i > 2:
            ltest.append(erreurtest)
        """

    return ltrain

train_test_MNIST()




























