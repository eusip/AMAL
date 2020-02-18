# coding: utf-8

import torch
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F


# In[4]:


class Module(object):
    def initialize_parameters(self):
        pass
    
    def forward(self, x):
        pass

    def backward_update_gradient(self, x, delta):
        pass

    def update_parameter(self, epsilon):
        pass

    def backward_delta(self, x, delta):
        pass

    def zero_grad(self):
        pass

    def initialize_parameters(self):
        pass


class Loss(object):
    def forward(self, y, y_pred):
        pass

    def backward(self, y, y_pred):
        pass


# In[85]:


##########################
# Module
##########################

class Lineaire(Module):
    '''a class which characterizes the functions necessary for a linear 
    regression model'''
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.initialize_parameters()

    def initialize_parameters(self):
        '''randomly initialize a weight tensor and a bias tensor'''
        self.w = torch.randn((self.input_size, self.output_size), dtype=torch.float)
        self.b = torch.randn(self.output_size, dtype=torch.float)

    def forward(self, x):
        '''linear regression equation'''
        return torch.mm(x, self.w) + self.b

    def backward_update_gradient(self, x, delta):  # delta is (y_pred-y)
        '''computation of the weight tensor gradient and the bias tensor
        gradient'''
        self.grad_w = torch.mean(x * delta, dim=0).reshape(self.w.size())
        self.grad_b = torch.mean(delta, dim=0).reshape(self.b.size())

    def update_parameters(self, epsilon):
        '''update of parameter values'''
        # print(self.w)
        # print(self.grad_w)
        self.w -= epsilon * self.grad_w
        self.b -= epsilon * self.grad_b

    def backward_delta(self, x, delta):
        pass

    def zero_grad(self):
        self.grad_w = torch.zeros(self.input_size, self.output_size)
        self.grad_b = torch.zeros(self.output_size)


# In[86]:


########################
# Loss fonction
########################
class MSE(Loss):
    def forward(self, y, y_pred):
        return torch.mean((y - y_pred) ** 2)

    def backward(self, y, y_pred):
        return -2 * (y - y_pred)


# ### Implémentation du mode stochastique, mini-batch et batch

# In[87]:


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


# In[88]:


#############
# Train/Test
#############
def train_test(file, epoch, epsilon, mode, model, loss):
    # load data
    data = pd.read_csv(file, sep='\s+')
    data = np.array(data)
    train = data[:int(len(data) * 0.8)]
    test = data[int(len(data) * 0.8):]
    xtrain = torch.FloatTensor(train[:, :-1])
    ytrain = torch.FloatTensor(train[:, -1]).reshape(-1, 1)
    xtest = torch.FloatTensor(test[:, :-1])
    ytest = torch.FloatTensor(test[:, -1]).reshape(-1, 1)
    ltrain = []
    ltest = []

    for i in range(epoch):
        # Train
        model.zero_grad()
        x, y = size_batch(xtrain, ytrain, mode)
        yhat = model.forward(x)
        err = loss.forward(y, yhat)
        delta = loss.backward(y, yhat)
        model.backward_update_gradient(x, delta)
        model.update_parameters(epsilon)

        # Test
        yhattest = model.forward(xtest)
        errtest = loss.forward(ytest, yhattest)
        # print('err', errtest.numpy(),'itér', i)
        ltrain.append(err)
        ltest.append(errtest)

    return ltrain, ltest


# In[89]:


################
# Print loss
################
def print_loss(ltrain, ltest, error, style):
    plt.figure()
    plt.plot(range(len(ltrain)), ltrain, 'b', label='train')
    plt.plot(range(len(ltest)), ltest, 'r', label='test')
    plt.title('Erreur= ' + error + ', Mode du gradient= ' + style)
    plt.legend()
    plt.show()


# In[92]:


#################
# hyper-parametre
#################
"""
torch.manual_seed(3)
nin = 13
nout = 1
file = "housing.data"
for mode, epoch, epsilon in zip(['batch', 'mini-batch', 'stoch'], [100, 100, 100], [0.000001, 0.0000001, 0.0000001]):
    model = Lineaire(nin, nout)
    loss = MSE()
    ltrain, ltest = [], []
    ltrain, ltest = train_test(file, epoch, epsilon, mode, model, loss)
    print_loss(ltrain, ltest, 'MSE', mode)
    model = None
"""

# On remarque que la loss diminue bien pour les 3 modèles de descentes de gradients.
# Plus la taille du batch est élévé (= mode batch), plus le modèle converge lentement.
# Plus la taille du batch est faible (= mode stochastique), plus l'entrainement est instable.
# Le mode 'mini-batch' permet d'avoir un compromis entre les deux qui allie vitesse d'apprentisage et stabilité.

# ## Perceptron

# Pour implémenter le perceptron j'ai simplement ajouté la Hinge_Loss.

# In[237]:


class Hinge(Loss):

    def forward(self, y, y_pred):
        return torch.mean(F.relu(-y * y_pred))

    def backward(self, y, y_pred):
        return torch.FloatTensor(np.where(y * y_pred > 0, 0, -y))


# # FIN TME 1

# In[238]:


#################
# load data MNIST
#################

## une fois le dataset telecharge, mettre download=False !
## Pour le test, train = False
## transform permet de faire un preprocessing des donnees (ici ?)
batch_size = 64
nb_digits = 10
train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True,
                                                          transform=transforms.Compose([transforms.ToTensor(),
                                                                                        transforms.Normalize((0.1307,),(0.3081,))])),
                                                          batch_size=batch_size, shuffle=True)
print(train_loader.dataset.train_data.size())

model = Lineaire(28 * 28, 10)
loss = Hinge()

for i, (data, target) in enumerate(train_loader):
    y_onehot = torch.FloatTensor(batch_size, nb_digits)
    y_onehot.zero_()
    y_onehot[np.arange(batch_size), target] = 1
    data = data.reshape(batch_size, -1)
    model.zero_grad()
    yhat = model.forward(data)
    print(yhat.size())
    print(target)
    err = loss.forward(y_onehot, yhat)
    delta = loss.backward(y_onehot, yhat)
    model.backward_update_gradient(data, delta)
    model.update_parameters(epsilon)

# do something...
## Encoding des labels en onehot

y_onehot.scatter_(1, target.view(-1, 1), 1)
