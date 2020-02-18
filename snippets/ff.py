import torch
import torch.nn as nn
import torch.optim as opt
from torchsummary import summary

from torch.autograd import Variable
from torch.nn import functional as F

import numpy as np
from sklearn.datasets import make_blobs

# gradient example
x = torch.ones(1, requires_grad=True)
y = x + 2
z = y * y * 2

z.backward()     # automatically calculates the gradient
# print(x.grad)    # ∂z/∂x = 12

# generate data points
def blob_label(y, label, loc): # assign labels
    target = np.copy(y)
    for l in loc:
        target[y == l] = label
    return target


class Perceptron(torch.nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.fc(x))
        return output


# single-layer perception model
class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # fc1 == w1
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)  #fc2 = w2
        self.sigmoid = nn.Sigmoid() 
    
    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        fc2 = self.fc2(relu)
        output = self.sigmoid(fc2)
        return output


if __name__ == '__main__':
    
    epoch = 20  # set # of epoches

    x_train, y_train = make_blobs(
        n_samples=40, n_features=2, cluster_std=1.5, shuffle=True)
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(blob_label(y_train, 0, [0]))
    y_train = torch.FloatTensor(blob_label(y_train, 1, [1, 2, 3]))
    # print(str(y_train))
    x_test, y_test = make_blobs(
        n_samples=10, n_features=2, cluster_std=1.5, shuffle=True)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(blob_label(y_test, 0, [0]))
    y_test = torch.FloatTensor(blob_label(y_test, 1, [1, 2, 3]))
    # print(str(y_test))

    model = Feedforward(2, 10)
    print(model)
    # summary(model, (40, 2, 10)) # misconfigured

    criterion = nn.BCELoss()
    optimizer = opt.SGD(model.parameters(), lr=0.1)

    model.eval()
    y_pred = model(x_test)
    before_train = criterion(y_pred.squeeze(), y_test)
    print('Test loss before training', before_train.item())

    model.train()
    for epoch in range(epoch):
        optimizer.zero_grad()    
        # Forward pass
        y_pred = model(x_train)    
        # Compute Loss
        loss = criterion(y_pred.squeeze(), y_train)
        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))    
        # Backward pass
        loss.backward()
        optimizer.step()

    model.eval()
    y_pred = model(x_test)
    after_train = criterion(y_pred.squeeze(), y_test)
    print('Test loss after training', after_train.item())
