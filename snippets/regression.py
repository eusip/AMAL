import torch 
import torch.nn as nn
import torch.optim as opt
from torchsummary import summary

from torch.autograd import Variable
from torch.nn import functional as F

x_data = Variable(torch.Tensor([[10.0], [9.0], [3.0], [2.0]]))
y_data = Variable(torch.Tensor([[90.0], [80.0], [50.0], [30.0]]))

class LinearRegression(nn.Module):    
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  
        
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(1, 1) 
    
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
        
if __name__ == '__main__':

    # model = LinearRegression()
    model = LogisticRegression()
    print(model)
    summary(model, (1, 1, 1))

    # criterion = nn.MSELoss()
    criterion = nn.BCELoss()

    optimizer = opt.SGD(model.parameters(), lr=0.01)

    for epoch in range(20):
        model.train()
        optimizer.zero_grad()    # Forward pass
        y_pred = model(x_data)    # Compute Loss
        loss = criterion(y_pred, y_data)    # Backward pass
        loss.backward()
        optimizer.step()

    new_x = Variable(torch.Tensor([[6.0]]))
    y_pred = model(new_x)
    print("predicted Y value: ", y_pred.data[0][0])
