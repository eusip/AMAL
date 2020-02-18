import pandas as pd
import torch.nn as nn


class RNNClassification(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        """

        """
        super(RNNClassification, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Linear(input_size, hidden_size)
        self.hidden_net = nn.Linear(hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def one_step(self, x, h):

        wx = self.encode(x)
        wh = self.hidden_net(h)

        ht = self.tanh(wx + wh)

        return ht

    def forward(self, X, h):
        for x in X:
            ht = self.one_step(x, h)

        return ht

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


# Charge data
# data = pd.read_csv("data/tempAMAL_train.csv")
# print(data.head())

# lr = 0.01
# model = RNNClassification(1, X.shape[1], 50)
# optim = torch.optim.adam(model.parameters(), lr=lr)
# loss = torch.nn.CrossEntropyLoss()

# X = torch.tensor(X)

# for _ in range(1000):
#     h = model.init_states(X.shape[1])
#     optim.zero_grad()
#     y_pred = model.forward(X, h)
#     l = loss(y_pred, y)
#     l.backward()
#     optim.step()
