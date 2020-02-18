# standard libraries
import os

# third-party libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader

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

class Autoencoder(nn.Module):
	"""an autoencoder for the MNIST dataset"""
	def __init__(self):
		super(Autoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Linear(28 * 28, 256),
			nn.ReLU(True),
			nn.Linear(256, 64),
			nn.ReLU(True))
		self.decoder = nn.Sequential(
			nn.Linear(64, 256),
			nn.ReLU(True),
			nn.Linear(256, 28 * 28),
			nn.Sigmoid())

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x


class HighwayNetwork(nn.Module):
	"""a highway network for the MNIST dataset"""
	def __init__(self, input_size, output_size, n_layers, 
			activation=nn.ReLU(), bias=-20.0):  # bias originally -1.0
		super(HighwayNetwork, self).__init__()
		self.transform_gate_list = nn.ModuleList(
			[nn.Linear(input_size, input_size) for _ in range(n_layers)])
		self.linear_term_list = nn.ModuleList(
			[nn.Linear(input_size, input_size) for _ in range(n_layers)])
		self.sigmoid = nn.Sigmoid()
		self.activation = activation  # alternative: nn.Tanh()
		# self.final_layer = nn.Linear(input_size, output_size)
		# self.output = nn.Softmax(self.final_layer, dim=1)
		self.model = nn.Sequential(nn.Linear(input_size, output_size), nn.Softmax(dim=1))
		for transform_gate in self.transform_gate_list:
			transform_gate.bias.data.fill_(bias)

	def forward(self, x):
		out = x

		for transform_gate, linear_term in \
			zip(self.transform_gate_list, self.linear_term_list):
			gate = self.sigmoid(transform_gate(out))
			out = gate * self.activation(linear_term(out)) + (1.0 - gate) * out
			#out = self.activation(linear_term(out))

		# out = self.output(out)
		out = self.model(out)
		return out


def one_step(x, h):  # x - batch x dim; h - batch x latent
	pass  # return batch x latent (batch x dimout)

# class RNN(nn.Module):
# 	"""a recurrent neural network"""
# 	def __init__(self, length, batch, dim, latent):
# 		super(RNN, self).__init__()
# 	self.h = pass
# 	self.final_layer = nn.Linear(input_size, output_size)
# 	self.output = nn.Softmax(self.final_layer)
	
# 	def forward(self, x, h):  # x - length x batch x dim; h - batch x latent
# 		x = one_step()
# 		return x


# class CNN(nn.Module):
# 	"""a convolutional neural network"""
# 	def __init__(self):
# 		super(CNN, self).__init__()
# 		self.conv1 = nn.Conv2d(1, 10, 5)
# 		self.conv2 = nn.Conv2d(10, 20, 5)
# 		self.conv2_drop = nn.Dropout2d()
# 		self.fc1 = nn.Linear(4*4*20, 50)  # (320, 50)
# 		self.fc2 = nn.Linear(50, 10)

# 	def forward(self, x):
# 		x = F.relu(F.max_pool2d(self.conv1(x), 2))
# 		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
# 		x = x.view(-1, 320)
# 		x = F.relu(self.fc1(x))
# 		x = F.dropout(x)
# 		x = self.fc2(x)
# 		return F.log_softmax(x)

########## the functions/classes below are not currently used ##########

# generate the layers of a highway network
def generate_linear_layers(input_size, n_layers):
	return [nn.Linear(input_size, input_size) for _ in range(n_layers)]

class State():
	def __init__(self, model, optim):
		self.model = model
		self.optim = optim
		self.epoch, self.iteration = 0, 0

class LinearRegression(nn.Module):
	"""linear regression model"""
	def __init__(self, input_size, output_size):
		super(LinearRegression, self).__init__()
		self.w = nn.Parameter(torch.randn(input_size, output_size,
			requires_grad=True, dtype=torch.float64))
		self.b = nn.Parameter(torch.randn(output_size, requires_grad=True,
			dtype=torch.float64))
		self.ctx = Context()
		self.linear = Linear

	def forward(self, x):
		y_pred = self.linear.forward(self.ctx, x, self.w, self.b)
		return y_pred


class Housing(Dataset):
	"""Boston housing dataset"""
	def __init__(self, x, y):
		super(Housing, self).__init__()
		self.data = np.loadtxt('data/housing.data')
		# self.x = data[:,:12]
		# self.y = data[:,13]	
	
	def __getitem__(self, index):
		return self.data[:,index]
	
	def __len__(self):
		return np.size(self.data, axis=0)
