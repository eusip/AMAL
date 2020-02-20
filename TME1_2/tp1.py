# standard libraries
import random

# local libraries
from classes import Context, Linear, MSELoss

# third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


def validate_functions():
	"""validate the gradients of the Linear and MSELoss fuctions"""
	#hyper-parameters
	input_size = 10
	output_size = 5
	x = torch.randn((input_size, output_size),
				 requires_grad=True, dtype=torch.float64)
	w = torch.randn(output_size, 1, requires_grad=True, dtype=torch.float64)
	b = torch.randn(input_size, 1, requires_grad=True, dtype=torch.float64)
	y_pred = torch.randn(input_size, 1, requires_grad=True, dtype=torch.float64)
	y = torch.randn(input_size, 1, requires_grad=True, dtype=torch.float64)

	# validate the gradient of the linear function -- Jacobian mismatch is expected
	linear_chk = Linear.apply
	torch.autograd.gradcheck(linear_chk, (x, w, b,),
						  eps=1e-2, atol=1e-2, raise_exception=True)

	# validate the gradient of the loss function -- Jacobian mismatch is expected
	MSELoss_chk = MSELoss.apply
	torch.autograd.gradcheck(MSELoss_chk, (y_pred, y),
 							eps=1e-2, atol=1e-2, raise_exception=True)

##################### fully manual training step #######################

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
			# forward pass (predict y)
			y_pred = linear.forward(ctx, x, w, b)
			# compute loss
			loss = criterion.forward(ctx, y_pred, y)

		return loss

	return train_test_step

########################################################################

def load_data():
	"""Boston housing dataset"""
	data = np.loadtxt('../data/housing/housing.data')
	train = data[:int(len(data) * 0.8)]  # the first 80% of the dataset
	test = data[int(len(data) * 0.8):]  # the last 20% of the dataset
	x_train = torch.from_numpy(train[:, :-1])  # all features except MEDV
	y_train = torch.from_numpy(train[:, -1]).reshape(-1, 1)  # MEDV
	x_test = torch.from_numpy(test[:, :-1])   # all features except MEDV
	y_test = torch.from_numpy(test[:, -1]).reshape(-1, 1)  # MEDV

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
	plt.figure()
	plt.plot(range(len(train)), train, 'r', label='Training')
	plt.plot(range(len(test)), test, 'b', label='Testing')
	plt.title('MSE Loss (batch type: ' + mode + ')')
	plt.legend()
	plt.show()


if __name__ == '__main__':
	torch.manual_seed(34)

	# validate_functions()

	# tensorboard writer
	writer = SummaryWriter()

	# hyper-parameters
	input_size = 13
	output_size = 1
	n_epochs = 100  # number of epochs
	lr = 1e-6  # learning rate for manually updating gradients

	# load data
	data = np.loadtxt('housing/housing.data')
	train = data[:int(len(data) * 0.8)]  # the first 80% of the dataset
	test = data[int(len(data) * 0.8):]  # the last 20% of the dataset
	x_train = torch.from_numpy(train[:, :-1])  # all features except MEDV
	y_train = torch.from_numpy(train[:, -1]).reshape(-1, 1)  # MEDV
	x_test = torch.from_numpy(test[:, :-1])   # all features except MEDV
	y_test = torch.from_numpy(test[:, -1]).reshape(-1, 1)  # MEDV

	ctx = Context()
	linear = Linear()  # custom linear prediction function
	criterion = MSELoss()  # custom MSE loss function
	train_test_step = build_train_test_step(
		input_size, output_size, ctx, linear, criterion, lr)

	# weight and bias
	w = torch.randn(input_size, output_size, dtype=torch.float64)
	b = torch.randn(output_size, dtype=torch.float64)

	for mode in ('batch','mini-batch'):
		# lists of training/testing losses for each epoch
		train_losses, test_losses = [], []
		for epoch in range(n_epochs):
			# model training
			x_batch, y_batch = batch_size(x_train, y_train, mode)
			loss = train_test_step(w, b, x_batch, y_batch, mode='train')
			train_losses.append(loss)
			# model testing
			x_batch, y_batch = batch_size(x_test, y_test, mode)
			test_loss = train_test_step(w, b, x_batch, y_batch, mode='test')
			test_losses.append(test_loss)
			writer.add_scalar('Loss/tp1/train', loss, epoch)
			writer.add_scalar('Loss/tp1/test', test_loss, epoch)
		writer.close()
		plot_losses(train_losses, test_losses, mode)