# standard libraries
import os
import shutil
import time

# third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import save_image


PRINT_INTERVAL = 50
CUDA = False

# hyper-parameters
BATCH_SIZE = 64
N_EPOCHS = 50
LR = 1e-3

# tensorboard writer
global writer
writer = SummaryWriter()

# batch index
global index_batch_train_loss
index_batch_train_loss = 1

class Autoencoder(nn.Module):
	"""an autoencoder for the MNIST dataset"""

	def __init__(self, input_size, hidden_size, output_size):
		super(Autoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(True),
			nn.Linear(hidden_size, output_size),
			nn.ReLU(True))
		self.decoder = nn.Sequential(
			nn.Linear(output_size, hidden_size),
			nn.ReLU(True),
			nn.Linear(hidden_size, input_size),
			nn.Sigmoid())

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x


class HighwayNetwork(nn.Module):
	"""a highway network for the MNIST dataset"""

	def __init__(self, input_size, output_size, n_layers, bias=-20.0):  # bias originally -1.0
		super(HighwayNetwork, self).__init__()
		self.transform_gate_list = nn.ModuleList(
			[nn.Linear(input_size, input_size) for _ in range(n_layers)])
		self.linear_term_list = nn.ModuleList(
			[nn.Linear(input_size, input_size) for _ in range(n_layers)])
		self.sigmoid = nn.Sigmoid()
		self.activation = nn.ReLU()  # alternative: nn.Tanh()
		# self.final_layer = nn.Linear(input_size, output_size)
		# self.output = nn.Softmax(self.final_layer, dim=1)
		self.model = nn.Sequential(
			nn.Linear(input_size, output_size), nn.Softmax(dim=1))
		for transform_gate in self.transform_gate_list:
			transform_gate.bias.data.fill_(bias)

	def forward(self, x):
		out = x

		for transform_gate, linear_term in \
				zip(self.transform_gate_list, self.linear_term_list):
			gate = self.sigmoid(transform_gate(out))
			out = gate * self.activation(linear_term(out)) + (1.0 - gate) * out
			# out = self.activation(linear_term(out))

		# out = self.output(out)
		out = self.model(out)
		return out


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self, name, fmt=':f'):
		self.name = name
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __str__(self):
		fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
		return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
	"""Keeps track of progress during the training process"""
	def __init__(self, num_batches, meters, prefix=""):
		self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
		self.meters = meters
		self.prefix = prefix

	def display(self, batch):
		entries = [self.prefix + self.batch_fmtstr.format(batch)]
		entries += [str(meter) for meter in self.meters]
		print('\t'.join(entries))

	def _get_batch_fmtstr(self, num_batches):
		num_digits = len(str(num_batches // 1))
		fmt = '{:' + str(num_digits) + 'd}'
		return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified
	values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res


def load_MNIST(batch_size):
	"""this function loads (and if necessary downloads) the MNIST dataset """

	transform = transforms.Compose([transforms.ToTensor(),
									transforms.Normalize((0.1307,), (0.3081,))])

	train_dataset = datasets.MNIST('.', download=False, train=True,
								   transform=transform)

	test_dataset = datasets.MNIST('.', download=False, train=False,
								  transform=transform)
	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
							  shuffle=True, pin_memory=CUDA, num_workers=4)
	test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
							 shuffle=True, pin_memory=CUDA, num_workers=4)

	# get first batch of training images
	dataiter = iter(train_loader)
	images, labels = dataiter.next()

	# create grid of images
	img_grid = torchvision.utils.make_grid(images, normalize=True)

	# write grid to tensorboard
	# writer.add_image('mnist_images', img_grid)

	return train_loader, test_loader


def epoch(data, checkpoint, model, criterion, optimizer=None, counter=None):
	"""this function carries out a single learning or evaluation epoch"""

	global writer, index_batch_train_loss

	if optimizer:  # train
		batch_time = AverageMeter('Time', ':6.3f')
		data_time = AverageMeter('Data', ':6.3f')
		losses = AverageMeter('Loss', ':.4e')
		top1 = AverageMeter('Acc@1', ':6.2f')
		top5 = AverageMeter('Acc@5', ':6.2f')
		progress = ProgressMeter(
			len(data),
			[batch_time, data_time, losses, top1, top5],
			prefix="Train Epoch: [{}]".format(counter))

		model.train()
		start_time = time.time()
		for i, (image, target) in enumerate(data):
			image = image.view(image.size(0), -1)
			# image = image.reshape(-1, 28 * 28)

			# data loading time
			data_time.update(time.time() - start_time)

			if CUDA:
				image = image.cuda()
				target = target.cuda()

			# forward pass
			output = model(image)
			# compute loss
			if checkpoint == 'autoencoder':
				loss = criterion(output, image)
			if checkpoint == 'highway':
				loss = criterion(output, target)

			# accuracy and record loss
			acc1, acc5 = accuracy(output, target, topk=(1, 5))
			losses.update(loss.item(), image.size(0))
			top1.update(acc1[0], image.size(0))
			top5.update(acc5[0], image.size(0))

			# zero gradients
			optimizer.zero_grad()
			# backward pass
			loss.backward()
			# update parameters
			optimizer.step()

			# elapsed time
			batch_time.update(time.time() - start_time)
			start_time = time.time()

			if i % PRINT_INTERVAL == 0:
				progress.display(i)
				writer.add_scalar("batches/loss/train", losses.avg,
									index_batch_train_loss)
				writer.add_scalar("batches/accuracy/train", top1.avg,
								  	index_batch_train_loss)
				index_batch_train_loss += 1
	else:  # test
		batch_time = AverageMeter('Time', ':6.3f')
		losses = AverageMeter('Loss', ':.4e')
		top1 = AverageMeter('Acc@1', ':6.2f')
		top5 = AverageMeter('Acc@5', ':6.2f')
		progress = ProgressMeter(
			len(data),
			[batch_time, losses, top1, top5],
			prefix="Test ")

		model.eval()
		with torch.no_grad():
			start_time = time.time()
			for i, (image, target) in enumerate(data):
				image = image.view(image.size(0), -1)
				# image = image.reshape(-1, 28 * 28)

				if CUDA:
					image = image.cuda()
					target = target.cuda()

				# forward pass
				output = model(image)
				# compute loss
				if checkpoint == 'autoencoder':
					loss = criterion(output, image)
				if checkpoint == 'highway':
					loss = criterion(output, target)

				# accuracy and record loss
				acc1, acc5 = accuracy(output, target, topk=(1, 5))
				losses.update(loss.item(), image.size(0))
				top1.update(acc1[0], image.size(0))
				top5.update(acc5[0], image.size(0))

				# elapsed time
				batch_time.update(time.time() - start_time)
				start_time = time.time()

				if i % PRINT_INTERVAL == 0:
					progress.display(i)
					print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
						  .format(top1=top1, top5=top5))
	return top1, top5, losses


def autoencoder(resume=False):
	if not os.path.exists('./checkpoints'):
		os.mkdir('./checkpoints')

	pathname = './checkpoints/autoencoder.pt'
	train_path = 'autoencoder/train'
	test_path = 'autoencoder/test'

	input_size = 28 * 28
	hidden_size = 256
	output_size = 64

	model = Autoencoder(input_size, hidden_size, output_size)
	criterion = nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

	if CUDA:
		model = model.cuda()
		criterion = criterion.cuda()

	if resume == True:
		try:
			os.path.isfile(pathname)
			print("=> loading checkpoint")
			checkpoint = torch.load(pathname)
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			start_epoch = checkpoint['epoch']
			print("=> loaded checkpoint (epoch {} of {})"
				  .format(checkpoint['epoch'], N_EPOCHS))
		except KeyError:
			print("=> checkpoint does not exist.")
	else:
		start_epoch = 0

	train, test = load_MNIST(BATCH_SIZE)

	# top1_train, top5_train, losses_train = 0, 0, 0
	# top1_test, top5_test, losses_test = 0, 0, 0

	for ep in range(start_epoch, N_EPOCHS):
		print("=================\n==== EPOCH " +
					str(ep + 1) + " ====\n=================\n")

		top1_train, top5_train, losses_train = epoch(
			train, 'autoencoder', model, criterion, optimizer, ep + 1)  # train
		top1_test, top5_test, losses_test = epoch(
			test, 'autoencoder', model, criterion)  # test

		# save checkpoint
		torch.save({
			'state_dict': model.state_dict(),
			'optimizer': optimizer.state_dict(),
			'epoch': ep + 1
		}, pathname)

		writer.add_scalar('autoencoder/epochs/loss/train', losses_train.avg, ep)
		writer.add_scalar(
			'autoencoder/epochs/accuracy/train', top1_train.avg, ep)
		writer.add_scalar('autoencoder/epochs/loss/test', losses_test.avg, ep)
		writer.add_scalar(
			'autoencoder/epochs/accuracy/test', top1_test.avg, ep)


def highway_network(resume=False):
	if not os.path.exists('./checkpoints'):
		os.mkdir('./checkpoints')

	pathname = './checkpoints/highway.pt'
	train_path = 'highway/train'
	test_path = 'highway/test'

	input_size = 28 * 28
	output_size =  10
	n_layers = 2

	model = HighwayNetwork(input_size, output_size, n_layers)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

	if CUDA:
		model = model.cuda()
		criterion = criterion.cuda()

	if resume == True:
		try:
			os.path.isfile(pathname)
			print("=> loading checkpoint")
			checkpoint = torch.load(pathname)
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			start_epoch = checkpoint['epoch']
			print("=> loaded checkpoint (epoch {} of {})"
				  .format(checkpoint['epoch'], N_EPOCHS))
		except KeyError:
			print("=> checkpoint does not exist.")
	else:
		start_epoch = 0

	train, test = load_MNIST(BATCH_SIZE)

	# top1_train, top5_train, losses_train = 0, 0, 0
	# top1_test, top5_test, losses_test = 0, 0, 0

	for ep in range(start_epoch, N_EPOCHS):
		print("=================\n==== EPOCH " +
					str(ep + 1) + " ====\n=================\n")

		top1_train, top5_train, losses_train = epoch(
			train, 'highway', model, criterion, optimizer, ep + 1)  # train
		top1_test, top5_test, losses_test = epoch(
			test, 'highway', model, criterion)  # test

		# save checkpoint
		torch.save({
			'state_dict': model.state_dict(),
			'optimizer': optimizer.state_dict(),
			'epoch': ep + 1
		}, pathname)

		# bug to resolve ?
		writer.add_scalar('highway/epochs/loss/train', losses_train.avg, ep)
		writer.add_scalar(
			'highway/epochs/accuracy/train', top1_train.avg, ep)
		writer.add_scalar('highway/epochs/loss/test', losses_test.avg, ep)
		writer.add_scalar(
			'highway/epochs/accuracy/test', top1_test.avg, ep)

if __name__ == '__main__':
	autoencoder(resume=False)
	highway_network(resume=False)
