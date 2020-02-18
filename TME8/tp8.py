# standard libraries
import argparse
import os 

# local libraries

# third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# hyper-parameters
PRINT_INTERVAL = 50
BATCH_SIZE = 2  # 300
N_EPOCHS = 1000
LR = 1e-3
WD = 1e-5  # weight decay is not L2 regularisation for Adam
LAMBDA = 1e-5
INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 100
OUTPUT_SIZE = 10

class Linear(nn.Module):  # input_size = 28 * 28 hidden_size = 100, output_size = 10
	"""three-layer linear regression model"""
	def __init__(self):
		# self.features = nn.Sequential(nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
		# 						   	  nn.BatchNorm1d(HIDDEN_SIZE),
		# 						   	  nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
		# 						   	  nn.BatchNorm1d(HIDDEN_SIZE),
		# 						   	  nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
		# 						   	  nn.BatchNorm1d(HIDDEN_SIZE)
		# 						   	 )
		# use a list of the linear layers
		self.classifier = nn.Sequential(nn.Dropout(p=0.2),
										nn.Linear(hidden_size, output_size))
		
	layers = []
	track_layers = []

	def forward(self, x):
		pass
		# for l in layers:
		#     x = l.x  # get the input of the layer
		#     if l in track_layers:
		#         pass  # store gradient of x x = store_approx_grad(x)
		#     	outs.append(x)
		# return x outs

def load_data(batch_size):
	transform = transforms.Compose([transforms.ToTensor(),
								 transforms.Normalize((0.1307,), (0.3081,))])
	train_ds = datasets.MNIST('../data', download=False, train=True,
						   transform=transform)
	test_ds = datasets.MNIST('../data', download=False, train=False,
						  transform=transform)
	train_sampler = RandomSampler(train_ds, replacement=True, 
								num_samples = int(len(train_ds) * 0.04))
	valid_sampler = RandomSampler(train_ds, replacement=True, 
								num_samples = int(len(train_ds) * 0.01))
	train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, sampler=train_sampler)
	valid_dl = DataLoader(dataset=train_ds, batch_size=batch_size, sampler=valid_sampler)
	test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True)
	
	return train_dl, valid_dl, test_dl

def store_approx_grad(var):
	def hook(grad):
		var.grad = grad
	return var.register_hook(hook)

def epoch(data, model, criterion, optimizer=None):
	# indique si le modele est en mode eval ou train (certaines couches se
    # comportent différemment en train et en eval)
    model.eval() if optimizer is None else model.train()

    # objets pour stocker les moyennes des metriques
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()
    avg_batch_time = AverageMeter()

    global pd_train_loss, writer, index_batch_train_loss

    # on itere sur les batchs du dataset
    tic = time.time()
    for i, (input, target) in enumerate(data):
        if CUDA:  # si on fait du GPU, passage en CUDA
            input = input.cuda()
            target = target.cuda()

        # forward
        output = model(input)
        loss = compute_loss(criterion, output, target)

        # backward si on est en "train"
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # calcul des metriques
        acc = accuracy(output, target)
        batch_time = time.time() - tic
        tic = time.time()

        # mise a jour des moyennes
        avg_loss.update(loss.item())
        avg_acc.update(acc.item())
        avg_batch_time.update(batch_time)

        if optimizer:
            pd_train_loss = pd_train_loss.append(
                dict({'train_loss': avg_loss.val}), ignore_index=True)

        # affichage des infos
        if i % PRINT_INTERVAL == 0:
            print(
                '[{0:s} Batch {1:03d}/{2:03d}]\t'
                'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {acc.val:5.1f} ({acc.avg:5.1f})'.format(
                    "EVAL" if optimizer is None else "TRAIN",
                    i,
                    len(data),
                    batch_time=avg_batch_time,
                    loss=avg_loss,
                    acc=avg_acc))

            if optimizer:
                writer.add_scalar(
                    "Batches/Accuracy/Train",
                    avg_acc.val,
                    index_batch_train_loss)

                writer.add_scalar(
                    "Batches/Loss/Train",
                    avg_loss.val,
                    index_batch_train_loss)
                index_batch_train_loss += 1

    # Affichage des infos sur l'epoch
    print('\n===============> Total time {batch_time:d}s\t'
          'Avg loss {loss.avg:.4f}\t'
          'Avg Accuracy {acc.avg:5.2f} %\n'.format(
              batch_time=int(avg_batch_time.sum), loss=avg_loss,
              acc=avg_acc))

    return avg_acc, avg_loss



def main():
	for root in ['../data', './images', './checkpoints']:
		if not os.path.exists(root):
			os.mkdir(root)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# tensorboard writer
	writer = SummaryWriter()

	train, valid, test = load_data(BATCH_SIZE)

	# for x in train:
	# 	print(x)
	# 	break

	model = Linear().to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

	for name, param in model.named_parameters():
		if param.requires_grad:
			print(name, param.data)

	# if os.path.isfile('./checkpoints/checkpoint_tp8.pth.tar'):
	# 		print("=> loading checkpoint")
	# 		checkpoint = torch.load('./checkpoints/checkpoint_tp8.pth.tar')
	# 		model.load_state_dict(checkpoint['state_dict'])
	# 		optimizer.load_state_dict(checkpoint['optimizer'])
	# 		start_epoch = checkpoint['epoch']
	# 		print("=> loaded checkpoint (epoch {})"
	# 					.format(checkpoint['epoch']))
	# else:
	# 	start_epoch = 0

	# print("=> beginning training")
	# for epoch in range(start_epoch, n_epochs):
	# 	# model training
	# 	for (data, target) in train_dl:
	# 		model.train()
	# 		output = model.forward(data)
	# 		l2 = torch.tensor(0.)
	# 		for param in model.parameters():
	# 			l2 += torch.norm(param)
	# 		# loss += lambda * l2
	# 		loss = criterion(output, target) + LAMBDA * l2
	# 		optimizer.zero_grad()
	# 		loss.backward()
	# 	# model validation
	# 	for (data, target) in train_dl:
	# 		pass	
	# 	# model testing
	# 	for (data, target) in test_dl:
	# 		model.eval()
	# 		with torch.no_grad():
	# 			output = model(data)
	# 			# test_loss = criterion(output, target)
	# 		test_acc = sum([1 if output[i].max(0)[1] == target[i] else 0 \
	# 			for i in range(output.shape[0])]) / output.shape[0]
		
	# 	if epoch % PRINT_INTERVAL == 0:
	# 		pass
	# 		# send weight of each layer to tensorboard
	# 		# send gradient of input to tensorboard
	# 		# send entropy of output to tensorboard


if __name__ == '__main__':
	main()