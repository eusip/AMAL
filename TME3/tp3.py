# standard libraries
import os, shutil

# local libraries
from classes import Autoencoder, HighwayNetwork

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

# generate a view of an input image tensor from MNIST
def to_img(x):
	x = x.view(x.size(0), 1, 28, 28)
	return x

if __name__ == '__main__':
	for root in ['../data', './images', './checkpoints']:
		if not os.path.exists(root):
			os.mkdir(root)
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# tensorboard writer
	writer = SummaryWriter()

	# hyper-parameters
	transform = transforms.Compose([transforms.ToTensor(),
									transforms.Normalize((0.1307,), (0.3081,))])
	batch_size = 64
	n_epochs = 5
	lr = 1e-3

	# load data
	train_ds = datasets.MNIST('../data', download=False, train=True, 
				transform=transform)
	test_ds = datasets.MNIST('../data', download=False, train=False,
				transform=transform)

	train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
	test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True)

	# print('=> total number of training batches: {}'.format(len(train_dl)))
	# print('=> total number of test batches: {}'.format(len(test_dl)))

	# show images from each batch
	# img_iter = iter(train_dl)
	# images, _ = img_iter.next()
	# figure = plt.figure()
	# num_of_images = 60

	# for index in range(1, num_of_images + 1):
	#     plt.subplot(6, 10, index)
	#     plt.axis('off')
	#     plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')

	########################### autoencoder ############################

	# model = Autoencoder().to(device)
	# criterion = nn.BCELoss()
	# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

	# if os.path.isfile('./checkpoints/checkpoint_ae.pth.tar'):
	# 	print("=> loading checkpoint")
	# 	checkpoint = torch.load('./checkpoints/checkpoint_ae.pth.tar')
	# 	model.load_state_dict(checkpoint['state_dict'])
	# 	optimizer.load_state_dict(checkpoint['optimizer'])
	# 	start_epoch = checkpoint['epoch']
	# 	print("=> loaded checkpoint (epoch {})"
	# 				.format(checkpoint['epoch']))
	# else:
	# 	start_epoch = 0

	# print("=> beginning training")
	# for epoch in range(start_epoch, n_epochs):
	# 	for data in train_dl:
	# 		img, _ = data
	# 		img = img.view(img.size(0), -1)
	# 		# variable creation
	# 		img = img.to(device)  # img = Variable(img).to(device)
	# 		# forward pass
	# 		output = model(img)
	# 		# compute BCE loss
	# 		loss = criterion(output, img)
	# 		# compute MSE loss
	# 		MSE_loss = nn.MSELoss()(output, img)
	# 		# zero gradients
	# 		optimizer.zero_grad()
	# 		# backward pass
	# 		loss.backward()
	#		if epoch % 100 == 0:
	# 			# update parameters
	# 			optimizer.step()
	# 			# iteration += 1
	# 		torch.save({
	# 					'state_dict': model.state_dict(),
	# 					'optimizer': optimizer.state_dict(),
	# 					'epoch': epoch + 1
	# 				}, './checkpoints/checkpoint_ae.pth.tar')

	# 	print('=> epoch [{}/{}], BCEloss:{:.4f}, MSE_loss:{:.4f}'
	# 		  .format(epoch + 1, n_epochs, loss.data, MSE_loss.data))
	# 	if epoch % 10 == 0:
	# 		x = to_img(img.cpu().data)
	# 		x_hat = to_img(output.cpu().data)
	# 		save_image(x, './images/ae_x_{}.png'.format(epoch))
	# 		save_image(x_hat, './images/ae_x_hat_{}.png'.format(epoch))
	# 		img 

	######## write out tensors in order to view in tensorboard #########
	# images, _ = next(iter(train_dl))
	# images = images.view(images.shape[0], -1) # access view of flattened tensor
	# grid = torchvision.utils.make_grid(images)
	# writer.add_image('images', grid, 0)
	# writer.add_graph(model, images)
	# writer.close()

	######################### highway network ##########################

	# nb_digits = 10
	# y_onehot = torch.FloatTensor(batch_size, nb_digits)
	# print(train_dl.dataset.data.size())

	model = HighwayNetwork(28 * 28, 10, 2).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

	if os.path.isfile('./checkpoints/checkpoint_hn.pth.tar'):
		print("=> loading checkpoint")
		checkpoint = torch.load('./checkpoints/checkpoint_hn.pth.tar')
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		start_epoch = checkpoint['epoch']
		print("=> loaded checkpoint (epoch {})"
					.format(checkpoint['epoch']))
	else:
		start_epoch = 0

	print("=> beginning training")
	for epoch in range(start_epoch, n_epochs):
		i = 0
		acc = 0
		test_acc = 0
		for (data, target) in train_dl:
			data = data.reshape(-1, 28 * 28)
			model.train()
			output = model.forward(data)
			loss = criterion(output, target)
			optimizer.zero_grad()
			loss.backward()
			if epoch % 100 == 0:
				optimizer.step()
			torch.save({
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch + 1
                            }, './checkpoints/checkpoint_hn.pth.tar')
			i += 1
			acc = sum([1 if output[i].max(0)[1] == target[i] else 0 \
				for i in range(output.shape[0])]) / output.shape[0]


		print("Pieces of data {}".format(i))
		print("Epoch {} : Loss {:.4f}".format(epoch, loss.mean().item()))
		print("Training Accuracy {}%".format(acc * 100))

		# for (data, target) in test_dl:
		# 	model.eval()
		# 	with torch.no_grad():
		# 		output = model(data)
		# 		# test_loss = criterion(output, target)
		# 	test_acc = sum([1 if output[i].max(0)[1] == target[i] else 0 \
		# 		for i in range(output.shape[0])]) / output.shape[0]

		# print("Test Accuracy {}%".format(acc * 100))

		if epoch % 1 == 0:
			writer.add_scalar('Loss/tp3/train', loss, epoch)
			writer.add_scalar('Accuracy/tp3/train', acc, epoch)
			# writer.add_scalar('Accuracy/tp3/test', test_acc, epoch)

######################## highway network scrath ########################

# for i, (data, target) in enumerate(train_dl):
# 		data = data.reshape(-1, 28 * 28)
# 		# set model to train
# 		model.train()
# 		#forward_pass = torch.nn.Softmax()(torch.nn.Linear(28 * 28, nb_digits)(model.forward(data)))
# 		output = model.forward(data)
# 		loss = criterion(output, target)
# 		loss.backward()
# 		# for param in model.parameters():
# 		#    print(param.grad.data.sum())

# 		#print(model.linear_term_list[0].weight)
# 		optimizer.step()
# 		acc = sum([1 if output[i].max(0)[1] == target[i]
#                     else 0 for i in range(output.shape[0])]) / output.shape[0]

# 		print("Epoch {} : Loss {:.4f}".format(i, loss.mean().item()))
# 		print("Accuracy {}%".format(acc * 100))

# 		#print(list(model.parameters())[0].grad)
