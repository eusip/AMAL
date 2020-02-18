# standard libraries

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

PRINT_INTERVAL = 50
CUDA = False

# hyper-parameters
BATCH_SIZE = 64
N_EPOCHS = 5
LR = 1e-3

# tensorboard writer
WRITER = SummaryWriter()

def one_step(x, h):  # x - batch x dim; h - batch x latent
	pass  # return batch x latent (batch x dimout)

class RNN(nn.Module):
	"""a recurrent neural network"""
	def __init__(self, length, batch, dim, latent):
		super(RNN, self).__init__()
		self.h = pass
		self.final_layer = nn.Linear(INPUT_SIZE, OUTPUT_SIZE)
		self.output = nn.Softmax(self.final_layer)

		def forward(self, x, h):  # x - length x batch x dim; h - batch x latent
			x = one_step()
			return x

LETTRES = string.ascii_letters + string.punctuation + string.digits + ' '
id2lettre = dict(zip(range(1, len(LETTRES) + 1), LETTRES))
id2lettre[0] = ' '  # NULL CHARACTER
lettre2id = dict(zip(id2lettre.values(), id2lettre.keys()))


def normalize(s):
	return ' '.join(c for c in unicodedata.normalize('NFD', s) if c in LETTRES)

def string2code(s):
	return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
	if type(t) != list:
		t = t.tolist()
	return ' '.join(id2lettre[i] for i in t)
	
if __name__ == '__main__':

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	batch_size = pass
	n_epoches = pass
	lr = pass

	criterion = nn.CrossEntropyLoss()
