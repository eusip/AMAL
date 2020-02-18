import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader

from utils import TempDataset

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
cudnn.benchmark = True

# Parameters
params = {'batch_size': 64,
          'shuffle': True}
max_epochs = 100

# Datasets
# partition =  # IDs
# labels =  # Labels

# Generators
# training_set = Dataset(partition['train'], labels)
# training_generator = data.DataLoader(training_set, **params)

# validation_set = Dataset(partition['validation'], labels)
# validation_generator = data.DataLoader(validation_set, **params)

temperatures_dataset = TempDataset('data/tempAMAL_train.csv')
training_generator = DataLoader(temperatures_dataset, **params)

print(len(temperatures_dataset))
# Loop over epochs
# for epoch in range(max_epochs):
    # Training
    # for local_batch, local_labels in training_generator:
        # Transfer to GPU
        # local_batch, local_labels = local_batch.to(
            # device), local_labels.to(device)

        # print(local_batch, local_labels)
        # Model computations

    # Validation
    # with torch.set_grad_enabled(False):
    #     for local_batch, local_labels in validation_generator:
    #         # Transfer to GPU
    #         local_batch, local_labels = local_batch.to(
    #             device), local_labels.to(device)

        # Model computations
