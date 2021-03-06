import argparse
import gzip
import math
import os
import pickle
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tp5_preprocess import *

PRINT_INTERVAL = 50
CUDA = False
VOCAB_SIZE = 1000
EMBEDDING_DIM = 50


class CNN(nn.Module):
    def __init__(self):
        """

        """
        super(CNN, self).__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.output_size = 1
        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=EMBEDDING_DIM,
                out_channels=16,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(32, 2)
        )

    def output_size_function(self, input_size, kernel_size, padding, stride):
        return ((input_size - kernel_size + (padding * 2)) / stride) + 1

    def forward(self, X):
        output = self.embedding(X)
        output = self.features(output.view(X.size(0), -1, X.size(1)))
        output = output.view(X.size(0), -1, 32)
        output = self.classifier(output)
        output, indices = torch.max(output, dim=1)
        return output


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, keep_all=False):
        self.reset()
        self.data = None
        if keep_all:
            self.data = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.data is not None:
            self.data.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_data(filename):
    """ Read pth file
    :param filename: file name
    :returns: tensors
    :rtype: torch.tensor
    """

    with gzip.open(filename, 'rb') as f:
        return torch.load(f)


def get_corpus_sentiment140(batch_size):
    """FIXME! briefly describe function

    :param batch_size:
    :returns:
    :rtype:

    """
    train_corpus = load_data("corpus/train-1000.pth")
    test_corpus = load_data("corpus/test-1000.pth")

    train_loader = torch.utils.data.DataLoader(
        train_corpus, batch_size=batch_size,
        shuffle=True, collate_fn=train_corpus.collate)

    test_loader = torch.utils.data.DataLoader(
        test_corpus, batch_size=batch_size,
        shuffle=True, collate_fn=test_corpus.collate)

    return train_loader, test_loader


def accuracy(Yhat, Y):
    """FIXME! briefly describe function

    :param Yhat:
    :param Y:
    :returns:
    :rtype:

    """
    acc = (torch.argmax(Yhat, dim=1) ==
           torch.argmax(Y, dim=1)).sum() * 100 / len(Y)

    return acc


def compute_loss(criterion, Yhat, Y):
    """FIXME! briefly describe function

    :param criterion:
    :param Yhat:
    :param Y:
    :returns:
    :rtype:

    """
    return criterion(Yhat, torch.argmax(Y.long(), dim=1))


def epoch(data, model, criterion, optimizer=None):
    """ Fait une passe (appelée epoch en anglais) sur les données
    `data` avec le modèle `model`. Evalue `criterion` comme loss.
    Si `optimizer` est fourni, effectue une epoch d'apprentissage
    en utilisant l'optimiseur donné, sinon, effectue une epoch
    d'évaluation (pas de backward) du modèle.


    :param data:
    :param model:
    :param criterion:
    :param optimizer:

    """

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
        target_transform = torch.Tensor(
            [[0, 1] if y == 0 else [1, 0] for y in target])
        loss = compute_loss(criterion, output, target_transform)

        # backward si on est en "train"
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # calcul des metriques
        acc = accuracy(output, target_transform)
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


def main(params):
    """
    :param params : {"batch_size": 128, "epochs": 5, "lr": 0.1,
                     path": '/tmp/datasets/mnist'}
    """

    # define model, loss, optim
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), params.lr)

    if CUDA:  # si on fait du GPU, passage en CUDA
        model = model.cuda()
        criterion = criterion.cuda()

    # On récupère les données
    train_loader, test_loader = get_corpus_sentiment140(
        params.batch_size)

    # init plots
    global writer
    writer = SummaryWriter()
    loss_train, acc_train, loss_test, acc_test = 0, 0, 0, 0

    global pd_train_loss
    pd_train_loss = pd.DataFrame(columns=['train_loss'])

    global index_batch_train_loss
    index_batch_train_loss = 0

    # Declare DataFrame for saving results
    cols_acc_loss_avg = [
        'loss_train_avg',
        'loss_test_avg',
        'acc_train_avg',
        'acc_test_avg']
    pd_acc_loss_avg = pd.DataFrame(columns=cols_acc_loss_avg)

    # On itère sur les epochs
    for i in range(params.epochs):
        print("=================\n=== EPOCH " +
              str(i + 1) + " =====\n=================\n")
        # Phase de train
        acc_train, loss_train = epoch(
            train_loader, model, criterion, optimizer)
        # Phase d'evaluation
        acc_test, loss_test = epoch(test_loader, model, criterion)

        # Plot Tensorboard Results
        writer.add_scalar('Epochs/Loss/train', loss_train.avg, i)
        writer.add_scalar('Epochs/Accuracy/train', acc_train.avg, i)
        writer.add_scalar('Epochs/Loss/test', loss_test.avg, i)
        writer.add_scalar('Epochs/Accuracy/test', acc_test.avg, i)

        # Save results
        pd_acc_loss_avg = pd_acc_loss_avg.append(dict(zip(
            cols_acc_loss_avg, [loss_train.avg, loss_test.avg, acc_train.avg, acc_test.avg])), ignore_index=True)

    pd_acc_loss_avg.to_csv(
        'results/acc_loss.csv',
        index=False)
    pd_train_loss.to_csv(
        'results/batches_train_loss.csv',
        index=False)


if __name__ == "__main__":
    # Paramètres en ligne de commande
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epochs',
        default=5,
        type=int,
        metavar='N',
        help='number of total epochs to run')
    parser.add_argument(
        '--batch-size',
        default=128,
        type=int,
        metavar='N',
        help='mini-batch size (default: 128)')
    parser.add_argument(
        '--lr',
        default=0.1,
        type=float,
        metavar='LR',
        help='learning rate')
    parser.add_argument(
        '--cuda',
        dest='cuda',
        action='store_true',
        help='activate GPU acceleration')

    args = parser.parse_args()
    if args.cuda:
        CUDA = True
        cudnn.benchmark = True

    main(args)

    input("done")
