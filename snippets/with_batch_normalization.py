import argparse
import os
import time

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pandas as pd

from tme6 import *

# Permet de télécharger CIFAR10 depuis les serveurs UPMC
datasets.CIFAR10.url = "http://webia.lip6.fr/~robert/cours/rdfia/cifar-10-python.tar.gz"


PRINT_INTERVAL = 50
CUDA = False


class AlexNetPrime(nn.Module):
    def __init__(self):
        """ Architecture de réseaux de neurones proche de AlexNet

        """
        super(AlexNetPrime, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(64, 64, 5, stride=1, padding=2),
            nn.BatchNorm2d(64),            
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 1000),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1000, 10)
        )

    def forward(self, input):
        bsize = input.size(0)
        output = self.features(input)
        output = output.view(bsize, -1)
        output = self.classifier(output)

        return output


def get_dataset_cifar10(batch_size, path):
    """
    Cette fonction charge le dataset et effectue des transformations sur chaqu
    image (listées dans `transform=...`).
    """
    train_dataset = datasets.CIFAR10(
        path, train=True, download=True, transform=transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(size=32),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                  std=[0.202, 0.199, 0.201])]))

    val_dataset = datasets.CIFAR10(path, train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.CenterCrop(size=28),
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           mean=[0.491, 0.482, 0.447],
                                           std=[0.202, 0.199, 0.201])
                                   ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=CUDA,
        num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=CUDA,
        num_workers=4)

    return train_loader, val_loader


def epoch(data, model, criterion, optimizer=None):
    """
    Fait une passe (appelée epoch en anglais) sur les données `data` avec le
    modèle `model`. Evalue `criterion` comme loss.
    Si `optimizer` est fourni, effectue une epoch d'apprentissage en utilisant
    l'optimiseur donné, sinon, effectue une epoch d'évaluation (pas de backward)
    du modèle.
    """

    # indique si le modele est en mode eval ou train (certaines couches se
    # comportent différemment en train et en eval)
    model.eval() if optimizer is None else model.train()

    # objets pour stocker les moyennes des metriques
    avg_loss = AverageMeter()
    avg_top1_acc = AverageMeter()
    avg_top5_acc = AverageMeter()
    avg_batch_time = AverageMeter()
    global loss_plot
    global pd_train_loss
    # on itere sur les batchs du dataset
    tic = time.time()
    for i, (input, target) in enumerate(data):
        if CUDA:  # si on fait du GPU, passage en CUDA
            input = input.cuda()
            target = target.cuda()

        # forward
        output = model(input)
        loss = criterion(output, target)

        # backward si on est en "train"
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # calcul des metriques
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        batch_time = time.time() - tic
        tic = time.time()

        # mise a jour des moyennes
        avg_loss.update(loss.item())
        avg_top1_acc.update(prec1.item())
        avg_top5_acc.update(prec5.item())
        avg_batch_time.update(batch_time)
        if optimizer:
            pd_train_loss = pd_train_loss.append(
                dict({'train_loss': avg_loss.val}), ignore_index=True)
            loss_plot.update(avg_loss.val)
        # affichage des infos
        if i % PRINT_INTERVAL == 0:
            print(
                '[{0:s} Batch {1:03d}/{2:03d}]\t'
                'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:5.1f} ({top1.avg:5.1f})\t'
                'Prec@5 {top5.val:5.1f} ({top5.avg:5.1f})'.format(
                    "EVAL" if optimizer is None else "TRAIN",
                    i,
                    len(data),
                    batch_time=avg_batch_time,
                    loss=avg_loss,
                    top1=avg_top1_acc,
                    top5=avg_top5_acc))
            if optimizer:
                loss_plot.plot()

    # Affichage des infos sur l'epoch
    print('\n===============> Total time {batch_time:d}s\t'
          'Avg loss {loss.avg:.4f}\t'
          'Avg Prec@1 {top1.avg:5.2f} %\t'
          'Avg Prec@5 {top5.avg:5.2f} %\n'.format(
              batch_time=int(avg_batch_time.sum), loss=avg_loss,
              top1=avg_top1_acc, top5=avg_top5_acc))

    return avg_top1_acc, avg_top5_acc, avg_loss


def main(params):

    # ex de params :
    #   {"batch_size": 128, "epochs": 5, "lr": 0.1, "path": '/tmp/datasets/mnist'}

    # define model, loss, optim
    model = AlexNetPrime()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), params.lr, momentum=0.9)

    lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    if CUDA:  # si on fait du GPU, passage en CUDA
        model = model.cuda()
        criterion = criterion.cuda()

    # On récupère les données
    train, test = get_dataset_cifar10(params.batch_size, params.path)

    # init plots
    plot = AccLossPlot()
    global loss_plot
    loss_plot = TrainLossPlot()

    global pd_train_loss
    pd_train_loss = pd.DataFrame(columns=['train_loss'])

    # Declare DataFrame for saving results
    cols_acc_loss_avg = [
        'loss_train_avg',
        'loss_test_avg',
        'top1_acc_train_avg',
        'top1_acc_test_avg']
    pd_acc_loss_avg = pd.DataFrame(columns=cols_acc_loss_avg)

    # On itère sur les epochs
    for i in range(params.epochs):
        print("=================\n=== EPOCH " +
              str(i + 1) + " =====\n=================\n")
        # Phase de train
        top1_acc, avg_top5_acc, loss = epoch(
            train, model, criterion, optimizer)
        # Phase d'evaluation
        top1_acc_test, top5_acc_test, loss_test = epoch(test, model, criterion)

        lr_sched.step()

        # Save results
        pd_acc_loss_avg = pd_acc_loss_avg.append(dict(zip(
            cols_acc_loss_avg, [loss.avg, loss_test.avg, top1_acc.avg, top1_acc_test.avg])), ignore_index=True)

        # plot
        plot.update(loss.avg, loss_test.avg, top1_acc.avg, top1_acc_test.avg)

    pd_acc_loss_avg.to_csv(
        'results/batch_norm_loss_acc_0.05.csv',
        index=False)
    pd_train_loss.to_csv(
        'results/batch_norm_train_loss_0.05.csv',
        index=False)


if __name__ == '__main__':

    # Paramètres en ligne de commande
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        default='/tmp/datasets/cifar10',
        type=str,
        metavar='DIR',
        help='path to dataset')
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
