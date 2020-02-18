import re
from pathlib import Path

from datamaestro import prepare_dataset
from torch.utils.data import Dataset

class FolderText(Dataset):
    def __init__(self, classes, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = list(classes.keys())
        for label, folder in classes.items():
            for file in folder.glob("*.txt"):
                self.files.append(file)
                self.filelabels.append(label)

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        return self.tokenizer(self.files[ix].read_text()), self.filelabels[ix]


WORDS = re.compile(r"\S+")


def tokenizer(t):
    return list([x for x in re.findall(WORDS, t.lower())])

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
