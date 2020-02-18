import re
from pathlib import Path
from torch.utils.data import Dataset
from datamaestro import prepare_dataset

EMBEDDING_SIZE = 50

ds = prepare_dataset("edu.standford.aclimdb")
word2id, embeddings = prepare_dataset('edu.standford.glove.6b.%d' % EMBEDDING_SIZE).load()

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

train_data = FolderText(ds.train.classes, tokenizer, load=False)
test_data = FolderText(ds.test.classes, tokenizer, load=False)
