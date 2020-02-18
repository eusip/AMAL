import itertools
import logging
from tqdm import tqdm
import unicodedata
import string

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch

### Partie Tagging

logging.basicConfig(level=logging.INFO)

from datamaestro import prepare_dataset
ds = prepare_dataset('org.universaldependencies.french.gsd')

BATCH_SIZE=100

# Format de sortie
# https://pypi.org/project/conllu/

class VocabularyTagging:
    OOVID = 0

    def __init__(self, oov: bool):
        self.oov =  oov
        if oov:
            self.word2id = { "__OOV__": VocabularyTagging.OOVID }
            self.id2word = [ "__OOV__" ]
        else:
            self.word2id = {}
            self.id2word = []
    
    def __getitem__(self, i):
        return self.id2word[i]


    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return VocabularyTagging.OOVID
            raise


    def __len__(self):
        return len(self.id2word)


class TaggingDataset():
    def __init__(self, data, words: VocabularyTagging, tags: VocabularyTagging, adding=True):
        self.sentences = []

        for s in data:
            self.sentences.append(([words.get(token["form"], adding) for token in s], [tags.get(token["upostag"], adding) for token in s]))


    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, ix):
        return self.sentences[ix]

logging.info("Loading datasets...")
words = VocabularyTagging(True)
tags = VocabularyTagging(False)
train_data = TaggingDataset(ds.files["train"], words, tags, True)
dev_data = TaggingDataset(ds.files["dev"], words, tags, True)
test_data = TaggingDataset(ds.files["test"], words, tags, False)


logging.info("Vocabulary size: %d", len(words))



#### Partie traduction

PAD = 0
EOS = 1
SOS = 2
class VocabularyTrad:
    def __init__(self):
        self.word2id = {"<PAD>":PAD,"<EOS>":EOS,"<SOS>":2}
        self.id2word = {PAD:"<PAD>",EOS:"<EOS>",SOS:"<SOS>"}
    
    def get_sentence(self,sentence):
        return [self.get(x,True) for x in sentence.split(" ")]+[1]
    def get(self,w,adding=False):
        try:
            return self.word2id[w]
        except KeyError:
            if adding:
                self.word2id[w]=len(self.word2id)
                self.id2word[self.word2id[w]]=w
                return self.word2id[w]
            raise
    def __getitem__(self,i): return self.id2word[i]
    def __len__(self): return len(self.word2id)


def normalize(s):
    return ''.join(c if c in string.ascii_letters else " "
         for c in unicodedata.normalize('NFD', s.lower().strip()) 
         if  c in string.ascii_letters+" "+string.punctuation)



class TradDataset():
    def __init__(self,data,vocOrig,vocDest,adding=True,max_len=10):
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue            
            self.sentences.append((torch.tensor(vocOrig.get_sentence(orig)),torch.tensor(vocDest.get_sentence(orig))))
    def __len__(self):return len(self.sentences)
    def __getitem__(self,i): return self.sentences[i]


with open("") as f:
    lines = f.read()
vocEng = VocabularyTrad()
vocFra = VocabularyTrad()
datatrain = TradDataset(lines,vocEng,vocFra)
