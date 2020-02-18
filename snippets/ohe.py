# source:  https://www.youtube.com/watch?v=bvZnphPgz74

import itertools
import numpy as np 

# One-hot encoding example:
sentence = ['this','brown','fox','jumped','over','the','lazy','dog']
vocabulary = dict([(word, i) for i, word in enumerate(set(sentence))])

def one_hot_encode():
	ohe = np.zeros((len(sentence), len(vocabulary)))
	for i, word in enumerate(sentence):
		ohe[i, vocabulary[word]] = 1
		return ohe

####

sentences = '''
sam is red
hannah is not red
hannah is green
bob is green
bob not red 
sam not green
sarah is red
sarah not green
'''.strip().split('\n')
is_green = np.asarray([[0, 1, 1, 1, 1, 0, 0, 0]], dtype='int32').T

for s, g in zip(sentences, is_green):
	print(s, '-->', g)

tokenize = lambda x: x.strip().lower().split(' ')
sentences_tokenized = [tokenize(sentence) for sentence in sentences]
words = set(itertools.chain(*sentences_tokenized))

word2idx = dict((v, i ) for i, v in enumerate(words))
idx2word = list(words)
print('Vocab')
print(word2idx, end='\n\n')

to_idx = lambda x: [word2idx(word) for word in x]  # convert a list of words to a list of indices
sentences_idx = [to_idx(sentence) for sentence in sentences_tokenized]
sentences_array = np.asarray(sentences_idx, dtype='int32')
print('Sentences')
print(sentences_array)

# sentence_maxlen = 3
n_words = len(words)
n_embed_dims = 2 