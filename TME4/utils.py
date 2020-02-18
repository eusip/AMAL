import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from RNN import RNNClassification

# The data must be normalized from 0 to 1 per Baskiotis
# J'ai oublié de dire à mon groupe (5eme étage) qu'il fallait bien sur 
# normaliser les données(toujours le faire quand c'est possible).
# https://channel.lip6.fr/master-1-dac/pl/jxj15sqrcjgrundjo566apso1y
# 
# Bonjour, pour le tme4,  on va essayer de vous faire un retour rapide 
# sur votre code pour voir si tout va bien, on  a donc mis deux rendus 
# sur le moodle, un pr le groupe du 5eme étage et un autre pour le 4ème. 
# Pour ceux qui peuvent mettre leur rendu avant lundi, on pourra vous 
# faire un retour mardi.
# https://channel.lip6.fr/master-1-dac/pl/eo9x5mayg3nozqs3saezrdktgy

# Pour le forecasting, ceux qui veulent tester les RNNs vs baseline, vous
#  pouvez tester ARMA/ARIMA simple modele lineaire: 
# https: // en.wikipedia.org/wiki/Autoregressive_integrated_moving_average

# ta normalisation n'est pas bonne il faut normaliser par rapport à un 
# min et max de température, pas sur les données, pas sur une colonne de
#  données. Ca n'a pas de sens sinon (normalisation dépendante de la 
# longueur de la séquence, et la série  perd toute sa sémantique, ca 
# pourrait expliquer tes problemes de convergences). Par ailleurs je ne 
# te conseille pas  dans le one_step et le forward de retourner l'état 
# décodé, mais l'état caché. Ca rendra plus générique ton rnn.
# https: // channel.lip6.fr/master-1-dac/pl/9d1cfix5o3fwpj5i4m4cc1q4zc

# par ailleurs n'utilise pas SGD  mais Adam, tu convergeras plus vite 
# (tester avec ton code et ca marche).
# https://channel.lip6.fr/master-1-dac/pl/p319qaqjwt83pkaxb4pjau5mke

class TempDataset(Dataset):
    """
    Parameters
    ----------
    path_file: The path to the csv file. \n
    seq_length: The predefined fixed length for the temperature sequence (rows). \n
    nb_cities: The number of cities being considered (columns). \n
    train: Boolean for either the training or testing dataset. \n
    seed: Seed value for np.random.

    """
    def __init__(
            self,
            path_file='data/tempAMAL_train.csv',
            seq_length=10,
            nb_cities=5,
            train=True,  # not needed?     
            seed=42):

        data = pd.read_csv(path_file)
        data = data.drop('datetime', axis=1)
        all_X = data.values
        N, D = all_X.shape
        np.random.seed(seed)

        cities_indexes = np.random.choice(
            range(D), size=nb_cities, replace=False)
        X_chosen_cities = all_X[:, cities_indexes]
        temp = X_chosen_cities[0:seq_length]  # tensor - [records by cities]

        # scale array values        
        with np.nditer(temp, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = (x - np.amin(temp))/(np.amax(temp) - np.amin(temp))
                
        # tensor containing column subset of dataset  --transpose of temp
        X = torch.tensor([temp[:, city]
                          for city in range(len(cities_indexes))])
        # 1-D tensor containing proper classifications --relevant city values
        # repeated for length of temp array
        Y = torch.arange(len(cities_indexes)).repeat(N - seq_length + 1)

        for i in range(1, N - seq_length + 1):  # skip first row - column names
            temp = X_chosen_cities[i: i + seq_length]
            # scale array values
            with np.nditer(temp, op_flags=['readwrite']) as it:
                for x in it:
                    x[...] = (x - np.amin(temp))/(np.amax(temp) - np.amin(temp))

            X_temp = torch.tensor([temp[:, city]
                                   for city in range(len(cities_indexes))])
            X = torch.cat([X, X_temp], dim=0)

        self.X, self.Y = X, Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class TrumpDataset():
    """
    to be continued.
    """
    def __init__(self, path_file):
        
        data = pd.read_csv(path_file)

    def normalize(s):
    	return ' '.join(c for c in unicodedata.normalize('NFD', s) if c in LETTRES)


    def string2code(s):
    	return torch.tensor([lettre2id[c] for c in normalize(s)])


    def code2string(t):
    	if type(t) != list:
    		t = t.tolist()
    	return ' '.join(id2lettre[i] for i in t)

    LETTRES = string.ascii_letters + string.punctuation + string.digits + ' '
    id2lettre = dict(zip(range(1, len(LETTRES) + 1), LETTRES))
    id2lettre[0] = ' '  # NULL CHARACTER
    lettre2id = dict(zip(id2lettre.values(), id2lettre.keys()))


