# Projet AMAL
Authors: Ahmed BALDÉ, Ebenge USIP

This program was originally developed to run in Google Colab using a GPU but training can be executed locally by executing the file train.py. The following arguments are necessary in order to complete a training.

--dataset 'Cifar10' \
--data_root . \
--output_dir . \
--num\_keep\_checkpoints 20 \
--model 'bert' \
--load_checkpoint_file None \
--no_cuda False

Logs and Tensorboard runs are stored in the output directory.
