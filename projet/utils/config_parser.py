import argparse
from collections import OrderedDict

config_dict = OrderedDict(
	# === DATASET ===
	dataset="Cifar10",
	# added data_root parameter for passing Google Drive folder path -EU
	data_root="./data_root.tmp",
	model="bert",
	load_checkpoint_file=None,
	no_cuda=False,

	# === OPTIMIZER ===
	optimizer="SGD",
	optimizer_cosine_lr=True,
	optimizer_warmup_ratio=0.05,  # period of linear increase for lr scheduler
	optimizer_decay_at_epochs=[80, 150, 250],
	optimizer_decay_with_factor=10.0,
	optimizer_learning_rate=0.1,
	optimizer_momentum=0.9,
	optimizer_weight_decay=0.0001,
	batch_size=100,
	num_epochs=300,
	seed=42,

	# === From BERT ===
	vocab_size_or_config_json_file=-1,
	hidden_size=400,  # 768,
	# dimension of the position embedding for relative attention, if -1 will
	# default to  hidden_size
	position_encoding_size=-1,
	num_hidden_layers=6,
	num_attention_heads=9,
	# self-attention network parameter; output_size of all sublayers of the
	# transformer -EU
	intermediate_size=512,
	hidden_act="gelu",
	hidden_dropout_prob=0.1,
	attention_probs_dropout_prob=0.1,
	max_position_embeddings=16,
	type_vocab_size=2,
	initializer_range=0.02,
	layer_norm_eps=1e-12,

	# === BERT IMAGE===
	add_positional_encoding_to_input=False,
	use_learned_2d_encoding=False,
	# share learned relative position encoding for all layers
	share_position_encoding=False,
	# use attention between pixel values instead of only positional
	use_attention_data=False,
	use_gaussian_attention=True,
	attention_isotropic_gaussian=True,
	# remove heads with Sigma^{-1} close to 0 or very singular (kappa > 1000)
	# at epoch 0
	prune_degenerated_heads=False,
	# reinitialize randomly the heads mentioned above
	reset_degenerated_heads=False,
	# original heads (not pruned/reinit) position are fixed to their original
	# value
	fix_original_heads_position=False,
	# original heads (not pruned/reinit) value matrix are fixed to their
	# original value
	fix_original_heads_weights=False,
	# penalize singular covariance gaussian attention
	gaussian_spread_regularizer=0.,

	gaussian_init_sigma_std=0.01,
	gaussian_init_mu_std=2.,
	# use a computational trick for gaussian attention to avoid computing the
	# attention probas
	attention_gaussian_blur_trick=False,
	# concatenate the pixels value by patch of pooling_concatenate_size x
	# pooling_concatenate_size to redude dimension
	pooling_concatenate_size=2,
	pooling_use_resnet=False,

	# === LOGGING ===
	only_time_one_epoch=False,  # show timer after 1 epoch and stop
	only_list_parameters=False,
	num_keep_checkpoints=0,
	plot_attention_positions=True,
	output_dir="./output.tmp",
)


# Paramètres en ligne de commande
parser = argparse.ArgumentParser()
parser.add_argument(
	'--config_dict',
	action='store_false',
	help='Use either parsed arguments or the defined dictionary config_dict')
parser.add_argument(
	'--dataset',
	default='Cifar10',
	type=str,
	metavar='DIR',
	help='dataset')
parser.add_argument(
	'--data_root',
	default="./data_root.tmp",
	type=str,
	metavar='DIR',
	help='added data_root parameter for passing Google Drive folder path -EU')
parser.add_argument(
	'--model',
	default="bert",
	type=str,
	metavar='MODEL',
	help='Used model')
parser.add_argument(
	'--load_checkpoint_file',
	default=None,
	type=str)
parser.add_argument(
	'--no_cuda',
	dest='no_cuda',
	action='store_false',
	help='activate GPU acceleration')

# === OPTIMIZER ===
parser.add_argument(
	'--optimizer',
	default="SGD",
	type=str,
	metavar='OPTIMIZER',
	help='optimizer')
parser.add_argument(
	'--optimizer_cosine_lr',
	action='store_true',
	help='Optimizer cosine learning rate')
# parser.add_argument(
#     '--no-optimizer_cosine_lr',
#     action='store_false',
#     help='Optimizer cosine learning rate')
parser.add_argument(
	'--optimizer_warmup_ratio',
	default=0.0,
	type=float,
	help='period of linear increase for lr scheduler')
parser.add_argument(
	'--optimizer_decay_at_epochs',
	default=[80, 150, 250],
	type=list,
	help='Optimizer decay at epochs')
parser.add_argument(
	'--optimizer_decay_with_factor',
	default=10.0,
	type=float,
	help='Optimizer decay with factor')
parser.add_argument(
	'--optimizer_learning_rate',
	default=0.1,
	type=float,
	help='Optimizer learning rate')
parser.add_argument(
	'--optimizer_momentum',
	default=0.9,
	type=float,
	help='Optimizer momentum')
parser.add_argument(
	'--optimizer_weight_decay',
	default=0.0001,
	type=float,
	help='Optimizer weight decay')
parser.add_argument(
	'--batch_size',
	default=300,
	type=int,
	metavar='N',
	help='Batch size')
parser.add_argument(
	'--num_epochs',
	default=300,
	type=int,
	metavar='N',
	help='Number of epochs')
parser.add_argument(
	'--seed',
	default=42,
	type=int,
	metavar='N',
	help='For reproductibility')

# === From BERT ===
parser.add_argument(
	'--vocab_size_or_config_json_file',
	default=-1,
	type=int,
	metavar='N',
	help='Vocabulary size or config json file')
parser.add_argument(
	'--hidden_size',
	default=128,
	type=int,
	metavar='N',
	help='Hidden size')
parser.add_argument(
	'--position_encoding_size',
	default=-1,
	type=int,
	metavar='N',
	help='dimension of the position embedding for relative attention, if -1 will default to  hidden_size')
parser.add_argument(
	'--num_hidden_layers',
	default=2,
	type=int,
	metavar='N',
	help='Number of hidden layers')
parser.add_argument(
	'--num_attention_heads',
	default=8,
	type=int,
	metavar='N',
	help='Number of attention heads')
parser.add_argument(
	'--intermediate_size',
	default=512,
	type=int,
	metavar='N',
	help='self-attention network parameter; output_size of all sublayers of the transformer -EU')
parser.add_argument(
	'--hidden_act',
	default='gelu',
	type=str,
	help='Hidden act')
parser.add_argument(
	'--hidden_dropout_prob',
	default=0.1,
	type=float,
	help='Hidden dropout prob')
parser.add_argument(
	'--attention_probs_dropout_prob',
	default=0.1,
	type=float,
	help='Attention probs dropout prob')
parser.add_argument(
	'--max_position_embeddings',
	default=16,
	type=int,
	help='Maximum position embeddings')
parser.add_argument(
	'--type_vocab_size',
	default=2,
	type=int,
	help='Type vocabulary size')
parser.add_argument(
	'--initializer_range',
	default=0.02,
	type=float,
	help='Initializer range')
parser.add_argument(
	'--layer_norm_eps',
	default=1e-12,
	type=float,
	help='Layer normalization eps')

# === BERT IMAGE===
parser.add_argument(
	'--add_positional_encoding_to_input',
	action='store_false',
	help='Add positional encoding to input')
parser.add_argument(
	'--use_learned_2d_encoding',
	action='store_false',
	help='Use learned 2d encoding')
parser.add_argument(
	'--share_position_encoding',
	action='store_false',
	help='share learned relative position encoding for all layers')
parser.add_argument(
	'--use_attention_data',
	action='store_false',
	help='use attention between pixel values instead of only positional')
parser.add_argument(
	'--use_gaussian_attention',
	action='store_true',
	help='Use gaussian attention')
parser.add_argument(
	'--attention_isotropic_gaussian',
	action='store_false',
	help='Attention isotropic gaussian')
parser.add_argument(
	'--prune_degenerated_heads',
	action='store_false',
	help='remove heads with Sigma^{-1} close to 0 or very singular (kappa > 1000) at epoch 0')
parser.add_argument(
	'--reset_degenerated_heads',
	action='store_false',
	help='reinitialize randomly the heads mentioned above')
parser.add_argument(
	'--fix_original_heads_position',
	action='store_false',
	help='original heads (not pruned/reinit) position are fixed to their original value')
parser.add_argument(
	'--fix_original_heads_weights',
	action='store_false',
	help='original heads (not pruned/reinit) value matrix are fixed to their original value')
parser.add_argument(
	'--gaussian_spread_regularizer',
	default=0.,
	type=float,
	help='penalize singular covariance gaussian attention')
parser.add_argument(
	'--gaussian_init_sigma_std',
	default=0.01,
	type=float,
	help='Gaussian init sigma std')
parser.add_argument(
	'--gaussian_init_mu_std',
	default=2.,
	type=float,
	help='Gaussian init mu std')
parser.add_argument(
	'--attention_gaussian_blur_trick',
	action='store_false',
	help='use a computational trick for gaussian attention to avoid computing the attention probas')
parser.add_argument(
	'--pooling_concatenate_size',
	default=2,
	type=int,
	help='concatenate the pixels value by patch of pooling_concatenate_size x pooling_concatenate_size to redude dimension')
parser.add_argument(
	'--pooling_use_resnet',
	action='store_false',
	help='Pooling use resnet')

# === LOGGING ===
parser.add_argument(
	'--only_time_one_epoch',
	action='store_false',
	help='show timer after 1 epoch and stop')
parser.add_argument(
	'--only_list_parameters',
	action='store_false',
	help='Only list parameters')
parser.add_argument(
	'--num_keep_checkpoints',
	default=0,
	type=int,
	help='Number of checkpoints to keep')
parser.add_argument(
	'--plot_attention_positions',
	action='store_true',
	help='Plot attention positions')
parser.add_argument(
	'--output_dir',
	default='./output.tmp',
	type=str,
	help='Output dir')
