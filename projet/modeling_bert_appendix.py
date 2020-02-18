import copy
import json
import logging
import math
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.parameter import Parameter
import random
from modeling_bert import *
import torchvision.models as models
from torch.autograd import Variable
from enum import Enum

from utils.timer import default

timer = default()

logger = logging.getLogger(__name__)

class MaskedDataset(Dataset):
	"""
	Wrap a dataset of images and append a random mask to each sample
	"""

	def __init__(self, dataset, mask_size):
		self.dataset = dataset
		self.mask_size = mask_size

	def __getitem__(self, item):
		sample = self.dataset[item]
		image = sample[0]
		_, width, height = image.shape

		batch_mask = torch.ones([width, height], dtype=torch.uint8)
		mask_left = np.random.randint(0, width - self.mask_size)
		mask_top = np.random.randint(0, height - self.mask_size)
		batch_mask[mask_left : mask_left + self.mask_size, mask_top : mask_top + self.mask_size] = 0

		return sample + (batch_mask,)

	def __len__(self):
		return len(self.dataset)


class ResBottom(nn.Module):
	def __init__(self, origin_model, block_num=1):
		super(ResBottom, self).__init__()
		self.seq = nn.Sequential(*list(origin_model.children())[0 : (4 + block_num)])

	def forward(self, batch):
		return self.seq(batch)


class BertImage(nn.Module):
	"""
	Wrapper for a Bert encoder
	"""

	def __init__(self, config, num_classes):
		super().__init__()
		self.timer = default()

		self.with_resnet = config["pooling_use_resnet"]
		self.hidden_size = config["hidden_size"]
		self.pooling_concatenate_size = config["pooling_concatenate_size"]
		assert (config["pooling_concatenate_size"] == 1) or (
			not config["pooling_use_resnet"]
		), "Use either resnet or pooling_concatenate_size"


		if self.with_resnet:
			res50 = models.resnet50(pretrained=True)
			self.extract_feature = ResBottom(res50)

			# compute downscale factor and channel at output of ResNet
			_, num_channels_in, new_width, new_height = self.extract_feature(
				torch.rand(1, 3, 1024, 1024)
			).shape
			self.feature_downscale_factor = 1024 // new_width
		elif self.pooling_concatenate_size > 1:
			num_channels_in = 3 * (self.pooling_concatenate_size ** 2)
		else:
			num_channels_in = 3

		bert_config = BertConfig.from_dict(config)

		self.features_upscale = nn.Linear(num_channels_in, self.hidden_size)
		# self.features_downscale = nn.Linear(self.hidden_size, num_channels_in)

		self.encoder = BertEncoder(bert_config)
		self.classifier = nn.Linear(self.hidden_size, num_classes)
		# self.pixelizer = nn.Linear(self.hidden_size, 3)
		self.register_buffer("attention_mask", torch.tensor(1.0))

		# self.mask_embedding = Parameter(torch.zeros(self.hidden_size))
		# self.cls_embedding = Parameter(torch.zeros(self.hidden_size))
		# self.reset_parameters()

	# this function and references to it (line 67) can be removed -EU
	# def reset_parameters(self):
	#     # self.mask_embedding.data.normal_(mean=0.0, std=0.01)
	#     # self.cls_embedding.data.normal_(mean=0.0, std=0.01)  # TODO no hard coded
	#     # self.positional_encoding.reset_parameters()
	#     pass

	def random_masking(self, batch_images, batch_mask, device):
		"""
		with probability 10% we keep the image unchanged;
		with probability 10% we change the mask region to a normal distribution
		with 80% we mask the region as 0.
		:param batch_images: image to be masked
		:param batch_mask: mask region
		:param device:
		:return: masked image
		"""
		return batch_images
		# TODO disabled
		temp = random.random()
		if temp > 0.1:
			batch_images = batch_images * batch_mask.unsqueeze(1).float()
			if temp < 0.2:
				batch_images = batch_images + (
					((-batch_mask.unsqueeze(1).float()) + 1)
					* torch.normal(mean=0.5, std=torch.ones(batch_images.shape)).to(device)
				)
		return batch_images

	# this function and references to it can be removed -EU
	# def prune_heads(self, heads_to_prune):
	#     """ Prunes heads of the model.
	#         heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
	#     """
	#     for layer, heads in heads_to_prune.items():
	#         self.encoder.layer[layer].attention.prune_heads(heads)

	# this function and references to it can be removed -EU
	# def reset_heads(self, heads_to_reset):
	#     """ Prunes heads of the model.
	#         heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
	#     """
	#     for layer, heads in heads_to_reset.items():
	#         self.encoder.layer[layer].attention.reset_heads(heads)

	def forward(self, batch_images, batch_mask=None, feature_mask=None):

		"""
		Replace masked pixels with 0s
		If ResNet
		| compute features
		| downscale the mask
		Replace masked pixels/features by MSK token
		Use Bert encoder
		"""
		device = batch_images.device

		# compute ResNet features
		if self.with_resnet:

			with self.timer("resnet"):
				# replace masked pixels with 0, batch_images has NCHW format
				batch_features_unmasked = self.extract_feature(batch_images)

				if batch_mask is not None:
					batch_images = self.random_masking(batch_images, batch_mask, device)
					batch_features = self.extract_feature(batch_images)
				else:
					batch_features = batch_features_unmasked

				# downscale the mask
				if batch_mask is not None:
					# downsample the mask
					# mask any downsampled pixel if it contained one masked pixel originialy
					feature_mask = ~(
						F.max_pool2d((~batch_mask).float(), self.feature_downscale_factor).byte()
					)
			# permutation of NCHW to NHWC
			batch_features = batch_features.permute(0, 2, 3, 1)

		elif self.pooling_concatenate_size > 1:

			def downsample_concatenate(X, kernel):
				"""X is of shape B x H x W x C
				return shape B x (kernel*H) x (kernel*W) x (kernel*kernel*C)
				"""
				b, h, w, c = X.shape
				Y = X.contiguous().view(b, h, w // kernel, c * kernel)
				Y = Y.permute(0, 2, 1, 3).contiguous()
				Y = Y.view(b, w // kernel, h // kernel, kernel * kernel * c).contiguous()
				Y = Y.permute(0, 2, 1, 3).contiguous()
				return Y

			# permutation of NCHW to NHWC
			batch_features = batch_images.permute(0, 2, 3, 1)
			batch_features = downsample_concatenate(batch_features, self.pooling_concatenate_size)
			feature_mask = None
			if batch_mask is not None:
				feature_mask = batch_mask[
					:, :: self.pooling_concatenate_size, :: self.pooling_concatenate_size
				]

		else:
			batch_features = batch_images
			feature_mask = batch_mask
			# permutation of NCHW to NHWC
			batch_features = batch_features.permute(0, 2, 3, 1)

		# feature upscale to BERT dimension
		with self.timer("upscale"):
			batch_features = self.features_upscale(batch_features)

		# replace masked "pixels" by [MSK] token
		# if feature_mask is not None:
		# batch_features[~feature_mask] = self.mask_embedding

		# add positional embedding
		# batch_features = self.positional_encoding(batch_features)

		# replace classification token (top left pixel)
		b, w, h, _ = batch_features.shape
		# w_cls, h_cls = w // 2, h // 2
		# batch_features[:, w_cls, h_cls, :] = self.cls_embedding.view(1, -1)

		with self.timer("Bert encoder"):
			representations = self.encoder(
				batch_features,
				attention_mask=self.attention_mask,
				output_all_encoded_layers=False,  # TODO
			)[0]

		# mean pool for representation (features for classification)
		cls_representation = representations.view(b, -1, representations.shape[-1]).mean(dim=1)
		cls_prediction = self.classifier(cls_representation)

		return cls_prediction


class BertConfig(object):
	"""Configuration class to store the configuration of a `BertModel`.
	"""

	def __init__(
		self,
		vocab_size_or_config_json_file,
		hidden_size=None,
		position_encoding_size=None,
		num_hidden_layers=None,
		num_attention_heads=None,
		intermediate_size=None,
		hidden_act=None,
		hidden_dropout_prob=None,
		attention_probs_dropout_prob=None,
		max_position_embeddings=None,
		type_vocab_size=None,
		initializer_range=None,
		layer_norm_eps=None,
		use_learned_2d_encoding=None,
		share_position_encoding=None,
		use_attention_data=None,
		use_gaussian_attention=None,
		add_positional_encoding_to_input=None,
		positional_encoding=None,
		max_positional_encoding=None,
		attention_gaussian_blur_trick=None,
		attention_isotropic_gaussian=None,
		gaussian_init_sigma_std=None,
		gaussian_init_mu_std=None,
	):
		"""Constructs BertConfig.

		Args:
			vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
			hidden_size: Size of the encoder layers and the pooler layer.
			num_hidden_layers: Number of hidden layers in the Transformer encoder.
			num_attention_heads: Number of attention heads for each attention layer in
				the Transformer encoder.
			intermediate_size: The size of the "intermediate" (i.e., feed-forward)
				layer in the Transformer encoder.
			hidden_act: The non-linear activation function (function or string) in the
				encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
			hidden_dropout_prob: The dropout probabilitiy for all fully connected
				layers in the embeddings, encoder, and pooler.
			attention_probs_dropout_prob: The dropout ratio for the attention
				probabilities.
			max_position_embeddings: The maximum sequence length that this model might
				ever be used with. Typically set this to something large just in case
				(e.g., 512 or 1024 or 2048).
			type_vocab_size: The vocabulary size of the `token_type_ids` passed into
				`BertModel`.
			initializer_range: The sttdev of the truncated_normal_initializer for
				initializing all weight matrices.
			layer_norm_eps: The epsilon used by LayerNorm.
		"""
		if isinstance(vocab_size_or_config_json_file, str) or (
			sys.version_info[0] == 2 and isinstance(vocab_size_or_config_json_file, unicode)
		):
			with open(vocab_size_or_config_json_file, "r", encoding="utf-8") as reader:
				json_config = json.loads(reader.read())
			for key, value in json_config.items():
				self.__dict__[key] = value
		elif isinstance(vocab_size_or_config_json_file, int):
			self.vocab_size = vocab_size_or_config_json_file
			self.hidden_size = hidden_size
			self.position_encoding_size = position_encoding_size
			self.num_hidden_layers = num_hidden_layers
			self.num_attention_heads = num_attention_heads
			self.hidden_act = hidden_act
			self.intermediate_size = intermediate_size
			self.hidden_dropout_prob = hidden_dropout_prob
			self.attention_probs_dropout_prob = attention_probs_dropout_prob
			self.max_position_embeddings = max_position_embeddings
			self.type_vocab_size = type_vocab_size
			self.initializer_range = initializer_range
			self.layer_norm_eps = layer_norm_eps
			self.use_learned_2d_encoding = use_learned_2d_encoding
			self.use_gaussian_attention = use_gaussian_attention
			self.positional_encoding = positional_encoding
			self.max_positional_encoding = max_positional_encoding
			self.attention_gaussian_blur_trick = attention_gaussian_blur_trick
			self.attention_isotropic_gaussian = attention_isotropic_gaussian
			self.gaussian_init_sigma_std = gaussian_init_sigma_std
			self.gaussian_init_mu_std = gaussian_init_mu_std
		else:
			raise ValueError(
				"First argument must be either a vocabulary size (int)"
				"or the path to a pretrained model config file (str)"
			)

	@classmethod
	def from_dict(cls, json_object):
		"""Constructs a `BertConfig` from a Python dictionary of parameters."""
		config = BertConfig(vocab_size_or_config_json_file=-1)
		for key, value in json_object.items():
			config.__dict__[key] = value
		return config

	@classmethod
	def from_json_file(cls, json_file):
		"""Constructs a `BertConfig` from a json file of parameters."""
		with open(json_file, "r", encoding="utf-8") as reader:
			text = reader.read()
		return cls.from_dict(json.loads(text))

	def __repr__(self):
		return str(self.to_json_string())

	def to_dict(self):
		"""Serializes this instance to a Python dictionary."""
		output = copy.deepcopy(self.__dict__)
		return output

	def to_json_string(self):
		"""Serializes this instance to a JSON string."""
		return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

	def to_json_file(self, json_file_path):
		""" Save this instance to a json file."""
		with open(json_file_path, "w", encoding="utf-8") as writer:
			writer.write(self.to_json_string())


try:
	from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
	logger.info(
		"Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex ."
	)


class BertEncoder(nn.Module):
	def __init__(self, config, output_attentions=False, keep_multihead_output=False):
		super(BertEncoder, self).__init__()
		self.output_attentions = output_attentions
		layer_constructor = lambda: BertLayer(
			config, output_attentions=output_attentions, keep_multihead_output=keep_multihead_output
		)
		self.layer = nn.ModuleList([layer_constructor() for _ in range(config.num_hidden_layers)])

		if config.use_learned_2d_encoding and config.share_position_encoding:
			for layer in self.layer[1:]:
				self.layer[0].attention.self.row_embeddings = layer.attention.self.row_embeddings
				self.layer[0].attention.self.col_embeddings = layer.attention.self.col_embeddings

	def forward(
		self, hidden_states, attention_mask, output_all_encoded_layers=True, head_mask=None
	):
		all_encoder_layers = []
		all_attentions = []
		for i, layer_module in enumerate(self.layer):
			with timer(f"Bert layer {i}"):
				hidden_states = layer_module(
					hidden_states, attention_mask, head_mask[i] if head_mask is not None else None
				)
				if self.output_attentions:
					attentions, hidden_states = hidden_states
					all_attentions.append(attentions)
				if output_all_encoded_layers:
					all_encoder_layers.append(hidden_states)
		if not output_all_encoded_layers:
			all_encoder_layers.append(hidden_states)
		if self.output_attentions:
			return all_attentions, all_encoder_layers
		return all_encoder_layers


class BertLayer(nn.Module):
	def __init__(self, config, output_attentions=False, keep_multihead_output=False):
		super(BertLayer, self).__init__()
		self.output_attentions = output_attentions
		self.attention = BertAttention(
			config, output_attentions=output_attentions, keep_multihead_output=keep_multihead_output
		)
		self.intermediate = BertIntermediate(config)
		self.output = BertOutput(config)

	def forward(self, hidden_states, attention_mask, head_mask=None):
		attention_output = self.attention(hidden_states, attention_mask, head_mask)
		if self.output_attentions:
			attentions, attention_output = attention_output
		intermediate_output = self.intermediate(attention_output)
		layer_output = self.output(intermediate_output, attention_output)
		if self.output_attentions:
			return attentions, layer_output
		return layer_output


class BertAttention(nn.Module):
	def __init__(self, config, output_attentions=False, keep_multihead_output=False):
		super(BertAttention, self).__init__()
		self.output_attentions = output_attentions
		self.flatten_image = not config.use_gaussian_attention and not config.use_learned_2d_encoding
		self.use_gaussian_attention = config.use_gaussian_attention
		self.config = config

		assert not config.use_gaussian_attention or not config.use_learned_2d_encoding  # TODO change to enum args

		if config.use_gaussian_attention:
			attention_cls = GaussianSelfAttention
		elif config.use_learned_2d_encoding:
			attention_cls = Learned2DRelativeSelfAttention
		else:
			attention_cls = BertSelfAttention

		self.self = attention_cls(config, output_attentions=output_attentions, keep_multihead_output=keep_multihead_output)

		self.output = BertSelfOutput(config)

	def prune_heads(self, heads):
		mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
		for head in heads:
			mask[head] = 0
		mask = mask.view(-1).contiguous().eq(1)
		index = torch.arange(len(mask))[mask].long()

		# Prune linear layers
		if not self.use_gaussian_attention:
			self.self.query = prune_linear_layer(self.self.query, index)
			self.self.key = prune_linear_layer(self.self.key, index)
			self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

		if self.use_gaussian_attention:
			device = self.self.attention_spreads.data.device
			keep_heads = torch.ones(self.self.num_attention_heads, device=device, dtype=torch.bool)
			for head in heads:
				keep_heads[head] = 0
			self.self.attention_spreads.data = self.self.attention_spreads.data[keep_heads].contiguous()
			self.self.attention_centers.data = self.self.attention_centers.data[keep_heads].contiguous()

		dim = 0 if not self.use_gaussian_attention else 1
		self.self.value = prune_linear_layer(self.self.value, index, dim=dim)
		# Update hyper params
		self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
		self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads

	def reset_heads(self, heads):
		"""Only for Gaussian Attention"""
		assert self.use_gaussian_attention
		self.self.reset_heads(heads)

	def forward(self, input_tensor, attention_mask, head_mask=None):
		is_image = len(input_tensor.shape) == 4
		if is_image and self.flatten_image:
			batch, width, height, d = input_tensor.shape
			input_tensor = input_tensor.view([batch, -1, d])

		self_output = self.self(input_tensor, attention_mask, head_mask)
		if self.output_attentions:
			attentions, self_output = self_output
		attention_output = self.output(self_output, input_tensor)

		if is_image and self.flatten_image:
			attention_output = attention_output.view([batch, width, height, -1])

		if self.output_attentions:
			return attentions, attention_output
		return attention_output


def gaussian_kernel_2d(mean, std_inv, size):
	"""Create a 2D gaussian kernel

	Args:
		mean: center of the gaussian filter (shift from origin)
			(2, ) vector
		std_inv: standard deviation $Sigma^{-1/2}$
			can be a single number, a vector of dimension 2, or a 2x2 matrix
		size: size of the kernel
			pair of integer for width and height
			or single number will be used for both width and height

	Returns:
		A gaussian kernel of shape size.
	"""
	if type(mean) is torch.Tensor:
		device = mean.device
	elif type(std_inv) is torch.Tensor:
		device = std_inv.device
	else:
		device = "cpu"

	# repeat the size for width, height if single number
	if isinstance(size, numbers.Number):
		width = height = size
	else:
		width, height = size

	# expand std to (2, 2) matrix
	if isinstance(std_inv, numbers.Number):
		std_inv = torch.tensor([[std_inv, 0], [0, std_inv]], device=device)
	elif std_inv.dim() == 0:
		std_inv = torch.diag(std_inv.repeat(2))
	elif std_inv.dim() == 1:
		assert len(std_inv) == 2
		std_inv = torch.diag(std_inv)

	# Enforce PSD of covariance matrix
	covariance_inv = std_inv.transpose(0, 1) @ std_inv
	covariance_inv = covariance_inv.float()

	# make a grid (width, height, 2)
	X = torch.cat(
		[
			t.unsqueeze(-1)
			for t in reversed(
				torch.meshgrid(
					[torch.arange(s, device=device) for s in [width, height]]
				)
			)
		],
		dim=-1,
	)
	X = X.float()

	# center the gaussian in (0, 0) and then shift to mean
	X -= torch.tensor([(width - 1) / 2, (height - 1) / 2], device=device).float()
	X -= mean.float()

	# does not use the normalize constant of gaussian distribution
	Y = torch.exp((-1 / 2) * torch.einsum("xyi,ij,xyj->xy", [X, covariance_inv, X]))

	# normalize
	# TODO could compute the correct normalization (1/2pi det ...)
	# and send warning if there is a significant diff
	# -> part of the gaussian is outside the kernel
	Z = Y / Y.sum()
	return Z


class GaussianSelfAttention(nn.Module):
	def __init__(self, config, output_attentions=False, keep_multihead_output=False):
		super().__init__()
		self.attention_gaussian_blur_trick = config.attention_gaussian_blur_trick
		self.attention_isotropic_gaussian = config.attention_isotropic_gaussian
		self.gaussian_init_mu_std = config.gaussian_init_mu_std
		self.gaussian_init_sigma_std = config.gaussian_init_sigma_std
		self.config = config

		self.num_attention_heads = config.num_attention_heads
		self.attention_head_size = config.hidden_size
		# assert config.hidden_size % config.num_attention_heads == 0, "num_attention_heads should divide hidden_size"
		# self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
		self.all_head_size = self.num_attention_heads * config.hidden_size
		self.output_attentions = output_attentions

		# CAREFUL: if change something here, change also in reset_heads (TODO remove code duplication)
		# shift of the each gaussian per head
		self.attention_centers = nn.Parameter(
			torch.zeros(self.num_attention_heads, 2).normal_(0.0, config.gaussian_init_mu_std)
		)

		if config.attention_isotropic_gaussian:
			# only one scalar (inverse standard deviation)
			# initialized to 1 + noise
			attention_spreads = 1 + torch.zeros(self.num_attention_heads).normal_(0, config.gaussian_init_sigma_std)
		else:
			# Inverse standart deviation $Sigma^{-1/2}$
			# 2x2 matrix or a scalar per head
			# initialized to noisy identity matrix
			attention_spreads = torch.eye(2).unsqueeze(0).repeat(self.num_attention_heads, 1, 1)
			attention_spreads += torch.zeros_like(attention_spreads).normal_(0, config.gaussian_init_sigma_std)

		self.attention_spreads = nn.Parameter(attention_spreads)

		self.value = nn.Linear(self.all_head_size, config.hidden_size)

		if not config.attention_gaussian_blur_trick:
			# relative encoding grid (delta_x, delta_y, delta_x**2, delta_y**2, delta_x * delta_y)
			MAX_WIDTH_HEIGHT = 50
			range_ = torch.arange(MAX_WIDTH_HEIGHT)
			grid = torch.cat([t.unsqueeze(-1) for t in torch.meshgrid([range_, range_])], dim=-1)
			relative_indices = grid.unsqueeze(0).unsqueeze(0) - grid.unsqueeze(-2).unsqueeze(-2)
			R = torch.cat([relative_indices, relative_indices ** 2, (relative_indices[..., 0] * relative_indices[..., 1]).unsqueeze(-1)], dim=-1)
			R = R.float()
			self.register_buffer("R", R)
			self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

	def get_heads_target_vectors(self):
		if self.attention_isotropic_gaussian:
			a = c = self.attention_spreads ** 2
			b = torch.zeros_like(self.attention_spreads)
		else:
			# $\Sigma^{-1}$
			inv_covariance = torch.einsum('hij,hkj->hik', [self.attention_spreads, self.attention_spreads])
			a, b, c = inv_covariance[:, 0, 0], inv_covariance[:, 0, 1], inv_covariance[:, 1, 1]

		mu_1, mu_2 = self.attention_centers[:, 0], self.attention_centers[:, 1]

		t_h = -1/2 * torch.stack([
			-2*(a*mu_1 + b*mu_2),
			-2*(c*mu_2 + b*mu_1),
			a,
			c,
			2 * b
		], dim=-1)
		return t_h

	def get_attention_probs(self, width, height):
		"""Compute the positional attention for an image of size width x height
		Returns: tensor of attention probabilities (width, height, num_head, width, height)
		"""
		u = self.get_heads_target_vectors()

		# Compute attention map for each head
		attention_scores = torch.einsum('ijkld,hd->ijhkl', [self.R[:width,:height,:width,:height,:], u])
		# Softmax
		attention_probs = torch.nn.Softmax(dim=-1)(attention_scores.view(width, height, self.num_attention_heads, -1))
		attention_probs = attention_probs.view(width, height, self.num_attention_heads, width, height)

		return attention_probs

	def reset_heads(self, heads):
		device = self.attention_spreads.data.device
		reset_heads_mask = torch.zeros(self.num_attention_heads, device=device, dtype=torch.bool)
		for head in heads:
			reset_heads_mask[head] = 1

		# Reinitialize mu and sigma of these heads
		self.attention_centers.data[reset_heads_mask].zero_().normal_(0.0, self.gaussian_init_mu_std)

		if self.attention_isotropic_gaussian:
			self.attention_spreads.ones_().normal_(0, self.gaussian_init_sigma_std)
		else:
			self.attention_spreads.zero_().normal_(0, self.gaussian_init_sigma_std)
			self.attention_spreads[:, 0, 0] += 1
			self.attention_spreads[:, 1, 1] += 1

		# Reinitialize value matrix for these heads
		mask = torch.zeros(self.num_attention_heads, self.attention_head_size, dtype=torch.bool)
		for head in heads:
			mask[head] = 1
		mask = mask.view(-1).contiguous()
		self.value.weight.data[:, mask].normal_(mean=0.0, std=self.config.initializer_range)
		# self.value.bias.data.zero_()


	def blured_attention(self, X):
		"""Compute the weighted average according to gaussian attention without
		computing explicitly the attention coefficients.

		Args:
			X (tensor): shape (batch, width, height, dim)
		Output:
			shape (batch, width, height, dim x num_heads)
		"""
		num_heads = self.attention_centers.shape[0]
		batch, width, height, d_total = X.shape
		Y = X.permute(0, 3, 1, 2).contiguous()

		kernels = []
		kernel_width = kernel_height = 7
		assert kernel_width % 2 == 1 and kernel_height % 2 == 1, 'kernel size should be odd'

		for mean, std_inv in zip(self.attention_centers, self.attention_spreads):
			conv_weights = gaussian_kernel_2d(mean, std_inv, size=(kernel_width, kernel_height))
			conv_weights = conv_weights.view(1, 1, kernel_width, kernel_height).repeat(d_total, 1, 1, 1)
			kernels.append(conv_weights)

		weights = torch.cat(kernels)

		padding_width = (kernel_width - 1) // 2
		padding_height = (kernel_height - 1) // 2
		out = F.conv2d(Y, weights, groups=d_total, padding=(padding_width, padding_height))

		# renormalize for padding
		all_one_input = torch.ones(1, d_total, width, height, device=X.device)
		normalizer = F.conv2d(all_one_input, weights,  groups=d_total, padding=(padding_width, padding_height))
		out /= normalizer

		return out.permute(0, 2, 3, 1).contiguous()

	def forward(self, hidden_states, attention_mask, head_mask=None):
		assert len(hidden_states.shape) == 4
		b, w, h, c = hidden_states.shape

		if not self.attention_gaussian_blur_trick:
			attention_probs = self.get_attention_probs(w, h)
			attention_probs = self.dropout(attention_probs)

			input_values = torch.einsum('ijhkl,bkld->bijhd', attention_probs, hidden_states)
			input_values = input_values.contiguous().view(b, w, h, -1)
		else:
			input_values = self.blured_attention(hidden_states)

		output_value = self.value(input_values)

		if self.output_attentions:
			return output_value, attention_probs
		else:
			return output_value


class BertSelfAttention(nn.Module):
	def __init__(self, config, output_attentions=False, keep_multihead_output=False):
		super(BertSelfAttention, self).__init__()
		if config.hidden_size % config.num_attention_heads != 0:
			raise ValueError(
				"The hidden size (%d) is not a multiple of the number of attention "
				"heads (%d)" % (config.hidden_size, config.num_attention_heads)
			)
		self.output_attentions = output_attentions
		self.keep_multihead_output = keep_multihead_output
		self.multihead_output = None

		self.num_attention_heads = config.num_attention_heads
		self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size

		self.query = nn.Linear(config.hidden_size, self.all_head_size)
		self.key = nn.Linear(config.hidden_size, self.all_head_size)
		self.value = nn.Linear(config.hidden_size, self.all_head_size)

		self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = x.view(*new_x_shape)
		return x.permute(0, 2, 1, 3)

	def forward(self, hidden_states, attention_mask, head_mask=None):
		mixed_query_layer = self.query(hidden_states)
		mixed_key_layer = self.key(hidden_states)
		mixed_value_layer = self.value(hidden_states)

		query_layer = self.transpose_for_scores(mixed_query_layer)
		key_layer = self.transpose_for_scores(mixed_key_layer)
		value_layer = self.transpose_for_scores(mixed_value_layer)

		# Take the dot product between "query" and "key" to get the raw attention scores.
		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
		attention_scores = attention_scores / math.sqrt(self.attention_head_size)
		# Apply the attention mask is (precomputed for all layers in BertModel forward() function)
		attention_scores = attention_scores + attention_mask

		# Normalize the attention scores to probabilities.
		attention_probs = nn.Softmax(dim=-1)(attention_scores)

		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		attention_probs = self.dropout(attention_probs)

		# Mask heads if we want to
		if head_mask is not None:
			attention_probs = attention_probs * head_mask

		context_layer = torch.matmul(attention_probs, value_layer)
		if self.keep_multihead_output:
			self.multihead_output = context_layer
			self.multihead_output.retain_grad()

		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
		context_layer = context_layer.view(*new_context_layer_shape)
		if self.output_attentions:
			return attention_probs, context_layer
		return context_layer


class BertIntermediate(nn.Module):
	def __init__(self, config):
		super(BertIntermediate, self).__init__()
		self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
		if isinstance(config.hidden_act, str) or (
			sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
		):
			self.intermediate_act_fn = ACT2FN[config.hidden_act]
		else:
			self.intermediate_act_fn = config.hidden_act

	def forward(self, hidden_states):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.intermediate_act_fn(hidden_states)
		return hidden_states


class BertLayerNorm(nn.Module):
	def __init__(self, hidden_size, eps=1e-12):
		"""Construct a layernorm module in the TF style (epsilon inside the square root).
		"""
		super(BertLayerNorm, self).__init__()
		self.weight = nn.Parameter(torch.ones(hidden_size))
		self.bias = nn.Parameter(torch.zeros(hidden_size))
		self.variance_epsilon = eps

	def forward(self, x):
		u = x.mean(-1, keepdim=True)
		s = (x - u).pow(2).mean(-1, keepdim=True)
		x =smtube (x - u) / torch.sqrt(s + self.variance_epsilon)
		return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
	"""Construct the embeddings from word, position and token_type embeddings.
	"""

	def __init__(self, config):
		super(BertEmbeddings, self).__init__()
		self.add_positional_encoding = config.add_positional_encoding_to_input
		self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
		if self.add_positional_encoding:
			self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
		self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

		# self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
		# any TensorFlow checkpoint file
		self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, input_ids, token_type_ids=None):
		seq_length = input_ids.size(1)
		position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
		position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
		if token_type_ids is None:
			token_type_ids = torch.zeros_like(input_ids)

		words_embeddings = self.word_embeddings(input_ids)
		token_type_embeddings = self.token_type_embeddings(token_type_ids)

		embeddings = words_embeddings + token_type_embeddings

		if self.add_positional_encoding:
			position_embeddings = self.position_embeddings(position_ids)
			embeddings += position_embeddings

		embeddings = self.LayerNorm(embeddings)
		embeddings = self.dropout(embeddings)
		return embeddings