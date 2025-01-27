from typing import List

import torch
import torch.nn as nn


class Mlp(nn.Module):
	def __init__(
			self,
			input_size: int,
			history_size: int,
			hidden_dims_list: list[int],
			output_size: int
	):
		"""
		an MLP implementation that performs (B,H,F) -> (B,H*F) -> MLP -> (B,Out_dim)
		:param input_size: the number of features in the input, required to calculate the MLP input dim
		:param history_size: feature window, essentially, required to calculate the MLP input dim
		:param hidden_dims_list: the dimensions of the hidden layers only (without input and output)
		:param output_size: output dim for multi-target prediction
		"""
		super(Mlp, self).__init__()
		self.input_size = input_size
		self.history_size = history_size
		self.output_size = output_size
		self.flatten = nn.Flatten()
		self.hidden_dims_list = hidden_dims_list
		self.input_layer = nn.Linear(input_size * history_size, hidden_dims_list[0])
		self.hidden_layers = nn.ModuleList()
		for i in range(len(hidden_dims_list) - 1):
			self.hidden_layers.append(nn.Linear(hidden_dims_list[i], hidden_dims_list[i + 1]))
			self.hidden_layers.append(nn.ReLU())
		self.output_layer = nn.Linear(hidden_dims_list[-1], output_size)

	def __str__(self):
		return f"MLP[{self.input_size},{self.history_size},{self.hidden_dims_list},{self.output_size}]"

	def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
		x = self.flatten(x)
		x = self.input_layer(x)
		for layer in self.hidden_layers:
			x = layer(x)
		x = self.output_layer(x)
		return x.unsqueeze(1)
