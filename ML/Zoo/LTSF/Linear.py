import torch
import torch.nn as nn
from typing import Optional

"""
Adaptation from LTSFLinear git repo
"""


class Linear(nn.Module):
	"""
	Just one Linear layer
	"""

	def __init__(
			self,
			feature_lags: int,
			target_lags: int,
			n_features: int,
			individual: bool,
			output_size: Optional[int] = None
	):
		super(Linear, self).__init__()
		self.feature_lags = feature_lags
		self.target_lags = target_lags
		self.n_features = n_features
		self.individual = individual
		self.use_matching_layer = output_size is not None and output_size != n_features
		if self.use_matching_layer:
			self.output_dim_matching_layer = nn.Linear(n_features, output_size)
		if self.individual:
			self.Linear = nn.ModuleList()
			for i in range(self.n_features):
				self.Linear.append(nn.Linear(self.feature_lags, self.target_lags))
		else:
			self.Linear = nn.Linear(self.feature_lags, self.target_lags)

	def __str__(self):
		return f"LTSFLinear[{self.feature_lags},{self.target_lags},{self.n_features},{self.individual}]"

	def forward(self, x):
		# x: [Batch, Input length, Channel]
		if self.individual:
			output = torch.zeros([x.size(0), self.target_lags, x.size(2)], dtype=x.dtype).to(x.device)
			for i in range(self.n_features):
				output[:, :, i] = self.Linear[i](x[:, :, i])
			x = output
		else:
			x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
		if self.use_matching_layer:
			x = self.output_dim_matching_layer(x)
		return x  # [Batch, Output length, output_size]
