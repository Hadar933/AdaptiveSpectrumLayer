from typing import Optional
import torch.nn as nn
import torch
from ML.Zoo.TFT.feature_embedding import PerFeatureEmbedding
from ML.Zoo.TFT.gated_residual_network import GRN


class VSN(nn.Module):
	def __init__(
			self,
			d_hidden: int,
			n_features: int,
			dropout: float = 0.0,
			add_context: bool = False
	):
		super().__init__()
		self.weight_network = GRN(
			d_input=d_hidden * n_features,
			d_hidden=d_hidden,
			d_output=n_features,
			d_context=d_hidden if add_context else None,
			dropout=dropout
		)

		self.variable_network = nn.ModuleList(
			[
				GRN(d_hidden=d_hidden, dropout=dropout)
				for _ in range(n_features)
			]
		)

	def forward(
			self, variables: list[torch.Tensor], context: Optional[torch.Tensor] = None
	) -> tuple[torch.Tensor, torch.Tensor]:
		flatten = torch.cat(variables, dim=-1)
		if context is not None:
			context = context.expand_as(variables[0])
		weight = self.weight_network(flatten, context)
		weight = torch.softmax(weight.unsqueeze(-2), dim=-1)

		var_encodings = [grn(var) for var, grn in zip(variables, self.variable_network)]
		var_encodings = torch.stack(var_encodings, dim=-1)

		var_encodings = torch.sum(var_encodings * weight, dim=-1)

		return var_encodings, weight


if __name__ == '__main__':
	x = torch.rand(3, 4, 5)
	n_features = x.shape[-1]
	d_hidden = 10
	fe = PerFeatureEmbedding(n_features=n_features, embedding_size=d_hidden)
	vsn = VSN(d_hidden=d_hidden, n_features=n_features)
	embedding_list = fe(x)
	y = vsn(embedding_list)
