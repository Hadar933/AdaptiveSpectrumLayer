import torch
from torch import nn


class PerFeatureEmbedding(nn.Module):

	def __init__(
			self,
			n_features: int,
			embedding_size: int
	) -> None:
		"""
		embeds each feature in the input using a separate linear layer and without mixing
		:param n_features: number of features in the input
		:param embedding_size: the output size we embed to
		"""
		super(PerFeatureEmbedding, self).__init__()
		self.n_features = n_features
		self.embedding_size = embedding_size
		self.per_feature_layers = nn.ModuleList([nn.Linear(1, embedding_size) for _ in range(n_features)])

	def forward(
			self,
			x: torch.Tensor
	) -> list[torch.Tensor]:
		return [self.per_feature_layers[i](x[:, :, [i]]) for i in range(self.n_features)]


if __name__ == '__main__':
	x = torch.randn((3, 4, 5))
	fe = PerFeatureEmbedding(x.shape[-1], 10)
	y = fe(x)

