import torch
import torchinfo
from torch import nn
from typing import Optional


class GLU(nn.Module):
	def __init__(
			self,
			input_dim: int
	) -> None:
		super(GLU, self).__init__()
		self.input_dim = input_dim
		self.fc1 = nn.Linear(input_dim, input_dim)
		self.fc2 = nn.Linear(input_dim, input_dim)
		self.sigmoid = nn.Sigmoid()

	def forward(
			self,
			x: torch.Tensor
	) -> torch.Tensor:
		return self.fc2(x) * self.sigmoid(self.fc1(x))


class GRN(nn.Module):
	def __init__(
			self,
			d_hidden: int,
			d_input: Optional[int] = None,
			d_output: Optional[int] = None,
			d_context: Optional[int] = None,
			dropout: float = 0.0,
	):
		super().__init__()

		d_input = d_input or d_hidden
		d_context = d_context or 0
		if d_output is None:
			d_output = d_input
			self.add_skip = False
		else:
			if d_output != d_input:
				self.add_skip = True
				self.skip_proj = nn.Linear(in_features=d_input, out_features=d_output)
			else:
				self.add_skip = False

		self.mlp = nn.Sequential(
			nn.Linear(in_features=d_input + d_context, out_features=d_hidden),
			nn.ELU(),
			nn.Linear(in_features=d_hidden, out_features=d_output),
			nn.Dropout(p=dropout),
			GLU(input_dim=d_output),
		)

		self.layer_norm = nn.LayerNorm(d_output)

	def forward(
			self, x: torch.Tensor, context: Optional[torch.Tensor] = None
	) -> torch.Tensor:
		if self.add_skip:
			skip = self.skip_proj(x)
		else:
			skip = x

		if context is not None:
			x = torch.cat((x, context), dim=-1)
		x = self.mlp(x)
		x = self.layer_norm(x + skip)
		return x


if __name__ == '__main__':
	x = torch.rand(3, 4, 5)
	grn = GRN(d_input=x.shape[-1], d_hidden=3)
	y = grn(x)
	z = 1
