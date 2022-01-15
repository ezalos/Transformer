import torch
from torch import nn


class AttentionHead(nn.Module):
	def __init__(self, dim_in: int, dim_q: int, dim_k: int):
		super().__init__()
		self.q = nn.Linear(dim_in, dim_q)
		self.k = nn.Linear(dim_in, dim_k)
		self.v = nn.Linear(dim_in, dim_k)

	def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
		x = scaled_dot_product_attention(
				self.q(query), 
				self.k(key), 
				self.v(value))
		return x
