import torch
from torch import nn

class MultiHeadAttention(nn.Module):
	def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
		super().__init__()

		# Each attention head computes its own :
		#	- query, 
		#	- key,
		#	- value arrays, 
		# and then applies scaled dot-product attention. 

		# Conceptually, this means each head can attend to a different part
		# of the input sequence, independent of the others. 

		# Increasing the number of attention heads allows us to “pay attention” 
		# to more parts of the sequence at once, which makes the model more powerful.

		self.heads = nn.ModuleList(
			[AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
		)
		self.linear = nn.Linear(num_heads * dim_k, dim_in)

	def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
		return self.linear(
			torch.cat([h(query, key, value) for h in self.heads], dim=-1)
		)