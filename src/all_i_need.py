import torch
from torch import Tensor, nn
import torch.nn.functional as f


def scaled_dot_product_attention(
		query: Tensor,
		key: Tensor, 
		value: Tensor
	) -> Tensor:
	temp = query.bmm(key.transpose(1, 2))
	scale = query.size(-1) ** 0.5
	softmax = f.softmax(temp / scale, dim=-1)
	
	return softmax.bmm(value)



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

# Notice that MultiHeadAttention has no trainable components that operate  
# over the sequence dimension (axis 1). 
# Everything operates over the feature dimension (axis 2), 
# and so it is independent of sequence length. 

# We have to provide positional information to the model, 
# so that it knows about the relative position of data points 
# in the input sequences.


def position_encoding(
		seq_len: int, 
		dim_model: int, 
		device: torch.device = torch.device("cpu"),
) -> Tensor:
	pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
	dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
	phase = pos / (1e4 ** (dim // dim_model))

	return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))



def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
	return nn.Sequential(
		nn.Linear(dim_input, dim_feedforward),
		nn.ReLU(),
		nn.Linear(dim_feedforward, dim_input),
	)

class Residual(nn.Module):
	def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
		super().__init__()
		self.sublayer = sublayer
		self.norm = nn.LayerNorm(dimension)
		self.dropout = nn.Dropout(dropout)

	def forward(self, *tensors: Tensor) -> Tensor:
		# Assume that the "query" tensor is given first, so we can compute the
		# residual.  This matches the signature of 'MultiHeadAttention'.
		return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))

class TransformerEncoderLayer(nn.Module):
	def __init__(
		self,
		dim_model: int = 512,
		num_heads: int = 6,
		dim_feedforward: int = 2048,
		dropout: float = 0.1,
	):
		super().__init__()
		dim_q = dim_k = max(dim_model // num_heads, 1)
		self.attention = Residual(
			MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
			dimension=dim_model,
			dropout=dropout,
		)
		self.feed_forward = Residual(
			feed_forward(dim_model, dim_feedforward),
			dimension=dim_model,
			dropout=dropout,
		)

	def forward(self, src: Tensor) -> Tensor:
		src = self.attention(src, src, src)
		return self.feed_forward(src)


class TransformerEncoder(nn.Module):
	def __init__(
		self,
		num_layers: int = 6,
		dim_model: int = 512,
		num_heads: int = 8,
		dim_feedforward: int = 2048,
		dropout: float = 0.1,
	):
		super().__init__()
		self.layers = nn.ModuleList(
			[
				TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
				for _ in range(num_layers)
			]
		)

	def forward(self, src: Tensor) -> Tensor:
		seq_len, dimension = src.size(1), src.size(2)
		src += position_encoding(seq_len, dimension)
		for layer in self.layers:
			src = layer(src)

		return src


class TransformerDecoderLayer(nn.Module):
	def __init__(
		self,
		dim_model: int = 512,
		num_heads: int = 6,
		dim_feedforward: int = 2048,
		dropout: float = 0.1,
	):
		super().__init__()
		dim_q = dim_k = max(dim_model // num_heads, 1)
		self.attention_1 = Residual(
			MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
			dimension=dim_model,
			dropout=dropout,
		)
		self.attention_2 = Residual(
			MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
			dimension=dim_model,
			dropout=dropout,
		)
		self.feed_forward = Residual(
			feed_forward(dim_model, dim_feedforward),
			dimension=dim_model,
			dropout=dropout,
		)

	def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
		tgt = self.attention_1(tgt, tgt, tgt)
		tgt = self.attention_2(tgt, memory, memory)
		return self.feed_forward(tgt)


class TransformerDecoder(nn.Module):
	def __init__(
		self,
		num_layers: int = 6,
		dim_model: int = 512,
		num_heads: int = 8,
		dim_feedforward: int = 2048,
		dropout: float = 0.1,
	):
		super().__init__()
		self.layers = nn.ModuleList(
			[
				TransformerDecoderLayer(dim_model, num_heads, dim_feedforward, dropout)
				for _ in range(num_layers)
			]
		)
		self.linear = nn.Linear(dim_model, dim_model)

	def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
		seq_len, dimension = tgt.size(1), tgt.size(2)
		tgt += position_encoding(seq_len, dimension)
		for layer in self.layers:
			tgt = layer(tgt, memory)

		return torch.softmax(self.linear(tgt), dim=-1)


class Transformer(nn.Module):
	def __init__(
		self, 
		num_encoder_layers: int = 6,
		num_decoder_layers: int = 6,
		dim_model: int = 512, 
		num_heads: int = 6, 
		dim_feedforward: int = 2048, 
		dropout: float = 0.1, 
		activation: nn.Module = nn.ReLU(),
	):
		super().__init__()
		self.encoder = TransformerEncoder(
			num_layers=num_encoder_layers,
			dim_model=dim_model,
			num_heads=num_heads,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
		)
		self.decoder = TransformerDecoder(
			num_layers=num_decoder_layers,
			dim_model=dim_model,
			num_heads=num_heads,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
		)

	def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
		return self.decoder(tgt, self.encoder(src))

if __name__ == "__main__":
	src = torch.rand(64, 32, 512)
	tgt = torch.rand(64, 16, 512)
	out = Transformer()(src, tgt)
	print(out.shape)
	# torch.Size([64, 16, 512])