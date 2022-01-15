
# Notice that MultiHeadAttention has no trainable components that operate  
# over the sequence dimension (axis 1). 
# Everything operates over the feature dimension (axis 2), 
# and so it is independent of sequence length. 

# We have to provide positional information to the model, 
# so that it knows about the relative position of data points 
# in the input sequences.
# Vaswani et. al. encode positional information using trigonometric functions, according to the equation:

def position_encoding(
		seq_len: int, 
		dim_model: int, 
		device: torch.device = torch.device("cpu"),
) -> Tensor:
	pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
	dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
	phase = pos / (1e4 ** (dim // dim_model))

	return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))