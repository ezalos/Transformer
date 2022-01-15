# Transformers

Paper: [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

## Attention Formula

$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$



![https://miro.medium.com/max/1750/1*BzhKcJJxv974OxWOVqUuQQ.png](https://miro.medium.com/max/1750/1*BzhKcJJxv974OxWOVqUuQQ.png)

Q, K, and V are batches of matrices, with shape `(batch_size, seq_length, num_features)`


 - $Q$ Vector(Linear layer output) 
 	- related with what we encode
	- (output, it can be output of encoder layer or decoder layer)
 - $K$ Vector(Linear layer output)
 	- related with what we use as input to output.
 - $V$ Learned vector(Linear layer output) 
 	- as a result of calculations, related with input

 - The $ Attention $ is applied to the value $ V $ 

## Positional Encoding

MultiHeadAttention has no trainable components that operate over the sequence dimension (axis 1). Everything operates over the feature dimension (axis 2), and so it is independent of sequence length. We have to provide positional information to the model, so that it knows about the relative position of data points in the input sequences.

Vaswani et. al. encode positional information using trigonometric functions, according to the equation:

$$

PE_{(pos, 2i)} = \sin{\frac{pos}{10000^{{2i}/{d_{model}}}}}

\newline

PE_{(pos, 2i + 1)} = \cos{\frac{pos}{10000^{{2i}/{d_{model}}}}}

$$

## Transformer

![](https://miro.medium.com/max/771/1*j9MmpNZzbBqkWes0GN8IBQ.png)

