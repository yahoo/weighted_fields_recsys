# Weighted field embeddings for recommender systems
PyTorch embedding bag module and factorization machine models for multi-value fields with weights per value. For example, imagine a data-set of movies where the "genres" column may contain a list of genres with corresponding weights representing a measure of confidence in the movie belonging to the genre.

The basic component of this package is the `WeightedEmbeddingBag` class, which is similar to the PyTorch `torch.nn.EmbeddingBag` class, but supports weights and bag aggregation in mini-batches. It receives three parameters: indices, offsets, and weights, depicted below.
![WeightedEmbeddingBag](doc/weighted_embedding_bag.png)

The indices array selects embeddings, the weights array is used to multiply embeddings by a corresponding weight, and the offsets array defines the endpoints of each bag. In the example above, we select the embedding vectors $(v_0, v_4, v_2, v_5)$. They will be multiplied by the weights $(0.2, 0.1, 0.9, 0.8)$ and grouped into three bags, the first includes only $v_0$, the second only $v_4$, and the third includes $v_2$ and $v_5$. Consequently, the output will be
$$
0.2 \cdot v_0, 0.1 \cdot v_4, 0.9 \cdot v_2 + 0.8 \cdot v_5
$$

Since offsets point to the _end-point_ of each bag, we can use mini-batches with a variable number of embeddings, and make sure that the offsets array specifies the last bag to end before the padding of each sample.

On top, we have implemented the classical Factorization Machine model in the `WeightedFM` class, and a fully-vectorized version of a field-aware factorization machine in the `WeightedFFM` class.