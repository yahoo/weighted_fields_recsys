import torch
from weighted_embedding_bag import WeightedEmbeddingBag
import math


class WeightedFM(torch.nn.Module):
    r"""
    A factorization machine (see Rendle et al. 2013) allowing weighted multi-value fields, based on the
    :class:`WeightedEmbeddingBag` module.

    :param num_features: The number of features of the factorization machine
    :type num_features: int

    :param embedding_dim: The dimension of each embedding vector. Each embedding vector is initialized
    to a uniformly-distributed vector in range [0, 1/sqrt(embedding_dim)].
    :type embedding_dim: int

    :param emb_kwargs: additional args passed to the :class:`WeightedEmbeddingBag` objects.
    """
    def __init__(self, num_features: int, embedding_dim: int, **emb_kwargs):
        super().__init__()
        self.vectors = WeightedEmbeddingBag(num_features, embedding_dim, **emb_kwargs)
        self.biases = WeightedEmbeddingBag(num_features, 1, **emb_kwargs)
        self.bias = torch.nn.Parameter(torch.tensor(0.))

        vec_init_scale = 1. / math.sqrt(embedding_dim)
        with torch.no_grad():
            torch.nn.init.uniform_(self.vectors.emb, 0, vec_init_scale)
            torch.nn.init.zeros_(self.biases.emb)

    def forward(self, indices: torch.Tensor, weights: torch.Tensor, offsets: torch.Tensor):
        r"""

        :param indices:
        :param weights:
        :param offsets:
        :return:
        """
        vectors = self.vectors(indices, weights, offsets)
        biases = self.biases(indices, weights, offsets).squeeze()

        square_of_sum = vectors.sum(dim=1).square()
        sum_of_square = vectors.square().sum(dim=1)
        pairwise = 0.5 * (square_of_sum - sum_of_square).sum(dim=1)
        linear = biases.squeeze().sum(dim=1)

        return pairwise + linear + self.bias
