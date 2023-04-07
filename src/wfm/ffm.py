# Copyright 2023, Yahoo.

import torch
from .weighted_embedding_bag import WeightedEmbeddingBag
import math


class WeightedFFM(torch.nn.Module):
    r"""
    A field-aware factorization machine (see Juan et al. 2016) allowing weighted multi-value fields, based on the
    :class:`WeightedEmbeddingBag` module.

    :param num_features: The number of features of the factorization machine
    :type num_features: int

    :param field_dim: The dimension of the embedding vector's field overlap block. The embedding
    vectors will be initialized to a uniformly-distributed vector in range [0, 1/sqrt(field_dim)].
    :type field_dim: int

    :param num_fields: The number of fields.
    :type num_fields: int

    :param emb_kwargs: additional args passed to the :class:`WeightedEmbeddingBag` objects.
    """
    def __init__(self, num_features: int, field_dim: int, num_fields: int, **emb_kwargs):
        super().__init__()
        self.field_dim = field_dim
        self.num_fields = num_fields
        i_indices, j_indices = torch.tril_indices(num_fields, num_fields, -1)
        self.register_buffer('i_indices', i_indices)
        self.register_buffer('j_indices', j_indices)
        self.vectors = WeightedEmbeddingBag(num_features, field_dim * num_fields,
                                            **emb_kwargs)
        self.biases = WeightedEmbeddingBag(num_features, 1,
                                           **emb_kwargs)
        self.bias = torch.nn.Parameter(torch.tensor(0.))

        vec_init_scale = 1. / math.sqrt(field_dim)
        with torch.no_grad():
            torch.nn.init.uniform_(self.vectors.weight, 0, vec_init_scale)
            torch.nn.init.zeros_(self.biases.weight)

    def _fast_ffm_pairwise(self, batch_size, vectors, fields):
        fields_i = fields[:, self.i_indices]
        fields_j = fields[:, self.j_indices]

        batches = torch.arange(batch_size, device=vectors.device)
        vectors_i = vectors[batches[:, None], fields_i, fields_j]
        vectors_j = vectors[batches[:, None], fields_j, fields_i]

        pairwise = (vectors_i * vectors_j).sum(dim=[-1, -2])
        return pairwise

    def forward(self,
                indices: torch.Tensor,
                weights: torch.Tensor,
                offsets: torch.Tensor,
                fields: torch.Tensor):
        r"""
        Returns FFM scores corresponding to a mini-batch of weighted sums of embedding bags. The scores
        are computed according to the full FFM score function, including the linear and bias terms. The indices,
        weights, and offsets parameters are as in the :class:`WeightedEmbeddingBag` class.

        :param indices: BxN matrix of embedding indices in the mini-batch
        :type indices: torch.Tensor

        :param weights: BxN matrix of corresponding embedding weights
        :type weights: torch.Tensor

        :param offsets: BxM matrix of bag end-point offsets
        :type offsets: torch.Tensor

        :param fields: BxM matrix of the field each bag corresponds to.
        :type fields: torch.Tensor
        """
        vectors = self.vectors(indices, weights, offsets)
        biases = self.biases(indices, weights, offsets)

        batch_size = vectors.shape[0]
        vectors = vectors.view(batch_size, self.num_fields, self.num_fields, self.field_dim)
        pairwise = self._fast_ffm_pairwise(batch_size, vectors, fields)
        linear = biases.squeeze().sum(dim=1)
        return pairwise + linear + self.bias
