from unittest import TestCase
from src.wfm.weighted_embedding_bag import WeightedEmbeddingBag
import torch


class TestWeightedEmbeddingBag(TestCase):

    def test_init(self):
        emb_bag = WeightedEmbeddingBag(10, 3)
        self.assertTrue(emb_bag.weight.shape[0] == 10)
        self.assertTrue(emb_bag.weight.shape[1] == 3)

        emb_bag = WeightedEmbeddingBag(2, 2, sparse=True, norm_type=2)
        self.assertTrue(emb_bag.weight.shape[0] == 2)
        self.assertTrue(emb_bag.weight.shape[1] == 2)

        emb_bag = WeightedEmbeddingBag(15, 1, norm_type=3)
        self.assertTrue(emb_bag.weight.shape[0] == 15)
        self.assertTrue(emb_bag.weight.shape[1] == 1)

    def test_forward(self):
        def dimension_checks(indices, weights, offsets):
            self.assertEqual(len(indices.shape), 2)
            self.assertEqual(len(weights.shape), 2)
            self.assertEqual(len(offsets.shape), 2)
            self.assertEqual(indices.shape[0], weights.shape[0])
            self.assertEqual(indices.shape[0], offsets.shape[0])
            self.assertEqual(indices.shape[1], weights.shape[1])

        def equality_check(expected, actual):
            actual = torch.squeeze(actual)
            self.assertEqual(len(expected.shape), len(actual.shape))
            shape_equalities = torch.unique(torch.tensor(expected.shape) == torch.tensor(actual.shape))
            self.assertEqual(len(shape_equalities), 1)
            self.assertTrue(shape_equalities[0])

            # Below we use torch.isclose instead of torch.eq, to avoid float precision.
            value_equalities = torch.unique(torch.isclose(expected, actual))
            self.assertEqual(len(value_equalities), 1)
            self.assertTrue(value_equalities[0])

        def invalid_forward_parameter():
            with self.assertRaises(Exception):
                emb_bag = WeightedEmbeddingBag(10, 3, fake_parameter=3)
                indices = torch.IntTensor([[1]])
                weights = torch.Tensor([[1.]])
                offsets = torch.IntTensor([[0]])
                emb_bag(indices, offsets, weights)

        invalid_forward_parameter()

        def one_index_selection():
            """selects a single index in each batch with different weights"""
            emb_bag = WeightedEmbeddingBag(3, 4)

            # single batch test with weight = 1.0
            indices = torch.IntTensor([[1]])
            weights = torch.Tensor([[1.]])
            offsets = torch.IntTensor([[0]])
            dimension_checks(indices, weights, offsets)
            expected = emb_bag.weight[1]
            actual = emb_bag(indices, offsets, weights)
            equality_check(expected, actual)

            # single batch test with weight = 0.3
            indices = torch.IntTensor([[0]])
            weights = torch.Tensor([[0.3]])
            offsets = torch.IntTensor([[0]])
            dimension_checks(indices, weights, offsets)
            expected = 0.3 * emb_bag.weight[0]
            actual = emb_bag(indices, offsets, weights)
            equality_check(expected, actual)

            # multi batch test
            indices = torch.IntTensor([[1], [1]])
            weights = torch.Tensor([[0.5], [2.]])
            offsets = torch.IntTensor([[0], [0]])
            dimension_checks(indices, weights, offsets)
            expected = torch.stack((0.5 * emb_bag.weight[1], 2. * emb_bag.weight[1]))
            actual = emb_bag(indices, offsets, weights)
            equality_check(expected, actual)

        one_index_selection()

        def multiple_selections():
            """selects two or more different indices, in each batch with different weights and offsets. In some cases
            indices and offsets are repeated. """
            emb_bag = WeightedEmbeddingBag(10, 3)

            # single batch test with weight = 1.0
            indices = torch.IntTensor([[1, 5]])
            weights = torch.Tensor([[1., 1.]])
            offsets = torch.IntTensor([[1]])
            dimension_checks(indices, weights, offsets)
            expected = emb_bag.weight[1] + emb_bag.weight[5]
            actual = emb_bag(indices, offsets, weights)
            equality_check(expected, actual)

            # multi batch test:
            indices = torch.IntTensor([[1, 3, 4, 5, 5], [1, 2, 3, 3, 3]])
            weights = torch.Tensor([[0.5, 0.2, 0.3, 0.5, 0.5], [2., -2., 0.3, 0.3, 0.7]])
            offsets = torch.IntTensor([[0, 2, 4], [1, 2, 4]])
            dimension_checks(indices, weights, offsets)
            expected = torch.stack((
                torch.stack(
                    (0.5 * emb_bag.weight[1], 0.2 * emb_bag.weight[3] + 0.3 * emb_bag.weight[4], emb_bag.weight[5])),
                torch.stack(
                    (2. * emb_bag.weight[1] + (-2.) * emb_bag.weight[2], 0.3 * emb_bag.weight[3], emb_bag.weight[3]))
            ))
            actual = emb_bag(indices, offsets, weights)
            equality_check(expected, actual)
            # sanity check:
            equality_check(emb_bag(indices, offsets, weights), emb_bag.forward(indices, offsets, weights))

        multiple_selections()

        def github_example():
            # github example tested:
            emb_bag = WeightedEmbeddingBag(6, 4)
            print(emb_bag.weight)
            indices = torch.IntTensor([[0, 4, 2, 5], [1, 1, 4, 5], [1, 2, 3, 5]])
            weights = torch.Tensor([[0.2, 0.1, 0.9, 0.8], [0.5, 0.2, 0.3, 0.3], [2., -2., 0.3, 0.6]])
            offsets = torch.IntTensor([[0, 1, 3], [0, 2, 3], [1, 2, 3]])
            dimension_checks(indices, weights, offsets)
            expected = torch.stack((
                torch.stack((0.2 * emb_bag.weight[0], 0.1 * emb_bag.weight[4],
                             0.9 * emb_bag.weight[2] + 0.8 * emb_bag.weight[5])),
                torch.stack((0.5 * emb_bag.weight[1], 0.2 * emb_bag.weight[1] + 0.3 * emb_bag.weight[4],
                             0.3 * emb_bag.weight[5])),
                torch.stack((2. * emb_bag.weight[1] + (-2.) * emb_bag.weight[2], 0.3 * emb_bag.weight[3],
                             0.6 * emb_bag.weight[5]))
            ))
            actual = emb_bag(indices, offsets, weights)
            equality_check(expected, actual)

        github_example()
