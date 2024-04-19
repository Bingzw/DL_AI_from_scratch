import unittest
import torch
from torchsummary import summary
from reco_model.deepfm.dfmnet import DeepFMNet


class TestDeepFMNet(unittest.TestCase):
    def setUp(self):
        self.num_dense_features = 5
        self.embedding_sizes = [(20, 5), (20, 10)]
        self.mlp_dims = [32, 16]
        self.deepfm = DeepFMNet(self.num_dense_features, self.embedding_sizes, self.mlp_dims)
        self.batch_sizes = [1, 2, 4, 8, 16]

    def test_forward(self):
        for batch_size in self.batch_sizes:
            dense_features = torch.randn(batch_size, self.num_dense_features)
            sparse_features = torch.randint(0, 20, (batch_size, len(self.embedding_sizes)))
            output = self.deepfm(dense_features, sparse_features)
            self.assertEqual(output.shape, (batch_size, 1))


if __name__ == '__main__':
    unittest.main()