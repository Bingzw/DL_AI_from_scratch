import unittest
import torch
from reco_model.deepfm.dfmnet import DeepFMNet, DeepFMModule


class TestDeepFM(unittest.TestCase):
    def setUp(self):
        self.num_dense_features = 5
        self.embedding_sizes = [(20, 5), (20, 10)]
        self.mlp_dims = [32, 16]
        self.deepfm = DeepFMNet(self.num_dense_features, self.embedding_sizes, self.mlp_dims)
        self.deepfm_module = DeepFMModule(self.num_dense_features, self.embedding_sizes, self.mlp_dims)
        self.batch_size = 10
        self.dense_features = torch.randn(self.batch_size, self.num_dense_features)
        self.sparse_features = torch.randint(0, 20, (self.batch_size, len(self.embedding_sizes)))

    def test_forward(self):
        output = self.deepfm(self.dense_features, self.sparse_features)
        self.assertEqual(output.shape, (self.batch_size, 1))

    def test_module_forward(self):
        output = self.deepfm_module(self.dense_features, self.sparse_features)
        self.assertEqual(output.shape, (self.batch_size, 1))

    def test_module_training_step(self):
        batch = (self.dense_features, self.sparse_features, torch.randint(0, 2, (self.batch_size,)))
        optimizer_idx = 0
        loss = self.deepfm_module.training_step(batch, optimizer_idx)
        self.assertTrue(torch.is_tensor(loss) and len(loss.shape) == 0)  # Loss should be a scalar tensor


if __name__ == '__main__':
    unittest.main()