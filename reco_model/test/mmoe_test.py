import unittest
import torch
from reco_model.multitask.mmoe import MMOENet, MMOEModule


class TestMMOENet(unittest.TestCase):
    def setUp(self):
        self.num_dense_features = 10
        self.sparse_cardinality = [20, 20]
        self.hidden_dim = 32
        self.bottom_mlp_dims = [64, 32]
        self.tower_mlp_dims = [32, 16]
        self.num_tasks = 2
        self.num_experts = 3
        self.dropout = 0.1
        self.mmoe = MMOENet(self.num_dense_features, self.sparse_cardinality, self.hidden_dim,
                             self.bottom_mlp_dims, self.tower_mlp_dims, self.num_tasks,
                             self.num_experts, self.dropout)
        self.mmoe_module = MMOEModule(self.num_dense_features, self.sparse_cardinality, self.hidden_dim,
                                      self.bottom_mlp_dims, self.tower_mlp_dims, self.num_tasks,
                                      self.num_experts, self.dropout)

    def test_forward(self):
        dense_features = torch.randn(4, self.num_dense_features)
        sparse_features = torch.randint(high=max(self.sparse_cardinality), size=(4, len(self.sparse_cardinality)))
        output = self.mmoe(dense_features, sparse_features)
        self.assertEqual(len(output), self.num_tasks)
        for task_output in output:
            self.assertEqual(task_output.size(), torch.Size([4]))

    def test_module_forward(self):
        dense_features = torch.randn(4, self.num_dense_features)
        sparse_features = torch.randint(high=max(self.sparse_cardinality), size=(4, len(self.sparse_cardinality)))
        output = self.mmoe_module(dense_features, sparse_features)
        self.assertEqual(len(output), self.num_tasks)
        for task_output in output:
            self.assertEqual(task_output.size(), torch.Size([4]))

    def test_module_training_step(self):
        batch = (torch.randn(4, self.num_dense_features),
                 torch.randint(high=max(self.sparse_cardinality), size=(4, len(self.sparse_cardinality))),
                 torch.randint(0, self.num_tasks, (4,)))
        optimizer_idx = 0
        loss = self.mmoe_module.training_step(batch, optimizer_idx)
        self.assertTrue(torch.is_tensor(loss) and len(loss.shape) == 0)


if __name__ == '__main__':
    unittest.main()