import unittest
import torch
from reco_model.bst.bst import BSTModule, BehaviorSequenceTransformer, PositionalEmbedding


class TestBST(unittest.TestCase):
    def setUp(self):
        self.sparse_cardinality = {
            'movie_id': 81,
            'user_id': 100,
            'sex': 2,
            'age_group': 10,
            'occupation': 10,
            'rating': 5
        }
        self.batch_size = 5
        self.sequence_length = 8
        self.mlp_dims = [16, 8]
        self.dropout_rate = 0.1
        self.bst = BehaviorSequenceTransformer(
            sparse_cardinality=self.sparse_cardinality,
            sequence_length=self.sequence_length,
            mlp_dims=self.mlp_dims,
            dropout_rate=self.dropout_rate
        )
        self.bst_module = BSTModule(
            sparse_cardinality=self.sparse_cardinality,
            sequence_length=self.sequence_length,
            mlp_dims=self.mlp_dims,
            dropout_rate=self.dropout_rate
        )
        self.pe_test = PositionalEmbedding(self.sequence_length, 32)

    def test_positional_embedding_forward(self):
        x = torch.rand(self.batch_size, self.sequence_length)
        output = self.pe_test(x)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (self.batch_size, self.sequence_length, 32))

    def test_forward(self):
        user_id = torch.randint(0, self.sparse_cardinality['user_id'], (self.batch_size,))
        movie_history = torch.randint(0, self.sparse_cardinality['movie_id'], (self.batch_size, self.sequence_length-1))
        target_movie_id = torch.randint(0, self.sparse_cardinality['movie_id'], (self.batch_size,))
        sex = torch.randint(0, self.sparse_cardinality['sex'], (self.batch_size,))
        age_group = torch.randint(0, self.sparse_cardinality['age_group'], (self.batch_size,))
        occupation = torch.randint(0, self.sparse_cardinality['occupation'], (self.batch_size,))

        output = self.bst(user_id, movie_history, sex, age_group, occupation, target_movie_id)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (self.batch_size, 1))

    def test_module_forward(self):
        user_id = torch.randint(0, self.sparse_cardinality['user_id'], (self.batch_size,))
        movie_history = torch.randint(0, self.sparse_cardinality['movie_id'], (self.batch_size, self.sequence_length-1))
        target_movie_id = torch.randint(0, self.sparse_cardinality['movie_id'], (self.batch_size,))
        sex = torch.randint(0, self.sparse_cardinality['sex'], (self.batch_size,))
        age_group = torch.randint(0, self.sparse_cardinality['age_group'], (self.batch_size,))
        occupation = torch.randint(0, self.sparse_cardinality['occupation'], (self.batch_size,))

        output = self.bst_module(user_id, movie_history, sex, age_group, occupation, target_movie_id)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (self.batch_size, 1))

    def test_module_training_step(self):
        user_id = torch.randint(0, self.sparse_cardinality['user_id'], (self.batch_size,))
        movie_history = torch.randint(0, self.sparse_cardinality['movie_id'], (self.batch_size, self.sequence_length-1))
        target_movie_id = torch.randint(0, self.sparse_cardinality['movie_id'], (self.batch_size,))
        sex = torch.randint(0, self.sparse_cardinality['sex'], (self.batch_size,))
        age_group = torch.randint(0, self.sparse_cardinality['age_group'], (self.batch_size,))
        occupation = torch.randint(0, self.sparse_cardinality['occupation'], (self.batch_size,))
        labels = torch.randint(1, 6, (self.batch_size,))

        batch = (user_id, movie_history, sex, age_group, occupation, target_movie_id, labels)
        optimizer_idx = 0
        loss = self.bst_module.training_step(batch, optimizer_idx)
        self.assertTrue(torch.is_tensor(loss) and len(loss.shape) == 0)  # Loss should be a scalar tensor


if __name__ == '__main__':
    unittest.main()