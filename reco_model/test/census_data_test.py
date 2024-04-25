import unittest
import torch
from torch.utils.data import DataLoader
from reco_model.census_income import CensusIncomeDataset, CensusDataModule


class TestCensusIncomeDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = CensusIncomeDataset(train_path="../../data/reco_data/census-income.data.gz",
                                           test_path="../../data/reco_data/census-income.test.gz")

    def test_len(self):
        self.assertEqual(len(self.dataset), len(self.dataset.df_all))

    def test_getitem(self):
        dense_feature, sparse_feature, labels = self.dataset[0]
        self.assertIsInstance(dense_feature, torch.Tensor)
        self.assertIsInstance(sparse_feature, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)


class TestCensusDataModule(unittest.TestCase):
    def setUp(self):
        self.data_module = CensusDataModule(train_path="../../data/reco_data/census-income.data.gz",
                                            test_path="../../data/reco_data/census-income.test.gz",
                                            batch_size=10)

    def test_dataloaders(self):
        self.data_module.setup()
        self.assertIsInstance(self.data_module.train_dataloader(), DataLoader)
        self.assertIsInstance(self.data_module.val_dataloader(), DataLoader)
        self.assertIsInstance(self.data_module.test_dataloader(), DataLoader)

    def test_load_one_batch(self):
        self.data_module.setup()
        train_loader = self.data_module.train_dataloader()
        batch = next(iter(train_loader))
        self.assertEqual(len(batch), 3)  # dense_features, sparse_features, labels
        self.assertIsInstance(batch[0], torch.Tensor)  # dense_features
        self.assertIsInstance(batch[1], torch.Tensor)  # sparse_features
        self.assertIsInstance(batch[2], torch.Tensor)  # labels
        self.assertEqual(batch[2].shape[0], 10)
        self.assertEqual(batch[2].shape[1], 2)
        print(batch[2])


if __name__ == '__main__':
    unittest.main()