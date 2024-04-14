import pandas as pd
import torch
import torch.utils.data as data
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split


class DLRMDataSet(data.Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = pd.read_csv(data_path)
        self.features = self.data.iloc[:, :-1]
        self.labels = self.data.iloc[:, -1]
        self.sparse_columns = ['C' + str(i) for i in range(1, 27)]
        self.dense_columns = ['I' + str(i) for i in range(1, 14)]
        self.sparse_features = self.features[self.sparse_columns]
        self.dense_features = self.features[self.dense_columns]
        self.embedding_cardinality = [self.features[column].nunique() for column in self.sparse_columns]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dense_feature = torch.tensor(self.dense_features.iloc[idx].values, dtype=torch.float32)
        sparse_feature = torch.tensor(self.sparse_features.iloc[idx].values, dtype=torch.long)
        # Get label
        label = torch.tensor(self.labels.iloc[idx])

        return dense_feature, sparse_feature, label


class DLRMDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=32, hidden_dim=10, shuffle=True):
        """
        :param data_path: path of the raw data
        :param batch_size: batch_size
        :param hidden_dim: the hidden dimension of the embedding layer for sparse features
        :param shuffle: shuffle the data or not
        """
        super().__init__()
        self.data = DLRMDataSet(data_path)
        self.embedding_sizes = [(num_categories, hidden_dim) for num_categories in self.data.embedding_cardinality]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def setup(self, stage=None):
        train_size = int(0.6 * len(self.data))
        val_size = int(0.2 * len(self.data))
        test_size = len(self.data) - train_size - val_size
        self.train_data, self.val_data, self.test_data = random_split(self.data, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

