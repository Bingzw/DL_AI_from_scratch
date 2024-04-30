import numpy as np
import torch
import torch.utils.data as data
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, random_split
from reco_model.bst.util import rating_data_sequence_creation


class MovieRatingDataSet(data.Dataset):
    def __init__(self, user_path, movie_path, rating_path, sequence_length, step_size=1):
        super().__init__()
        self.ratings_seq_df, self.sparse_cardinality = rating_data_sequence_creation(user_path, movie_path, rating_path,
                                                                                     sequence_length, step_size)
        self.ratings_seq_df['user_id'] = self.ratings_seq_df['user_id'].astype(np.int64)
        self.sparse_columns = ['user_id', 'movie_id', 'sex', 'age_group', 'occupation', 'rating']

        for column in ['sex', 'age_group', 'occupation']:
            lbe = LabelEncoder()
            self.ratings_seq_df[column] = lbe.fit_transform(self.ratings_seq_df[column])

    def __len__(self):
        return len(self.ratings_seq_df)

    def __getitem__(self, idx):
        record = self.ratings_seq_df.iloc[idx]
        user_id = record['user_id']
        movie_history_seq = eval(record['sequence_movie_ids'])
        movie_history_ratings = eval(record['sequence_ratings'])
        target_movie_id = movie_history_seq[-1]
        target_movie_rating = movie_history_ratings[-1]

        movie_history_seq = torch.LongTensor(movie_history_seq[:-1])

        sex = record['sex']
        age_group = record['age_group']
        occupation = record['occupation']
        return user_id, movie_history_seq, sex, age_group, occupation, \
               target_movie_id, target_movie_rating


class MovieRatingSeqDataModule(pl.LightningDataModule):
    def __init__(self, user_path, movie_path, rating_path, sequence_length, step_size, batch_size=32, shuffle=True):
        """
        :param data_path: path of the raw data
        :param batch_size: batch_size
        :param hidden_dim: the hidden dimension of the embedding layer for sparse features
        :param shuffle: shuffle the data or not
        """
        super().__init__()
        self.dataset = MovieRatingDataSet(user_path, movie_path, rating_path, sequence_length, step_size)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def setup(self, stage=None):
        train_size = int(0.6 * len(self.dataset))
        val_size = int(0.2 * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        self.train_data, self.val_data, self.test_data = random_split(self.dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=self.shuffle, persistent_workers=True,
                          num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, persistent_workers=True,
                          num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, persistent_workers=True,
                          num_workers=2)