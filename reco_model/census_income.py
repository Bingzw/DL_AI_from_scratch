import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import pandas as pd
from sklearn.metrics import roc_auc_score
import pandas as pd
import torch
import torch.utils.data as data
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, random_split


class CensusIncomeDataset(data.Dataset):
    def __init__(self, train_path, test_path):
        super().__init__()
        self.column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour',
                             'hs_college', 'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin',
                             'sex', 'union_member', 'unemp_reason', 'full_or_part_emp', 'capital_gains',
                             'capital_losses', 'stock_dividends', 'tax_filer_stat', 'region_prev_res', 'state_prev_res',
                             'det_hh_fam_stat', 'det_hh_summ', 'instance_weight', 'mig_chg_msa', 'mig_chg_reg',
                             'mig_move_reg', 'mig_same', 'mig_prev_sunbelt', 'num_emp', 'fam_under_18',
                             'country_father', 'country_mother', 'country_self', 'citizenship', 'own_or_self',
                             'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']
        self.label_columns = ['income_50k', 'marital_stat']
        self.sparse_columns = ['class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college',
                                    'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                                    'unemp_reason', 'full_or_part_emp', 'tax_filer_stat', 'region_prev_res',
                                    'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg',
                                    'mig_move_reg', 'mig_same', 'mig_prev_sunbelt', 'fam_under_18', 'country_father',
                                    'country_mother', 'country_self', 'citizenship', 'vet_question']
        self.dense_columns = list(set(self.column_names) - set(self.sparse_columns) - set(self.label_columns))
        self.df_train_raw = pd.read_csv(train_path, delimiter=',', header=None, index_col=None, names=self.column_names)
        self.df_test_raw = pd.read_csv(test_path, delimiter=',', header=None, index_col=None, names=self.column_names)
        self.df_all = pd.concat([self.df_train_raw, self.df_test_raw])  # concatenate the train and test data and
        # re-split later
        self.df_all.dropna(inplace=True)

        # process label columns
        self.df_all['income_50k'] = self.df_all['income_50k'].apply(lambda x: 1 if x == ' 50000+.' else 0)
        self.df_all['marital_stat'] = self.df_all['marital_stat'].apply(lambda x: 0 if x == ' Never married' else 1)

        # reformat the data types
        self.df_all[self.sparse_columns] = self.df_all[self.sparse_columns].astype(str)
        self.df_all[self.dense_columns] = self.df_all[self.dense_columns].astype(float)

        # convert the sparse features to category encoding
        for feat in self.sparse_columns:
            lbe = LabelEncoder()
            self.df_all[feat] = lbe.fit_transform(self.df_all[feat])

        # normalize the dense features
        mms = MinMaxScaler()
        self.df_all[self.dense_columns] = mms.fit_transform(self.df_all[self.dense_columns])

        # get the cardinality of the sparse features
        self.embedding_cardinality = [self.df_all[column].nunique() for column in self.sparse_columns]

        self.sparse_features = self.df_all[self.sparse_columns]
        self.dense_features = self.df_all[self.dense_columns]
        self.label_income = self.df_all['income_50k']
        self.label_marital = self.df_all['marital_stat']

    def __len__(self):
        return len(self.df_all)

    def __getitem__(self, idx):
        dense_feature = torch.tensor(self.dense_features.iloc[idx].values, dtype=torch.float32)
        sparse_feature = torch.tensor(self.sparse_features.iloc[idx].values, dtype=torch.long)
        # Get label
        label_income = torch.tensor(self.label_income.iloc[idx])
        label_marital = torch.tensor(self.label_marital.iloc[idx])
        labels = torch.stack([label_income, label_marital], dim=0)

        return dense_feature, sparse_feature, labels


class CensusDataModule(pl.LightningDataModule):
    def __init__(self, train_path, test_path, batch_size=32, shuffle=True):
        """
        :param data_path: path of the raw data
        :param batch_size: batch_size
        :param hidden_dim: the hidden dimension of the embedding layer for sparse features
        :param shuffle: shuffle the data or not
        """
        super().__init__()
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.dataset = CensusIncomeDataset(train_path, test_path)
        self.sparse_cardinality = self.dataset.embedding_cardinality
        self.batch_size = batch_size
        self.shuffle = shuffle

    def setup(self, stage=None):
        train_size = int(0.6 * len(self.dataset))
        val_size = int(0.2 * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        self.train_data, self.val_data, self.test_data = random_split(self.dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
