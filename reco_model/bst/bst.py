import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import torch.optim as optim


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, hidden_dim):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, hidden_dim)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class BehaviorSequenceTransformer(nn.Module):
    def __init__(self, sparse_cardinality, sequence_length, mlp_dims, movie_hidden_dim_per_head=3, num_head=3,
                 dropout_rate=0.1):
        """
        the Behavior Sequence Transformer model
        :param sparse_cardinality: a dictionary contains the cardinality of each sparse feature
        :param sequence_length: the length of the sequence
        :param mlp_dims: the dimensions of the MLP layers, note that an initial layer and an output layer are added on
        top of the mlp_dims implicitly
        :param movie_hidden_dim_per_head: the dimension of the hidden layer for each head in the transformer layer, the
        input of transformer hidden dim is the sum of the embedding dim and the positional embedding dim, which equals
        to num_head * hidden_dim_per_head
        :param num_head: the number of heads in the transformer layer
        :param dropout_rate: dropout rate
        """
        super().__init__()
        self.sparse_cardinality = sparse_cardinality
        self.num_head = num_head
        # embedding layers
        self.user_embedding = nn.Embedding(sparse_cardinality['user_id'],
                                           int(math.sqrt(sparse_cardinality['user_id'])))
        self.movie_embedding = nn.Embedding(sparse_cardinality['movie_id'],
                                            self.num_head * movie_hidden_dim_per_head)
        self.sex_embedding = nn.Embedding(sparse_cardinality['sex'], 2)
        self.age_group_embedding = nn.Embedding(sparse_cardinality['age_group'],
                                                int(math.sqrt(sparse_cardinality['age_group'])))
        self.occupation_embedding = nn.Embedding(sparse_cardinality['occupation'],
                                                 int(math.sqrt(sparse_cardinality['occupation'])))
        self.rating_embedding = nn.Embedding(sparse_cardinality['rating'], 2)
        self.position_embedding = PositionalEmbedding(max_len=sequence_length,
                                                      hidden_dim=self.num_head * movie_hidden_dim_per_head)
        # the max_len is sequence len + 1 since we add the target movie below
        # Transformer layer
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.movie_embedding.embedding_dim +
                                                                    self.position_embedding.pe.embedding_dim,
                                                            nhead=self.num_head,
                                                            dropout=dropout_rate)

        # MLP layer
        mlp_layers = []
        initial_mlp_input_dim = self.user_embedding.embedding_dim + \
                                (self.movie_embedding.embedding_dim + self.position_embedding.pe.embedding_dim) * \
                                sequence_length + \
                                self.sex_embedding.embedding_dim + self.age_group_embedding.embedding_dim + \
                                self.occupation_embedding.embedding_dim
        mlp_layers.append(nn.Linear(initial_mlp_input_dim, mlp_dims[0]))
        mlp_layers.append(nn.ReLU())
        for i in range(1, len(mlp_dims)):
            mlp_layers.append(nn.Linear(mlp_dims[i - 1], mlp_dims[i]))
            mlp_layers.append(nn.BatchNorm1d(mlp_dims[i]))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout_rate))
        mlp_layers.append(nn.Linear(mlp_dims[-1], 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, user_id, movie_history, sex, age_group, occupation, target_movie_id):
        # sequence feature
        movie_history_embedding = self.movie_embedding(movie_history)
        target_movie_embedding = self.movie_embedding(target_movie_id).unsqueeze(1)
        transformer_input = torch.cat([movie_history_embedding, target_movie_embedding], dim=1)
        position_embedding = self.position_embedding(transformer_input)
        transformer_input = torch.cat((transformer_input, position_embedding), dim=-1)
        transformer_output = self.transformer_layer(transformer_input)
        transformer_output = torch.flatten(transformer_output, start_dim=1)
        # other features
        user_id_embedding = self.user_embedding(user_id)
        sex_embedding = self.sex_embedding(sex)
        age_group_embedding = self.age_group_embedding(age_group)
        occupation_embedding = self.occupation_embedding(occupation)
        user_features = torch.cat((user_id_embedding, sex_embedding, age_group_embedding, occupation_embedding), dim=1)
        # concatenate all features
        all_features = torch.concat((user_features, transformer_output), dim=1)
        output = self.mlp(all_features)

        return output


class BSTModule(pl.LightningModule):
    def __init__(self, sparse_cardinality, sequence_length, mlp_dims, movie_hidden_dim_per_head=3, num_head=3,
                 dropout_rate=0.1, lr=0.001):
        """
        the Behavior Sequence Transformer module
        :param sparse_cardinality: a dictionary contains the cardinality of each sparse feature
        :param sequence_length: the length of the sequence
        :param mlp_dims: the dimensions of the MLP layers, note that an initial layer and an output layer are added on
        top of the mlp_dims implicitly
        :param movie_hidden_dim_per_head: the dimension of the hidden layer for each head in the transformer layer, the
        input of transformer hidden dim is the sum of the embedding dim and the positional embedding dim, which equals
        to num_head * hidden_dim_per_head
        :param num_head: the number of heads in the transformer layer
        :param dropout_rate: dropout rate
        :param lr: the learning rate
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = BehaviorSequenceTransformer(sparse_cardinality, sequence_length, mlp_dims,
                                                 movie_hidden_dim_per_head, num_head, dropout_rate)
        self.lr = lr
        self.loss = nn.MSELoss()  # apply MSE loss to forecast the ratings for the targeted movie
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.val_loss = []
        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError()

    def forward(self, user_id, movie_history, target_movie_id, sex, age_group, occupation):
        return self.model(user_id, movie_history, target_movie_id, sex, age_group, occupation)

    def training_step(self, batch, batch_idx):
        user_id, movie_history_seq, sex, age_group, occupation, target_movie_id, labels = batch
        outputs = self(user_id, movie_history_seq, sex, age_group, occupation, target_movie_id).squeeze()
        loss = self.loss(outputs, labels.float())
        self.log('train_loss', loss)
        result ={'loss': loss, 'preds': outputs, 'targets': labels}
        self.training_step_outputs.append(result)
        return loss

    def validation_step(self, batch, batch_idx):
        user_id, movie_history_seq, sex, age_group, occupation, target_movie_id, labels = batch
        outputs = self(user_id, movie_history_seq, sex, age_group, occupation, target_movie_id).squeeze()
        loss = self.loss(outputs, labels.float())
        self.log('val_loss', loss)
        self.val_loss.append(loss)
        result ={'loss': loss, 'preds': outputs, 'targets': labels}
        self.validation_step_outputs.append(result)

    def test_step(self, batch, batch_idx):
        user_id, movie_history_seq, sex, age_group, occupation, target_movie_id, labels = batch
        outputs = self(user_id, movie_history_seq, sex, age_group, occupation, target_movie_id).squeeze()
        loss = self.loss(outputs, labels.float())
        self.log('test_loss', loss)
        result = {'loss': loss, 'preds': outputs, 'targets': labels}
        self.test_step_outputs.append(result)

    def on_train_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.training_step_outputs]).detach().cpu()
        targets = torch.cat([x['targets'] for x in self.training_step_outputs]).detach().cpu()
        # Calculate accuracy
        mae = self.mae(preds, targets)
        mse = self.mse(preds, targets)
        rmse = torch.sqrt(mse)
        self.log('train_mae', mae, prog_bar=True)
        self.log('train_rmse', rmse, prog_bar=True)
        self.training_step_outputs.clear()  # free memory

    def on_validation_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.validation_step_outputs]).detach().cpu()
        targets = torch.cat([x['targets'] for x in self.validation_step_outputs]).detach().cpu()
        mae = self.mae(preds, targets)
        mse = self.mse(preds, targets)
        rmse = torch.sqrt(mse)
        self.log('val_mae', mae, prog_bar=True)
        self.log('val_rmse', rmse, prog_bar=True)
        self.validation_step_outputs.clear()  # free memory

    def on_test_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.test_step_outputs]).detach().cpu()
        targets = torch.cat([x['targets'] for x in self.test_step_outputs]).detach().cpu()
        mae = self.mae(preds, targets)
        mse = self.mse(preds, targets)
        rmse = torch.sqrt(mse)
        self.log('test_mae', mae, prog_bar=True)
        self.log('test_rmse', rmse, prog_bar=True)
        self.test_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]






