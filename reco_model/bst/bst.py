import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, hidden_dim):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, hidden_dim)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class BehaviorSequenceTransformer(nn.Module):
    def __init__(self, sparse_cardinality, sequence_length, mlp_dims, dropout_rate=0.1):
        super().__init__()
        self.sparse_cardinality = sparse_cardinality
        # embedding layers
        self.user_embedding = nn.Embedding(sparse_cardinality['user_id'],
                                           int(math.sqrt(sparse_cardinality['user_id'])))
        self.movie_embedding = nn.Embedding(sparse_cardinality['movie_id'],
                                            int(math.sqrt(sparse_cardinality['movie_id'])))
        self.sex_embedding = nn.Embedding(sparse_cardinality['sex'], 2)
        self.age_group_embedding = nn.Embedding(sparse_cardinality['age_group'],
                                                int(math.sqrt(sparse_cardinality['age_group'])))
        self.occupation_embedding = nn.Embedding(sparse_cardinality['occupation'],
                                                 int(math.sqrt(sparse_cardinality['occupation'])))
        self.rating_embedding = nn.Embedding(sparse_cardinality['rating'], 2)
        self.position_embedding = PositionalEmbedding(max_len=sequence_length, hidden_dim=math.sqrt(sequence_length))

        # Transformer layer
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.movie_embedding.embedding_dim, nhead=3,
                                                            dropout=dropout_rate)

        # MLP layer
        mlp_layers = []
        initial_mlp_input_dim = self.user_embedding.embedding_dim + self.movie_embedding.embedding_dim + \
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


    def forward(self, user_id, movie_history, target_movie_id, movie_history_ratings, sex, age_group, occupation):
        movie_history_embedding = self.movie_embedding(movie_history)
        target_movie_embedding = self.movie_embedding(target_movie_id).unsqueeze(1)
        transformer_input = torch.cat([movie_history_embedding, target_movie_embedding], dim=1)






