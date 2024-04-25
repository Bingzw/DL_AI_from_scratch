import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, mlp_dims, dropout, output_layer=True):
        """
        :param input_dim: the input dimension
        :param hidden_dims: a list of integers where each integer represents the number of hidden units
        :param dropout: the dropout rate
        :param output_layer: a boolean value indicating whether to include the output layer
        """
        super().__init__()
        layers = []
        for hidden_dim in mlp_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = hidden_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MMOENet(nn.Module):
    """
    MMOE network for multi-task learning
    """
    def __init__(self, num_dense_features, sparse_cardinality, hidden_dim, expert_mlp_dims, tower_mlp_dims, num_tasks,
                 num_experts, dropout):
        """
        :param num_dense_features: the number of numerical features
        :param sparse_cardinality: a list of integers where each integer represents the number of categories for the sparse feature
        :param hidden_dim: the hidden dimension of the embedding layer for sparse features
        :param expert_mlp_dims: the dimensions of the bottom MLP for each expert
        :param tower_mlp_dims: the dimensions of the top tower MLP
        :param num_tasks: the number of tasks
        :param num_experts: the number of experts
        :param dropout: the dropout rate
        """
        super().__init__()
        self.num_dense_features = num_dense_features
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.embedding_sizes = [(num_categories, hidden_dim) for num_categories in sparse_cardinality]

        # Dense layer to handle the numerical features
        self.dense_layer = nn.Linear(num_dense_features, hidden_dim)

        # Embedding layers for categorical features
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(num_categories, embedding_dim) for num_categories, embedding_dim in self.embedding_sizes])
        self.embedding_output_dim = (len(self.embedding_sizes) + 1) * hidden_dim  # contain both dense and sparse embeddings

        self.expert = nn.ModuleList([MultiLayerPerceptron(self.embedding_output_dim, expert_mlp_dims, dropout,
                                                          output_layer=False)
                                     for _ in range(num_experts)])
        self.tower = nn.ModuleList([MultiLayerPerceptron(expert_mlp_dims[-1], tower_mlp_dims, dropout)
                                    for _ in range(num_tasks)])
        self.gate = nn.ModuleList([nn.Sequential(nn.Linear(self.embedding_output_dim, num_experts), nn.Softmax(dim=1))
                                   for _ in range(num_tasks)])

    def forward(self, dense_features, sparse_features):
        """
        :param dense_features: the numerical features tensor
        :param sparse_features: the categorical features tensor
        :return: the output tensor
        """
        # Embed the sparse features
        sparse_embeddings = [embedding_layer(sparse_features[:, i])
                             for i, embedding_layer in enumerate(self.embedding_layers)]  # list of (batch_size, hidden_dim)
        sparse_embeddings = torch.cat(sparse_embeddings, dim=1)  # (batch_size, hidden_dim * len(self.embedding_sizes))
        # Embed the dense features
        dense_embeddings = self.dense_layer(dense_features)  # (batch_size, hidden_dim)
        # Concatenate dense and sparse features
        embedding = torch.cat([dense_embeddings, sparse_embeddings], dim=1)  # (batch_size, hidden_dim * (len(self.embedding_sizes) + 1))

        gate_value = [self.gate[i](embedding).unsqueeze(1) for i in range(self.num_tasks)]  # list of (batch_size, 1, num_experts)
        expert_value = torch.concat([self.expert[i](embedding).unsqueeze(1) for i in range(self.num_experts)], dim=1)  # (batch_size, num_experts, bottom_mlp_dims[-1])
        task_value = [torch.bmm(gate_value[i], expert_value).squeeze(1) for i in range(self.num_tasks)]  # list of (batch_size, bottom_mlp_dims[-1])

        results = [torch.sigmoid(self.tower[i](task_value[i])).squeeze(1) for i in range(self.num_tasks)]  # list of (batch_size, 1)

        return results


class MMOEModule(pl.LightningModule):
    def __init__(self, num_dense_features, sparse_cardinality, hidden_dim, expert_mlp_dims, tower_mlp_dims, num_tasks,
                 num_experts, dropout, lr=1e-3):
        """
        :param num_dense_features: the number of numerical features
        :param sparse_cardinality: a list of integers where each integer represents the number of categories for the sparse feature
        :param hidden_dim: the hidden dimension of the embedding layer for sparse features
        :param expert_mlp_dims: the dimensions of the bottom MLP for each expert
        :param tower_mlp_dims: the dimensions of the top tower MLP
        :param num_tasks: the number of tasks
        :param num_experts: the number of experts
        :param dropout: the dropout rate
        :param lr: the learning rate
        """
        super().__init__()
        self.model = MMOENet(num_dense_features, sparse_cardinality, hidden_dim, expert_mlp_dims, tower_mlp_dims,
                             num_tasks, num_experts, dropout)
        self.lr = lr
        self.loss = nn.BCELoss() # binary cross-entropy loss
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.val_loss = []

    def forward(self, dense_features, sparse_features):
        return self.model(dense_features, sparse_features)

    def training_step(self, batch, batch_idx):
        dense_features, sparse_features, labels = batch
        outputs = self(dense_features, sparse_features).squeeze()
        loss_list = [self.loss(outputs, labels[:, i].float()) for i in range(labels.size(1))]
        loss = sum(loss_list)
        loss = loss / len(loss_list)
        self.log('train_loss', loss)
        result ={'loss': loss, 'preds': outputs, 'targets': labels}
        self.training_step_outputs.append(result)
        return loss

    def validation_step(self, batch, batch_idx):
        dense_features, sparse_features, labels = batch
        outputs = self(dense_features, sparse_features).squeeze()
        loss_list = [self.loss(outputs, labels[:, i].float()) for i in range(labels.size(1))]
        loss = sum(loss_list)
        loss = loss / len(loss_list)
        self.log('val_loss', loss)
        self.val_loss.append(loss)
        result ={'loss': loss, 'preds': outputs, 'targets': labels}
        self.validation_step_outputs.append(result)

    def test_step(self, batch, batch_idx):
        dense_features, sparse_features, labels = batch
        outputs = self(dense_features, sparse_features).squeeze()
        loss_list = [self.loss(outputs, labels[:, i].float()) for i in range(labels.size(1))]
        loss = sum(loss_list)
        loss = loss / len(loss_list)
        self.log('test_loss', loss)
        result = {'loss': loss, 'preds': outputs, 'targets': labels}
        self.test_step_outputs.append(result)

    def on_train_epoch_end(self):
        for i in range(self.num_tasks):
            preds = torch.cat([x['preds'][i] for x in self.training_step_outputs]).detach().cpu().numpy()
            targets = torch.cat([x['targets'][:, i] for x in self.training_step_outputs]).detach().cpu().numpy()
            # Convert predictions to binary (0 or 1) using threshold of 0.5
            binary_preds = torch.from_numpy(preds > 0.5).float().cpu().numpy()
            # Calculate accuracy
            correct = (binary_preds == targets).sum().item()
            acc = correct / len(targets)
            self.log(f'train_acc_task{i}', acc, prog_bar=True)
            auc = roc_auc_score(targets, preds)
            self.log(f'train_auc_task{i}', auc, prog_bar=True)
            f1 = f1_score(targets, binary_preds)
            self.log(f'train_f1_task{i}', f1, prog_bar=True)
            precision = precision_score(targets, binary_preds)
            self.log(f'train_precision_task{i}', precision, prog_bar=True)
            recall = recall_score(targets, binary_preds)
            self.log(f'train_recall_task{i}', recall, prog_bar=True)
        self.training_step_outputs.clear()  # free memory

    def on_validation_epoch_end(self):
        auc_list = []
        for i in range(self.num_tasks):
            preds = torch.cat([x['preds'][i] for x in self.validation_step_outputs]).detach().cpu().numpy()
            targets = torch.cat([x['targets'][:, i] for x in self.validation_step_outputs]).detach().cpu().numpy()
            # Convert predictions to binary (0 or 1) using threshold of 0.5
            binary_preds = torch.from_numpy(preds > 0.5).float().cpu().numpy()
            # Calculate accuracy
            correct = (binary_preds == targets).sum().item()
            acc = correct / len(targets)
            self.log(f'val_acc_task{i}', acc, prog_bar=True)
            auc = roc_auc_score(targets, preds)
            auc_list.append(auc)
            self.log(f'val_auc_task{i}', auc, prog_bar=True)
            f1 = f1_score(targets, binary_preds)
            self.log(f'val_f1_task{i}', f1, prog_bar=True)
            precision = precision_score(targets, binary_preds)
            self.log(f'val_precision_task{i}', precision, prog_bar=True)
            recall = recall_score(targets, binary_preds)
            self.log(f'val_recall_task{i}', recall, prog_bar=True)
        # calculate the aggregated auc
        aggregated_auc = sum(auc_list) / len(auc_list)
        self.log('val_aggregated_auc', aggregated_auc, prog_bar=True)
        self.validation_step_outputs.clear()  # free memory

    def on_test_epoch_end(self):
        for i in range(self.num_tasks):
            preds = torch.cat([x['preds'][i] for x in self.test_step_outputs]).detach().cpu().numpy()
            targets = torch.cat([x['targets'][:, i] for x in self.test_step_outputs]).detach().cpu().numpy()
            # Convert predictions to binary (0 or 1) using threshold of 0.5
            binary_preds = torch.from_numpy(preds > 0.5).float().cpu().numpy()
            # Calculate accuracy
            correct = (binary_preds == targets).sum().item()
            acc = correct / len(targets)
            self.log(f'test_acc_task{i}', acc, prog_bar=True)
            auc = roc_auc_score(targets, preds)
            self.log(f'test_auc_task{i}', auc, prog_bar=True)
            f1 = f1_score(targets, binary_preds)
            self.log(f'test_f1_task{i}', f1, prog_bar=True)
            precision = precision_score(targets, binary_preds)
            self.log(f'test_precision_task{i}', precision, prog_bar=True)
            recall = recall_score(targets, binary_preds)
            self.log(f'test_recall_task{i}', recall, prog_bar=True)
        self.test_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]






