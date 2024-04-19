import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score


class FactorizationMachine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        :param x: the input tensor of shape (batch_size, feature_dim), it includes both dense and sparse features
        :return: factorized output tensor
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        output = 0.5 * (square_of_sum - sum_of_square)
        return output.view(-1, 1)


class DeepFMNet(nn.Module):
    def __init__(self, num_dense_features, embedding_sizes, mlp_dims):
        """
        :param num_dense_features: the number of numerical features
        :param embedding_sizes: a list of tuples where each tuple contains the number of categories and the
        embedding dimension
        """
        super().__init__()
        self.dense_features = num_dense_features

        # Embedding layers for categorical features
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(num_categories, embedding_dim) for num_categories, embedding_dim in embedding_sizes])
        self.embedding_output_dim = sum([embedding_dim for _, embedding_dim in embedding_sizes])

        # Factorization Machine layer
        self.fm = FactorizationMachine()

        # MLP layer for the concated dense and sparse outputs
        mlp_layers = []
        mlp_layers.append(nn.Linear(self.dense_features + self.embedding_output_dim, mlp_dims[0]))
        mlp_layers.append(nn.ReLU())
        for i in range(1, len(mlp_dims)):
            mlp_layers.append(nn.Linear(mlp_dims[i - 1], mlp_dims[i]))
            mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(mlp_dims[-1], 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, dense_features, sparse_features):
        """
        :param dense_features: the dense features
        :param sparse_features: the sparse features
        :return: the output tensor, the shape is (batch_size, 1)
        """
        # Embed the sparse features
        sparse_embeddings = [embedding_layer(sparse_features[:, i]) for i, embedding_layer in
                             enumerate(self.embedding_layers)]
        sparse_embeddings = torch.cat(sparse_embeddings, dim=1)

        # Concatenate dense and sparse features
        x = torch.cat([dense_features, sparse_embeddings], dim=1)

        # Factorization Machine
        x_fm = self.fm(x)

        # Pass through MLP
        x_mlp = self.mlp(x)

        # Combine FM and MLP outputs
        x = x_fm + x_mlp
        x = torch.sigmoid(x)
        return x


class DeepFMModule(pl.LightningModule):
    def __init__(self, num_dense_features, embedding_sizes, mlp_dims, lr=1e-3):
        """
        :param num_dense_features: the number of numerical features
        :param embedding_sizes: a list of tuples where each tuple contains the number of categories and the embedding
        :param mlp_dims: the dimensions of the MLP
        :param lr: the learning rate
        """
        super().__init__()
        self.model = DeepFMNet(num_dense_features, embedding_sizes, mlp_dims)
        self.lr = lr
        self.loss = nn.BCELoss()  # apply BCELOSS since we have applied sigmoid in deepfm net
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.val_loss = []

    def forward(self, dense_features, sparse_features):
        return self.model(dense_features, sparse_features)

    def training_step(self, batch, batch_idx):
        dense_features, sparse_features, labels = batch
        outputs = self(dense_features, sparse_features).squeeze()
        loss = self.loss(outputs, labels.float())
        self.log('train_loss', loss)
        preds = torch.sigmoid(outputs)
        result ={'loss': loss, 'preds': preds, 'targets': labels}
        self.training_step_outputs.append(result)
        return loss

    def validation_step(self, batch, batch_idx):
        dense_features, sparse_features, labels = batch
        outputs = self(dense_features, sparse_features).squeeze()
        loss = self.loss(outputs, labels.float())
        self.log('val_loss', loss)
        self.val_loss.append(loss)
        # Convert predictions to binary (0 or 1) using sigmoid function
        preds = torch.sigmoid(outputs)
        result ={'loss': loss, 'preds': preds, 'targets': labels}
        self.validation_step_outputs.append(result)

    def test_step(self, batch, batch_idx):
        dense_features, sparse_features, labels = batch
        outputs = self(dense_features, sparse_features).squeeze()
        loss = self.loss(outputs, labels.float())
        self.log('test_loss', loss)
        preds = torch.sigmoid(outputs)
        result = {'loss': loss, 'preds': preds, 'targets': labels}
        self.test_step_outputs.append(result)

    def on_train_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.training_step_outputs]).detach().cpu().numpy()
        targets = torch.cat([x['targets'] for x in self.training_step_outputs]).detach().cpu().numpy()
        # Convert predictions to binary (0 or 1) using threshold of 0.5
        binary_preds = torch.from_numpy(preds > 0.5).float().cpu().numpy()
        # Calculate accuracy
        correct = (binary_preds == targets).sum().item()
        acc = correct / len(targets)
        self.log('train_acc', acc, prog_bar=True)
        auc = roc_auc_score(targets, preds)
        self.log('train_auc', auc, prog_bar=True)
        f1 = f1_score(targets, binary_preds)
        self.log('train_f1', f1, prog_bar=True)
        precision = precision_score(targets, binary_preds)
        self.log('train_precision', precision, prog_bar=True)
        recall = recall_score(targets, binary_preds)
        self.log('train_recall', recall, prog_bar=True)
        self.training_step_outputs.clear()  # free memory

    def on_validation_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.validation_step_outputs]).detach().cpu().numpy()
        targets = torch.cat([x['targets'] for x in self.validation_step_outputs]).detach().cpu().numpy()
        # Convert predictions to binary (0 or 1) using threshold of 0.5
        binary_preds = torch.from_numpy(preds > 0.5).float().cpu().numpy()
        # Calculate accuracy
        correct = (binary_preds == targets).sum().item()
        acc = correct / len(targets)
        self.log('val_acc', acc, prog_bar=True)
        auc = roc_auc_score(targets, preds)
        self.log('val_auc', auc, prog_bar=True)
        f1 = f1_score(targets, binary_preds)
        self.log('val_f1', f1, prog_bar=True)
        precision = precision_score(targets, binary_preds)
        self.log('val_precision', precision, prog_bar=True)
        recall = recall_score(targets, binary_preds)
        self.log('val_recall', recall, prog_bar=True)
        self.validation_step_outputs.clear()  # free memory

    def on_test_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.test_step_outputs]).detach().cpu().numpy()
        targets = torch.cat([x['targets'] for x in self.test_step_outputs]).detach().cpu().numpy()
        # Convert predictions to binary (0 or 1) using threshold of 0.5
        binary_preds = torch.from_numpy(preds > 0.5).float().cpu().numpy()
        # Calculate accuracy
        correct = (binary_preds == targets).sum().item()
        acc = correct / len(targets)
        self.log('test_acc', acc, prog_bar=True)
        auc = roc_auc_score(targets, preds)
        self.log('test_auc', auc, prog_bar=True)
        f1 = f1_score(targets, binary_preds)
        self.log('test_f1', f1, prog_bar=True)
        precision = precision_score(targets, binary_preds)
        self.log('test_precision', precision, prog_bar=True)
        recall = recall_score(targets, binary_preds)
        self.log('test_recall', recall, prog_bar=True)
        self.test_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]





