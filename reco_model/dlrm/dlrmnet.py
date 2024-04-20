import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score


class FeatureInteraction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        """
        :param inputs: the input tensor of shape (batch_size, feature_dim)
        :return: the interaction tensor of shape (batch_size, out_dim)
        """
        feature_dim = inputs.shape[1]

        concat_features = inputs.view(-1, feature_dim, 1)
        dot_products = torch.matmul(concat_features, concat_features.transpose(1, 2))
        ones = torch.ones_like(dot_products)

        mask = torch.triu(ones)
        out_dim = feature_dim * (feature_dim + 1) // 2

        flat_result = dot_products[mask.bool()]
        reshape_result = flat_result.view(-1, out_dim)

        return reshape_result


class DLRMNet(nn.Module):
    def __init__(self, num_dense_features, embedding_sizes, bottom_mlp_dims, top_mlp_dims):
        """
        :param num_dense_features: the number of numerical features
        :param embedding_sizes: a list of tuples where each tuple contains the number of categories and the embedding
        :param bottom_mlp_dims: the dimensions of the bottom MLP
        :param top_mlp_dims: the dimensions of the top MLP
        """
        super(DLRMNet, self).__init__()
        self.num_dense_features = num_dense_features

        # Embedding layers for categorical features
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(num_categories, embedding_dim) for num_categories, embedding_dim in embedding_sizes])
        self.embedding_output_dim = sum([embedding_dim for _, embedding_dim in embedding_sizes])

        # Bottom MLP
        bottom_layers = []
        bottom_layers.append(nn.Linear(num_dense_features, bottom_mlp_dims[0]))
        bottom_layers.append(nn.ReLU())
        for i in range(1, len(bottom_mlp_dims)):
            bottom_layers.append(nn.Linear(bottom_mlp_dims[i - 1], bottom_mlp_dims[i]))
            bottom_layers.append(nn.ReLU())
        self.bottom_mlp = nn.Sequential(*bottom_layers)

        # Interaction layer
        self.interaction_input_dim = bottom_mlp_dims[-1] + self.embedding_output_dim
        self.interaction_output_dim = (self.interaction_input_dim + 1) * self.interaction_input_dim // 2
        self.interaction_layer = FeatureInteraction()

        # Top MLP
        top_layers = []
        top_layers.append(nn.Linear(bottom_mlp_dims[-1] + self.interaction_output_dim, top_mlp_dims[0]))
        top_layers.append(nn.ReLU())
        for i in range(1, len(top_mlp_dims)):
            top_layers.append(nn.Linear(top_mlp_dims[i - 1], top_mlp_dims[i]))
            top_layers.append(nn.ReLU())
        top_layers.append(nn.Linear(top_mlp_dims[-1], 1))
        self.top_mlp = nn.Sequential(*top_layers)

    def forward(self, dense_features, sparse_features):
        """
        :param dense_features: the numerical features tensor
        :param sparse_features: the categorical features tensor
        :return: the output tensor
        """
        # Pass dense features through bottom MLP
        dense_output = self.bottom_mlp(dense_features)

        # Embed categorical features
        embeddings = [embedding_layer(sparse_features[:, i]) for i, embedding_layer in
                      enumerate(self.embedding_layers)]
        embeddings = torch.cat(embeddings, dim=1)

        # Interaction layer: element-wise product between embeddings and bottom MLP output
        interaction = self.interaction_layer(torch.cat([dense_output, embeddings], dim=1))

        # Concatenate bottom MLP output, embeddings, and interactions
        x = torch.cat([dense_output, interaction], dim=1)

        # Pass through top MLP
        x = self.top_mlp(x)

        return x


class DLRMModule(pl.LightningModule):
    def __init__(self, num_dense_features, embedding_sizes, bottom_mlp_dims, top_mlp_dims, lr=1e-3):
        """
        :param num_dense_features: the number of numerical features
        :param embedding_sizes: a list of tuples where each tuple contains the number of categories and the embedding
        :param bottom_mlp_dims: the dimensions of the bottom MLP
        :param top_mlp_dims: the dimensions of the top MLP
        :param lr: the learning rate
        """
        super(DLRMModule, self).__init__()
        self.model = DLRMNet(num_dense_features, embedding_sizes, bottom_mlp_dims, top_mlp_dims)
        self.lr = lr
        self.loss = nn.BCEWithLogitsLoss()
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
        loss = nn.BCEWithLogitsLoss()(outputs, labels.float())
        self.log('val_loss', loss)
        self.val_loss.append(loss)
        # Convert predictions to binary (0 or 1) using sigmoid function
        preds = torch.sigmoid(outputs)
        result ={'loss': loss, 'preds': preds, 'targets': labels}
        self.validation_step_outputs.append(result)

    def test_step(self, batch, batch_idx):
        dense_features, sparse_features, labels = batch
        outputs = self(dense_features, sparse_features).squeeze()
        loss = nn.BCEWithLogitsLoss()(outputs, labels.float())
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
