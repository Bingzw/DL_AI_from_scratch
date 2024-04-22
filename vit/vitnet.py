import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from vit.util import img_to_patch


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Pre-LayerNorm Attention Block for vit
        :param embed_dim: the dimension of the input and attention feature embeddings
        :param hidden_dim: the dimension of the hidden layer in the feedforward network
        :param num_heads: the number of attention heads
        :param dropout: amount of dropout to apply
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        input_x = self.layer_norm_1(x)
        x = x + self.attn(query=input_x, key=input_x, value=input_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches,
                 dropout=0.0):
        """
        Vision Transformer
        :param embed_dim: the dimension of the input and attention feature embeddings
        :param hidden_dim: the dimension of the hidden layer in the feedforward network
        :param num_channels: the number of channels in the input image
        :param num_heads: the number of attention heads
        :param num_layers: the number of attention blocks
        :param num_classes: the number of classes in the dataset
        :param patch_size: the size of the patches
        :param num_patches: the number of patches a image can has: H/patch_size * W/patch_size
        :param dropout: amount of dropout to apply
        """
        super().__init__()

        self.patch_size = patch_size
        self.input_layer = nn.Linear(num_channels*(patch_size**2), embed_dim)
        self.transformer = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout)
                                           for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # a token used to represent the whole image
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, embed_dim))

    def forward(self, x):
        # x shape: [B, C, H, W]
        x = img_to_patch(x, self.patch_size)  # [B, num_patches=(H/patch_size)*(W/patch_size), C*patch_size^2]
        B, T, _ = x.shape
        x = self.input_layer(x)  # [B, num_patches, embed_dim]

        # add cls token and position embedding
        cls_token = self.cls_token.repeat(B, 1, 1)  # [B, 1, embed_dim]
        x = torch.cat([cls_token, x], dim=1)  # [B, 1+num_patches, embed_dim]
        x = x + self.pos_embedding[:, :T+1]  # [B, 1+num_patches, embed_dim]

        # apply transformer
        x = self.dropout(x)
        x = x.transpose(0, 1)  # [1+num_patches, B, embed_dim]
        x = self.transformer(x)  # [1+num_patches, B, embed_dim]

        # get the cls token and apply the mlp head
        cls = x[0]  # [B, embed_dim]
        out = self.mlp_head(cls)  # [B, num_classes]
        return out


class ViT(pl.LightningModule):
    def __init__(self, model_kwargs, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionTransformer(**model_kwargs)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calcualte_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calcualte_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._calcualte_loss(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._calcualte_loss(batch, "test")

