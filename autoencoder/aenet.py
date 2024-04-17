import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channels: int, latent_dim: int, act_fn: object = nn.GELU):
        """
        :param num_input_channels: input channels of image, 3 for CIFAR
        :param base_channels: number of channels in the first layer
        :param latent_dim: dimensionality of latent representation z
        :param acf_fn: activation funcion used throughout the network
        """
        super(Encoder, self).__init__()
        c_hid = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 32*32 -> 16*16
            act_fn(),
            nn.LayerNorm([c_hid, 16, 16]),  # apply layer normalization instead of batch normalization to avoid
            # correlation between data
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),  # 16*16 -> 16*16
            act_fn(),
            nn.LayerNorm([c_hid, 16, 16]),
            nn.Conv2d(c_hid, c_hid*2, kernel_size=3, padding=1, stride=2),  # 16*16 -> 8*8
            act_fn(),
            nn.LayerNorm([c_hid*2, 8, 8]),
            nn.Conv2d(c_hid*2, c_hid*2, kernel_size=3, padding=1),  # 8*8 -> 8*8
            act_fn(),
            nn.LayerNorm([c_hid*2, 8, 8]),
            nn.Conv2d(c_hid*2, c_hid*2, kernel_size=3, padding=1, stride=2),  # 8*8 -> 4*4
            act_fn(),
            nn.LayerNorm([c_hid*2, 4, 4]),
            nn.Flatten(),
            nn.Linear(c_hid*2*4*4, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channels: int, latent_dim: int, act_fn: object = nn.GELU):
        """
        :param num_input_channels: input channels of image, 3 for CIFAR
        :param base_channels: number of channels in the first layer
        :param latent_dim: dimensionality of latent representation z
        :param acf_fn: activation funcion used throughout the network
        """
        super(Decoder, self).__init__()
        c_hid = base_channels
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, c_hid*2*4*4),
            act_fn(),
            nn.LayerNorm([c_hid*2*4*4]),
            nn.Unflatten(1, (c_hid*2, 4, 4)),
            nn.ConvTranspose2d(c_hid*2, c_hid*2, kernel_size=3, padding=1, stride=2, output_padding=1),  # 4*4 -> 8*8
            # (H - 1)*stride - 2*padding + kernel_size + output_padding
            act_fn(),
            nn.LayerNorm([c_hid*2, 8, 8]),
            nn.ConvTranspose2d(c_hid*2, c_hid*2, kernel_size=3, padding=1),  # 8*8 -> 8*8
            act_fn(),
            nn.LayerNorm([c_hid*2, 8, 8]),
            nn.ConvTranspose2d(c_hid*2, c_hid, kernel_size=3, padding=1, stride=2, output_padding=1),  # 8*8 -> 16*16
            act_fn(),
            nn.LayerNorm([c_hid, 16, 16]),
            nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, padding=1),  # 16*16 -> 16*16
            act_fn(),
            nn.LayerNorm([c_hid, 16, 16]),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, padding=1, stride=2, output_padding=1),  # 16*16 -> 32*32
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)


class Autoencoder(pl.LightningModule):
    def __init__(self, base_channel_size: int, latent_dim: int, encoder_class: object = Encoder,
                 decoder_class: object = Decoder, num_input_channels: int = 3, act_fn: object = nn.GELU,
                 width: int = 32, height: int = 32):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder_class(num_input_channels=3, base_channels=base_channel_size, latent_dim=latent_dim,
                                     act_fn=act_fn)
        self.decoder = decoder_class(num_input_channels=3, base_channels=base_channel_size, latent_dim=latent_dim,
                                     act_fn=act_fn)
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        x, _ = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x, reduction="none")
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)


