import torch.nn as nn


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
