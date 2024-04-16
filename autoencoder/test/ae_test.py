import unittest
import torch
from torchsummary import summary

from autoencoder.aenet import Encoder, Decoder
from autoencoder.util import set_seed


class TestAENet(unittest.TestCase):
    def test_ae_encoder(self):
        set_seed(100)
        num_input_channels = 3
        base_channels = 8
        latent_dim = 10
        act_fn = torch.nn.GELU
        ae_encoder = Encoder(num_input_channels=num_input_channels, base_channels=base_channels, latent_dim=latent_dim,
                             act_fn=act_fn)
        x = torch.randn(1, num_input_channels, 32, 32)
        y = ae_encoder(x)
        summary(ae_encoder, (3, 32, 32))
        self.assertEqual(y.shape, (1, latent_dim))
        self.assertEqual(torch.any(torch.isnan(y)), False)

    def test_ae_decoder(self):
        set_seed(100)
        num_input_channels = 3
        base_channels = 8
        latent_dim = 10
        act_fn = torch.nn.GELU
        ae_decoder = Decoder(num_input_channels=num_input_channels, base_channels=base_channels, latent_dim=latent_dim,
                             act_fn=act_fn)
        x = torch.randn(1, latent_dim)
        y = ae_decoder(x)
        summary(ae_decoder, (latent_dim,))
        self.assertEqual(y.shape, (1, num_input_channels, 32, 32))
        self.assertEqual(torch.any(torch.isnan(y)), False)


if __name__ == "__main__":
    unittest.main()