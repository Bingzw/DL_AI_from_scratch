import unittest
import torch
import pytorch_lightning as pl
from torchsummary import summary
from autoencoder.vaenet import VAEEncoder, VAEDecoder, VAE
from autoencoder.util import set_seed


class TestVAENet(unittest.TestCase):
    def test_vae_encoder(self):
        set_seed(100)
        num_input_channels = 3
        base_channels = 8
        latent_dim = 10
        act_fn = torch.nn.GELU
        vae_encoder = VAEEncoder(num_input_channels=num_input_channels, base_channels=base_channels, latent_dim=latent_dim,
                             act_fn=act_fn)
        x = torch.randn(1, num_input_channels, 32, 32)
        mu, log_var = vae_encoder(x)
        summary(vae_encoder, (3, 32, 32))
        self.assertEqual(mu.shape, (1, latent_dim))
        self.assertEqual(log_var.shape, (1, latent_dim))
        self.assertEqual(torch.any(torch.isnan(mu)), False)
        self.assertEqual(torch.any(torch.isnan(mu)), False)

    def test_vae_decoder(self):
        set_seed(100)
        num_input_channels = 3
        base_channels = 8
        latent_dim = 10
        act_fn = torch.nn.GELU
        vae_decoder = VAEDecoder(num_input_channels=num_input_channels, base_channels=base_channels, latent_dim=latent_dim,
                             act_fn=act_fn)
        x = torch.randn(1, latent_dim)
        y = vae_decoder(x)
        summary(vae_decoder, (latent_dim,))
        self.assertEqual(y.shape, (1, num_input_channels, 32, 32))
        self.assertEqual(torch.any(torch.isnan(y)), False)

    class TestVAE(unittest.TestCase):
        def setUp(self):
            self.VAE = VAE(base_channel_size=64, latent_dim=128)
            self.input_tensor = torch.randn(1, 3, 32, 32)
            self.trainer = pl.Trainer(fast_dev_run=True)

        def test_forward(self):
            output_tensor = self.VAE(self.input_tensor)
            self.assertEqual(self.input_tensor.shape, output_tensor.shape)

        def test_training_step(self):
            self.trainer.fit(self.VAE)

        def test_validation_step(self):
            self.trainer.validate(self.VAE)

        def test_test_step(self):
            self.trainer.test(self.VAE)


if __name__ == "__main__":
    unittest.main()