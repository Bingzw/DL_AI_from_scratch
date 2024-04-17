import unittest
import torch
import pytorch_lightning as pl
from torchsummary import summary
from autoencoder.aenet import Encoder, Decoder, Autoencoder
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

    class TestAutoencoder(unittest.TestCase):
        def setUp(self):
            self.autoencoder = Autoencoder(base_channel_size=64, latent_dim=128)
            self.input_tensor = torch.randn(1, 3, 32, 32)
            self.trainer = pl.Trainer(fast_dev_run=True)

        def test_forward(self):
            output_tensor = self.autoencoder(self.input_tensor)
            self.assertEqual(self.input_tensor.shape, output_tensor.shape)

        def test_training_step(self):
            self.trainer.fit(self.autoencoder)

        def test_validation_step(self):
            self.trainer.validate(self.autoencoder)

        def test_test_step(self):
            self.trainer.test(self.autoencoder)


if __name__ == "__main__":
    unittest.main()