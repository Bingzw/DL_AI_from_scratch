import unittest
import torch
from torchsummary import summary
from energy_net.energynet import CNNModel


class TestEnergyNet(unittest.TestCase):
    def test_cnn(self):
        hidden_features = 32
        out_dim = 1
        cnn = CNNModel(hidden_features=hidden_features, out_dim=out_dim)
        x = torch.randn(1, 1, 28, 28)
        y = cnn(x)
        summary(cnn, (1, 28, 28))

        self.assertEqual(torch.any(torch.isnan(y)), False)


if __name__ == "__main__":
    unittest.main()