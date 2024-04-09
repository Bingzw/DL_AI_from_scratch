import unittest
import torch
from torchsummary import summary

from cv_net.densenet import DenseLayer, DenseBlock, DenseNet
from cv_net.util.util import set_seed


class TestDenseNet(unittest.TestCase):
    def test_dense_layer(self):
        set_seed(100)
        c_in = 3
        bn_size = 2
        growth_rate = 3
        act_fn = torch.nn.ReLU
        dense_layer = DenseLayer(c_in=c_in, bn_size=bn_size, growth_rate=growth_rate, act_fn=act_fn)
        set_seed(100)
        x = torch.randn(1, c_in, 2, 2)
        y = dense_layer(x)
        self.assertEqual(y.shape, (1, 6, 2, 2))
        self.assertEqual(torch.any(torch.isnan(y)), False)

    def test_dense_block(self):
        set_seed(100)
        c_in = 3
        num_layers = 2
        bn_size = 2
        growth_rate = 3
        act_fn = torch.nn.ReLU
        densenet_block = DenseBlock(c_in=c_in, num_layers=num_layers, bn_size=bn_size, growth_rate=growth_rate, act_fn=act_fn)
        set_seed(100)
        x = torch.randn(1, c_in, 2, 2)
        y = densenet_block(x)
        self.assertEqual(y.shape, (1, 9, 2, 2))
        self.assertEqual(torch.any(torch.isnan(y)), False)

    def test_dense_net(self):
        set_seed(100)
        num_classes = 10
        num_layers = [2, 2]
        bn_size = 2
        growth_rate = 3
        densenet = DenseNet(num_classes=num_classes,
                            num_layers=num_layers,
                            bn_size=bn_size,
                            growth_rate=growth_rate,
                            act_fn_name="relu")
        set_seed(100)
        x = torch.randn(1, 3, 32, 32)
        y = densenet(x)
        summary(densenet, (3, 32, 32))
        self.assertEqual(y.shape, (1, 10))
        self.assertEqual(torch.any(torch.isnan(y)), False)


if __name__ == "__main__":
    unittest.main()