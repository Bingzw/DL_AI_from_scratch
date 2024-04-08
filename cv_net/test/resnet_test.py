import unittest
import torch
from torchsummary import summary

from cv_net.resnet import ResNetBlock, PreActResNetBlock, ResNet
from cv_net.util.util import set_seed


class TestResNet(unittest.TestCase):
    def test_inception_block(self):
        set_seed(100)
        c_in = 3
        act_fn = torch.nn.ReLU
        resnet_block = ResNetBlock(c_in=c_in, act_fn=act_fn)
        set_seed(100)
        x = torch.randn(1, c_in, 2, 2)
        y = resnet_block(x)
        y_expected = torch.Tensor(
            [[[[0.0000, 0.9553],
               [0.0000, 0.9668]],

              [[0.0000, 0.0000],
               [0.0000, 0.0000]],

              [[1.4915, 0.1243],
               [0.0000, 1.7678]]]]
        )
        self.assertEqual(y.shape, (1, 3, 2, 2))
        torch.testing.assert_close(y, y_expected, rtol=1e-3, atol=1e-3)

    def test_preact_resnet_block(self):
        set_seed(100)
        c_in = 3
        act_fn = torch.nn.ReLU
        resnet_block = PreActResNetBlock(c_in=c_in, act_fn=act_fn)
        set_seed(100)
        x = torch.randn(1, c_in, 2, 2)
        y = resnet_block(x)
        y_expected = torch.Tensor(
            [[[[0.0238, -0.1236],
               [-0.4827, 0.0728]],

              [[-1.3607, -1.9575],
               [-0.3679, -0.5799]],

              [[1.5147, 0.1103],
               [-1.1893, 0.8302]]]]
        )
        self.assertEqual(y.shape, (1, 3, 2, 2))
        torch.testing.assert_close(y, y_expected, rtol=1e-3, atol=1e-3)

    def test_resnet(self):
        set_seed(100)
        num_classes = 5
        c_in = 3
        act_fn_name="relu"
        resnet = ResNet(num_classes=num_classes, act_fn_name=act_fn_name)
        set_seed(100)
        x = torch.randn(1, c_in, 32, 32)
        y = resnet(x)
        summary(resnet, (3, 32, 32))
        self.assertEqual(y.shape, (1, num_classes))


if __name__ == "__main__":
    unittest.main()