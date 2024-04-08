import unittest
import torch
from torchsummary import summary

from cv_net.googlenet import InceptionBlock, GoogleNet
from cv_net.util.util import set_seed


class TestGoogleNet(unittest.TestCase):
    def test_inception_block(self):
        set_seed(100)
        c_in = 3
        c_red = {"3x3": 2, "5x5": 4}
        c_out = {"1x1": 2, "3x3": 2, "5x5": 2, "max": 4}
        act_fn = torch.nn.ReLU
        inception_block = InceptionBlock(c_in, c_red, c_out, act_fn)
        set_seed(100)
        x = torch.randn(1, c_in, 2, 2)
        y = inception_block(x)
        y_expected = torch.Tensor(
            [[[[0.0000, 0.0000],
               [1.5678, 0.0000]],

              [[1.3663, 0.0000],
               [0.0000, 0.4644]],

              [[0.9123, 0.0000],
               [0.0000, 1.0386]],

              [[0.9585, 0.0000],
               [0.9369, 0.0000]],

              [[0.0000, 1.6249],
               [0.0000, 0.0000]],

              [[0.9317, 0.0000],
               [1.0485, 0.0000]],

              [[0.0000, 0.0000],
               [0.0000, 0.0000]],

              [[0.0000, 0.0000],
               [0.0000, 0.0000]],

              [[0.0000, 0.0000],
               [0.0000, 0.0000]],

              [[0.0000, 0.0000],
               [0.0000, 0.0000]]]]
        )
        self.assertEqual(y.shape, (1, 10, 2, 2))
        torch.testing.assert_close(y, y_expected, rtol=1e-3, atol=1e-3)

    def test_googlenet(self):
        set_seed(100)
        num_classes = 5
        c_in = 3
        act_fn_name="relu"
        google_net = GoogleNet(num_classes=num_classes, act_fn_name=act_fn_name)
        set_seed(100)
        x = torch.randn(1, c_in, 32, 32)
        y = google_net(x)
        summary(google_net, (3, 32, 32))
        y_expected = torch.Tensor([[ 0.2634, -0.2414, -0.0182,  0.2073, -0.2675]])
        torch.testing.assert_close(y, y_expected, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
