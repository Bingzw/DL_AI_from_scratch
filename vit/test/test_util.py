import unittest
import torch
from vit.util import img_to_patch


class TestViTUtil(unittest.TestCase):
    def test_img_to_patch(self):
        x = torch.randn(1, 3, 28, 28)
        patch_size = 4
        x_patch_flatten = img_to_patch(x, patch_size, flatten_channels=True)
        x_patch_unflatten = img_to_patch(x, patch_size, flatten_channels=False)
        self.assertEqual(x_patch_flatten.shape, (1, 49, 48))
        self.assertEqual(x_patch_unflatten.shape, (1, 49, 3, 4, 4))


if __name__ == '__main__':
    unittest.main()
