import unittest
import torch
from autoregressive.pixelCNN import PixelCNN


class TestPixelCNN(unittest.TestCase):
    def setUp(self):
        self.c_in = 3
        self.c_hidden = 64
        self.model = PixelCNN(self.c_in, self.c_hidden)
        self.img_shape = (1, self.c_in, 28, 28)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_forward(self):
        input_img = torch.randint(0, 256, self.img_shape, dtype=torch.long)
        output = self.model(input_img)
        self.assertEqual(output.shape, (self.img_shape[0], 256, self.c_in, self.img_shape[2], self.img_shape[3]))

    def test_calc_likelihood(self):
        input_img = torch.randint(0, 256, self.img_shape, dtype=torch.long)
        loss = self.model.calc_likelihood(input_img)
        self.assertIsInstance(loss, torch.Tensor)

    def test_sample(self):
        # Test sampling function
        with torch.no_grad():
            sampled_img = self.model.sample(self.img_shape, self.device)
            self.assertEqual(sampled_img.shape, self.img_shape)


if __name__ == '__main__':
    unittest.main()
