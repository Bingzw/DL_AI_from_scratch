import unittest
import torch
import pytorch_lightning as pl
from torchsummary import summary
from normalizingflow.normalizing_flow import Dequantization
from normalizingflow.util import set_seed


class TestNF(unittest.TestCase):
    def test_dequantization(self):
        set_seed(100)
        orig_img = torch.rand(1, 3, 32, 32)
        ldj = torch.zeros(1, )
        dequant_module = Dequantization()
        deq_img, ldj = dequant_module(orig_img, ldj, reverse=False)
        reconst_img, ldj = dequant_module(deq_img, ldj, reverse=True)
        self.assertEqual(reconst_img.shape, orig_img.shape)


if __name__ == "__main__":
    unittest.main()