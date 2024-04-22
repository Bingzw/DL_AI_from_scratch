import unittest
import torch
from vit.vitnet import AttentionBlock, VisionTransformer


class TestViTNet(unittest.TestCase):
    def test_attention_block(self):
        x = torch.randn(1, 16, 64)  # [Batch, Sequence Length, Embedding Dimension]
        attn_block = AttentionBlock(64, 128, 8, 0.1)
        out = attn_block(x)
        self.assertEqual(out.shape, (1, 16, 64))

    def testViTTransformer(self):
        x = torch.randn(1, 3, 28, 28)
        embed_dim = 64
        hidden_dim = 128
        num_channels = 3
        num_heads = 4
        num_layers = 2
        num_classes = 10
        patch_size = 4
        num_patches = 49
        dropout = 0.1
        model = VisionTransformer(embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size,
                                  num_patches, dropout)
        out = model(x)
        self.assertEqual(out.shape, (1, num_classes))


if __name__ == '__main__':
    unittest.main()