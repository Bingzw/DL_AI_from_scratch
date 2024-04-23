import unittest
from torchvision.datasets import FakeData
from torch.utils.data import DataLoader
from torchvision import transforms
from selfsupervised.simclr import SimCLR
from selfsupervised.util import ContrastiveTransformations


class TestSimCLR(unittest.TestCase):
    def setUp(self):
        # Set up a fake dataset and a DataLoader for testing
        contrast_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                  transforms.RandomResizedCrop(size=96),
                                                  transforms.RandomApply([
                                                      transforms.ColorJitter(brightness=0.5,
                                                                             contrast=0.5,
                                                                             saturation=0.5,
                                                                             hue=0.1)
                                                  ], p=0.8),
                                                  transforms.RandomGrayscale(p=0.2),
                                                  transforms.GaussianBlur(kernel_size=9),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5,), (0.5,))
                                                  ])
        transform = ContrastiveTransformations(contrast_transforms, n_views=2)
        self.dataset = FakeData(size=100, image_size=(3, 32, 32), transform=transform)
        self.dataloader = DataLoader(self.dataset, batch_size=10)

    def test_initialization(self):
        # Test whether the SimCLR module can be initialized correctly
        model = SimCLR(hidden_dim=128, lr=0.001, temperature=0.1, weight_decay=0.0001, max_epochs=10)
        self.assertIsNotNone(model)

    def test_info_nce_loss(self):
        # Test whether the info_nce_loss method runs without errors
        model = SimCLR(hidden_dim=128, lr=0.001, temperature=0.1, weight_decay=0.0001, max_epochs=10)
        batch = next(iter(self.dataloader))
        loss = model.info_nce_loss(batch, mode='train')
        self.assertIsNotNone(loss)


if __name__ == '__main__':
    unittest.main()