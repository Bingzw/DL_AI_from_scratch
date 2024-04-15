## PyTorch
import torch
import torch.nn as nn
# Torchvision
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from util import set_seed




if __name__ == "__main__":
    SEED = 42
    CHECKPOINT_PATH = "../saved_models/energy_net"
    # set seed
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # download dataset
    DATASET_PATH = "../data/"
    train_dataset = MNIST(root=DATASET_PATH, train=True, download=True)
    print(f"Number of training samples: {len(train_dataset)}")
