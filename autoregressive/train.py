import os
import matplotlib.pyplot as plt
# PyTorch
import torch
import torch.utils.data as data
# Torchvision
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pixelCNN import PixelCNN
from util import set_seed, discretize, show_imgs


if __name__ == "__main__":
    SEED = 42
    CHECKPOINT_PATH = "../saved_models/autoregressive/"
    # set seed
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # download dataset
    DATASET_PATH = "../data/"
    # Transformations applied on each image => only make them a tensor
    transform = transforms.Compose([transforms.ToTensor(), discretize])

    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = MNIST(root=DATASET_PATH, train=True, transform=transform, download=True)
    pl.seed_everything(42)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])

    # Loading the test set
    test_set = MNIST(root=DATASET_PATH, train=False, transform=transform, download=True)

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True,
                                   num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)

    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "PixelCNN"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=2,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_bpd"),
                                    LearningRateMonitor("epoch")])
    result = None
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_model_name = ".ckpt"  # the correct path should be
    # pretrained_model_name = "/lightning_logs/version_0/checkpoints/epoch=77-step=36582.ckpt"
    if os.path.isfile(pretrained_model_name):
        print("Found pretrained model, loading...")
        model = PixelCNN.load_from_checkpoint(pretrained_model_name)
        ckpt = torch.load(pretrained_model_name, map_location=device)
        result = ckpt.get("result", None)
    else:
        model = PixelCNN(c_in=1, c_hidden=64)
        trainer.fit(model, train_loader, val_loader)
    model = model.to(device)

    if result is None:
        # Test best model on validation and test set
        val_result = trainer.test(model, val_loader, verbose=False)
        test_result = trainer.test(model, test_loader, verbose=False)
        result = {"test": test_result, "val": val_result}

    test_res = result["test"][0]
    print("Test bits per dimension: %4.3fbpd" % (
        test_res["test_loss"] if "test_loss" in test_res else test_res["test_bpd"]))

    # sampling
    pl.seed_everything(1)
    samples = model.sample(img_shape=(16, 1, 28, 28), device=device)
    show_imgs(samples.cpu())

