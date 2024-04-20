import os
import matplotlib.pyplot as plt
# PyTorch
import torch
import torch.utils.data as data
# Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from autoencoder.aenet import Autoencoder
from autoencoder.vaenet import VAE
from util import set_seed, compare_imgs, visualize_reconstructions


def get_train_images(num):
    return torch.stack([train_dataset[i][0] for i in range(num)], dim=0)


class GenerateCallback(pl.Callback):
    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconstructed_imgs = pl_module(input_imgs)
                pl_module.train()
            # Check if reconstructed_imgs is a tuple for vae
            if isinstance(reconstructed_imgs, tuple):
                reconstructed_imgs = reconstructed_imgs[0]
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconstructed_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, value_range=(-1, 1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)


def train_cifar(latent_dim, model_type="ae"):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, f"cifar10_{latent_dim}_{model_type}"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=6,
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                    GenerateCallback(get_train_images(8), every_n_epochs=2),
                                    LearningRateMonitor("epoch")])
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"cifar10_{latent_dim}_{model_type}" + pretrained_model_name)
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        if model_type == "ae":
            model = Autoencoder.load_from_checkpoint(pretrained_filename)
        elif model_type == "vae":
            model = VAE.load_from_checkpoint(pretrained_filename)
        else:
            raise ValueError("Unknown model type")
    else:
        if model_type == "ae":
            model = Autoencoder(base_channel_size=32, latent_dim=latent_dim)
        elif model_type == "vae":
            model = VAE(base_channel_size=32, latent_dim=latent_dim)
        else:
            raise ValueError("Unknown model type")
        trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    test_result = trainer.test(model, test_loader, verbose=False)
    result_dict = {"test": test_result[0]}

    return model, result_dict


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
    pretrained_model_name = ".ckpt"  # the correct path should be
    # pretrained_model_name = "/lightning_logs/version_0/checkpoints/epoch=77-step=36582.ckpt"
    # Transformations applied on each image => only make them a tensor
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=transform, download=True)
    pl.seed_everything(42)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [45000, 5000])

    # Loading the test set
    test_set = CIFAR10(root=DATASET_PATH, train=False, transform=transform, download=True)

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(train_set, batch_size=256, shuffle=True, drop_last=True, pin_memory=True,
                                   num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)

    """
    # check a few examples
    for i in range(2):
        # Load example image
        img, _ = train_dataset[i]
        img_mean = img.mean(dim=[1, 2], keepdims=True)

        # Shift image by one pixel
        SHIFT = 1
        img_shifted = torch.roll(img, shifts=SHIFT, dims=1)
        img_shifted = torch.roll(img_shifted, shifts=SHIFT, dims=2)
        img_shifted[:, :1, :] = img_mean
        img_shifted[:, :, :1] = img_mean
        compare_imgs(img, img_shifted, "Shifted -")

        # Set half of the image to zero
        img_masked = img.clone()
        img_masked[:, :img_masked.shape[1] // 2, :] = img_mean
        compare_imgs(img, img_masked, "Masked -")
    """

    # Train the model
    model_dict = {}
    for model_type in ["vae", "ae"]:
        for latent_dim in [32]:
            model, result = train_cifar(latent_dim, model_type=model_type)
            model_dict[(latent_dim, model_type)] = {"model": model, "result": result}

    # image reconstruction
    input_imgs = get_train_images(4)
    for latent_dim, model_type in model_dict:
        visualize_reconstructions(model_dict[(latent_dim, model_type)]["model"], model_type, input_imgs)

    # image generation
    for key in [(32, "ae"), (32, "vae")]:
        model = model_dict[key]["model"]
        latent_vectors = torch.randn(8, model.hparams.latent_dim, device=model.device)
        with torch.no_grad():
            imgs = model.decoder(latent_vectors)
            imgs = imgs.cpu()

        grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, value_range=(-1, 1), pad_value=0.5)
        grid = grid.permute(1, 2, 0)
        plt.figure(figsize=(8, 5))
        plt.imshow(grid)
        plt.axis('off')
        plt.show()