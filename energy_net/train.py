import random
import os
import matplotlib.pyplot as plt
# PyTorch
import torch
import torch.nn as nn
import torch.utils.data as data
# Torchvision
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from energy_net.energynet import DeepEnergyModel
from data_sampler import Sampler
from util import set_seed


class GenerateCallback(pl.Callback):  # used to track image generation process and visualize it in tensorboard
    def __init__(self, batch_size=8, vis_steps=8, num_steps=256, every_n_epochs=5):
        super().__init__()
        self.batch_size = batch_size         # Number of images to generate
        self.vis_steps = vis_steps           # Number of steps within generation to visualize
        self.num_steps = num_steps           # Number of steps to take during generation
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_epoch_end(self, trainer, pl_module):
        # Skip for all other epochs
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Generate images
            imgs_per_step = self.generate_imgs(pl_module)
            # Plot and add to tensorboard
            for i in range(imgs_per_step.shape[1]):
                step_size = self.num_steps // self.vis_steps
                imgs_to_plot = imgs_per_step[step_size-1::step_size,i]
                grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True, range=(-1,1))
                trainer.logger.experiment.add_image(f"generation_{i}", grid, global_step=trainer.current_epoch)

    def generate_imgs(self, pl_module):
        pl_module.eval()
        start_imgs = torch.rand((self.batch_size,) + pl_module.hparams["img_shape"]).to(pl_module.device)
        start_imgs = start_imgs * 2 - 1
        torch.set_grad_enabled(True)  # Tracking gradients for sampling necessary
        imgs_per_step = Sampler.generate_samples(pl_module.cnn, start_imgs, steps=self.num_steps, step_size=10, return_img_per_step=True)
        torch.set_grad_enabled(False)
        pl_module.train()
        return imgs_per_step


class SamplerCallback(pl.Callback):
    def __init__(self, num_imgs=32, every_n_epochs=5):
        super().__init__()
        self.num_imgs = num_imgs             # Number of images to plot
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            exmp_imgs = torch.cat(random.choices(pl_module.sampler.examples, k=self.num_imgs), dim=0)
            grid = torchvision.utils.make_grid(exmp_imgs, nrow=4, normalize=True, range=(-1,1))
            trainer.logger.experiment.add_image("sampler", grid, global_step=trainer.current_epoch)


class OutlierCallback(pl.Callback):
    def __init__(self, batch_size=1024):
        super().__init__()
        self.batch_size = batch_size

    def on_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            pl_module.eval()
            rand_imgs = torch.rand((self.batch_size,) + pl_module.hparams["img_shape"]).to(pl_module.device)
            rand_imgs = rand_imgs * 2 - 1.0
            rand_out = pl_module.cnn(rand_imgs).mean()
            pl_module.train()

        trainer.logger.experiment.add_scalar("rand_out", rand_out, global_step=trainer.current_epoch)


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
    # Transformations applied on each image => make them a tensor and normalize between -1 and 1
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])

    # Loading the training dataset. We need to split it into a training and validation part
    train_set = MNIST(root=DATASET_PATH, train=True, transform=transform, download=True)
    test_set = MNIST(root=DATASET_PATH, train=False, transform=transform, download=True)
    train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, num_workers=4,
                                   pin_memory=True)
    test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)

    img_shape = (1, 28, 28)
    batch_size = train_loader.batch_size
    lr = 1e-4
    beta1 = 0.0
    # train model
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "MNIST"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=10,
                         gradient_clip_val=0.1,
                         callbacks=[
                             ModelCheckpoint(save_weights_only=True, mode="min", monitor='val_contrastive_divergence'),
                             GenerateCallback(every_n_epochs=5),
                             SamplerCallback(every_n_epochs=5),
                             OutlierCallback(),
                             LearningRateMonitor("epoch")
                            ]
                         )

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "MNIST.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = DeepEnergyModel.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)
        model = DeepEnergyModel(img_shape=img_shape, batch_size=batch_size, lr=lr, beta1=beta1, device=device)
        trainer.fit(model, train_loader, test_loader)
        model = DeepEnergyModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # image generation
    model.to(device)
    pl.seed_everything(43)
    callback = GenerateCallback(batch_size=4, vis_steps=8, num_steps=256)
    imgs_per_step = callback.generate_imgs(model)
    imgs_per_step = imgs_per_step.cpu()

    for i in range(imgs_per_step.shape[1]):
        step_size = callback.num_steps // callback.vis_steps
        imgs_to_plot = imgs_per_step[step_size - 1::step_size, i]
        imgs_to_plot = torch.cat([imgs_per_step[0:1, i], imgs_to_plot], dim=0)
        grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True, value_range=(-1, 1),
                                           pad_value=0.5, padding=2)
        grid = grid.permute(1, 2, 0)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid)
        plt.xlabel("Generation iteration")
        plt.xticks([(imgs_per_step.shape[-1] + 2) * (0.5 + j) for j in range(callback.vis_steps + 1)],
                   labels=[1] + list(range(step_size, imgs_per_step.shape[0] + 1, step_size)))
        plt.yticks([])
        plt.show()


