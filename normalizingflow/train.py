import torch
import torch.nn as nn
import random
import numpy as np
import os
import time
import matplotlib.pyplot as plt
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
# Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from normalizing_flow import CouplingLayer, GatedConvNet, create_checkerboard_mask, Dequantization, \
    VariationalDequantization, SqueezeFlow, SplitFlow, create_channel_mask
from autoencoder.vaenet import VAE
from util import set_seed, discretize, show_imgs

from torchvision import transforms
from torchvision.datasets import MNIST


class ImageFlow(pl.LightningModule):
    def __init__(self, flows, import_samples=8):
        """
        Inputs:
            flows - A list of flows (each a nn.Module) that should be applied on the images.
            import_samples - Number of importance samples to use during testing (see explanation below). Can be changed at any time
        """
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.import_samples = import_samples
        # create prior distribution for latent space
        self.prior = torch.distributions.normal.Normal(0.0, 1.0)
        # example input for visualizing the graph
        self.example_input_array = train_set[0][0].unsqueeze(dim=0)

    def forward(self, imgs):
        return self._get_likelihood(imgs)

    def encode(self, imgs):
        # given a batch of images, return the latent representation z and ldj(log determinant Jacobian) of the
        # transformation
        z, ldj = imgs, torch.zeros(imgs.shape[0], device=self.device)
        for flow in self.flows:
            z, ldj = flow(z, ldj, reverse=False)
        return z, ldj

    def _get_likelihood(self, imgs, return_ll=False):
        """
        Given a batch of images, return the likelihood of those.
        If return_ll is True, this function returns the log likelihood of the input.
        Otherwise, the ouptut metric is bits per dimension (scaled negative log likelihood)
        """
        z, ldj = self.encode(imgs)
        log_pz = self.prior.log_prob(z).sum(dim=[1,2,3])
        log_px = ldj + log_pz
        nll = -log_px
        # calculate bits per dimension
        bpd = nll * np.log2(np.exp(1)) / np.prod(imgs.shape[1:])
        return bpd.mean() if not return_ll else log_px

    @torch.no_grad()
    def sample(self, img_shape, z_init=None):
        """
        Sample a batch of images from the flow
        """
        if z_init is None:
            z = self.prior.sample(sample_shape=img_shape).to(self.device)
        else:
            z = z_init.to(self.device)

        # Transform z to x by inverting the flows
        ldj = torch.zeros(img_shape[0], device=self.device)
        for flow in reversed(self.flows):
            z, ldj = flow(z, ldj, reverse=True)
        return z

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # An scheduler is optional, but can help in flows to get the last bpd improvement
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # Normalizing flows are trained by maximum likelihood => return bpd
        loss = self._get_likelihood(batch[0])
        self.log('train_bpd', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_likelihood(batch[0])
        self.log('val_bpd', loss)

    def test_step(self, batch, batch_idx):
        # Perform importance sampling during testing => estimate likelihood M times for each image
        samples = []
        for _ in range(self.import_samples):
            img_ll = self._get_likelihood(batch[0], return_ll=True)
            samples.append(img_ll)
        img_ll = torch.stack(samples, dim=-1)

        # To average the probabilities, we need to go from log-space to exp, and back to log.
        # Logsumexp provides us a stable implementation for this
        img_ll = torch.logsumexp(img_ll, dim=-1) - np.log(self.import_samples)

        # Calculate final bpd
        bpd = -img_ll * np.log2(np.exp(1)) / np.prod(batch[0].shape[1:])
        bpd = bpd.mean()

        self.log('test_bpd', bpd)


def create_simple_flow(use_vardeq=True):
    flow_layers = []
    if use_vardeq:
        vardeq_layers = [CouplingLayer(network=GatedConvNet(c_in=2, c_out=2, c_hidden=16),
                                       mask=create_checkerboard_mask(h=28, w=28, invert=(i % 2 == 1)),
                                       c_in=1) for i in range(4)]
        flow_layers += [VariationalDequantization(var_flows=vardeq_layers)]
    else:
        flow_layers += [Dequantization()]

    for i in range(8):
        flow_layers += [CouplingLayer(network=GatedConvNet(c_in=1, c_hidden=32),
                                      mask=create_checkerboard_mask(h=28, w=28, invert=(i % 2 == 1)),
                                      c_in=1)]

    flow_model = ImageFlow(flow_layers).to(device)
    return flow_model


def create_multiscale_flow():
    flow_layers = []

    vardeq_layers = [CouplingLayer(network=GatedConvNet(c_in=2, c_out=2, c_hidden=16),
                                   mask=create_checkerboard_mask(h=28, w=28, invert=(i % 2 == 1)),
                                   c_in=1) for i in range(4)]
    flow_layers += [VariationalDequantization(vardeq_layers)]

    flow_layers += [CouplingLayer(network=GatedConvNet(c_in=1, c_hidden=32),
                                  mask=create_checkerboard_mask(h=28, w=28, invert=(i % 2 == 1)),
                                  c_in=1) for i in range(2)]
    flow_layers += [SqueezeFlow()]
    for i in range(2):
        flow_layers += [CouplingLayer(network=GatedConvNet(c_in=4, c_hidden=48),
                                      mask=create_channel_mask(c_in=4, invert=(i % 2 == 1)),
                                      c_in=4)]
    flow_layers += [SplitFlow(),
                    SqueezeFlow()]
    for i in range(4):
        flow_layers += [CouplingLayer(network=GatedConvNet(c_in=8, c_hidden=64),
                                      mask=create_channel_mask(c_in=8, invert=(i % 2 == 1)),
                                      c_in=8)]

    flow_model = ImageFlow(flow_layers).to(device)
    return flow_model


def train_flow(flow, model_name="MNISTFlow", pretrained_model_path=".ckpt"):
    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, model_name),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=100,
                         gradient_clip_val=1.0,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_bpd"),
                                    LearningRateMonitor("epoch")],
                         check_val_every_n_epoch=5)
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    train_data_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True,
                                        num_workers=8)
    result = None

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, model_name + pretrained_model_path)
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        ckpt = torch.load(pretrained_filename, map_location=device)
        flow.load_state_dict(ckpt['state_dict'])
        result = ckpt.get("result", None)
    else:
        print("Start training", model_name)
        trainer.fit(flow, train_data_loader, val_loader)

    # Test best model on validation and test set if no result has been found
    # Testing can be expensive due to the importance sampling.
    if result is None:
        val_result = trainer.test(flow, val_loader, verbose=False)
        start_time = time.time()
        test_result = trainer.test(flow, test_loader, verbose=False)
        duration = time.time() - start_time
        result = {"test": test_result, "val": val_result, "time": duration / len(test_loader) / flow.import_samples}

    return flow, result


if __name__ == "__main__":
    SEED = 42
    CHECKPOINT_PATH = "../saved_models/normalizing_flow"
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
    # Transformations applied on each image => make them a tensor and discretize
    transform = transforms.Compose([transforms.ToTensor(),
                                    discretize])

    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = MNIST(root=DATASET_PATH, train=True, transform=transform, download=True)
    pl.seed_everything(42)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])

    # Loading the test set
    test_set = MNIST(root=DATASET_PATH, train=False, transform=transform, download=True)

    # We define a set of data loaders that we can use for various purposes later.
    # Note that for actually training a model, we will use different data loaders
    # with a lower batch size.
    train_loader = data.DataLoader(train_set, batch_size=256, shuffle=False, drop_last=False)
    val_loader = data.DataLoader(val_set, batch_size=64, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=64, shuffle=False, drop_last=False, num_workers=4)

    flow_dict = {"simple": {}, "vardeq": {}, "multiscale": {}}
    flow_dict["simple"]["model"], flow_dict["simple"]["result"] = train_flow(create_simple_flow(use_vardeq=False),
                                                                             model_name="MNISTFlow_simple",
                                                                             pretrained_model_path=pretrained_model_name)
    flow_dict["vardeq"]["model"], flow_dict["vardeq"]["result"] = train_flow(create_simple_flow(use_vardeq=True),
                                                                             model_name="MNISTFlow_vardeq",
                                                                             pretrained_model_path=pretrained_model_name)
    flow_dict["multiscale"]["model"], flow_dict["multiscale"]["result"] = train_flow(create_multiscale_flow(),
                                                                                     model_name="MNISTFlow_multiscale",
                                                                                     pretrained_model_path=pretrained_model_name)

    # sample from the flow
    pl.seed_everything(44)
    samples = flow_dict["vardeq"]["model"].sample(img_shape=[16, 1, 28, 28])
    show_imgs(samples.cpu())

    pl.seed_everything(42)
    samples = flow_dict["multiscale"]["model"].sample(img_shape=[16, 8, 7, 7])
    show_imgs(samples.cpu())