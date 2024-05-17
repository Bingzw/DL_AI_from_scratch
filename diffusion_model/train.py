import random
import imageio
import numpy as np
from argparse import ArgumentParser

from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import einops
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets.mnist import MNIST, FashionMNIST

from ddpm import DDPM
from denoise_network import UNet

from util import set_seed, show_images, show_first_batch, show_forward, generate_new_images


def training(ddpm, loader, n_epochs, optim, device, display=False, save_path="ddpm_model.pt"):
    mse = nn.MSELoss()
    best_loss = float('inf')
    n_steps = ddpm.n_steps

    for epoch in tqdm(range(n_epochs)):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, leave=False)):
            x0 = batch[0].to(device)
            n = len(x0)
            # generate some random noise for each image in the batch, timestamp and eta
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)
            # forward pass: generate the nosiy version of the input images
            noisy_imgs = ddpm(x0, t, eta)
            # backward pass: estimate the noised added to the image
            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))
            # optimization
            loss = mse(eta, eta_theta)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * len(x0) / len(loader.dataset)

        # Display images generated at this epoch
        if display:
            show_images(generate_new_images(ddpm, device=device), f"Images generated at epoch {epoch + 1}")

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"
        # save the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), save_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)


if __name__ == "__main__":
    SEED = 42
    path = os.getcwd()
    parent_dir = os.path.dirname(path)
    # set seed
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # download dataset
    DATASET_PATH = "../data/"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x - 0.5) * 2)]
    )
    # hyperparameters
    batch_size = 128
    n_epochs = 20
    lr = 0.001

    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = FashionMNIST(root=DATASET_PATH, train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    # Define model
    n_steps = 1000
    min_beta = 10**-4
    max_beta = 0.02
    ddpm = DDPM(UNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)
    # show_forward(ddpm, train_loader, device)
    
    # training
    model_save_path = os.path.join(parent_dir, 'saved_models/diffusion/ddpm_model.pt')
    optimizer = Adam(ddpm.parameters(), lr=lr)
    training(ddpm, train_loader, n_epochs, optimizer, device, display=False, save_path=model_save_path)

    # load model
    best_model = DDPM(UNet(n_steps), n_steps=n_steps, device=device)
    best_model.load_state_dict(torch.load(model_save_path, map_location=device))
    best_model.eval()
    print("generating new images")
    gif_path = os.path.join(parent_dir, 'saved_models/diffusion/fashion.gif')
    generated_imgs = generate_new_images(
        best_model,
        n_samples=100,
        device=device,
        gif_name=gif_path
    )
    show_images(generated_imgs, "Final result")
    img = Image.open(gif_path)
    img.show()




