# Autoencoder and Variational Autoencoder (VAE)

This repository provides implementations of autoencoders and variational autoencoders (VAEs) using PyTorch. Autoencoders and VAEs are popular deep learning models used for unsupervised learning and generative modeling tasks.

## Introduction

### Autoencoder

An autoencoder is a type of artificial neural network used for unsupervised learning of efficient codings. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction, and then reconstruct the original data from the encoding. It consists of an encoder network that maps the input data to a latent space representation and a decoder network that reconstructs the input data from the latent space representation.

### Variational Autoencoder (VAE)

A variational autoencoder (VAE) is a generative model that combines the framework of autoencoders with variational inference. Unlike traditional autoencoders, VAEs learn a probabilistic distribution over the latent space, allowing for the generation of new data samples. VAEs aim to maximize the likelihood of generating the input data while simultaneously regularizing the latent space to follow a prior distribution, typically a Gaussian distribution.

## Implementation

The autoencoder and VAE are implemented in Python using the PyTorch and PyTorch Lightning libraries. The autoencoder consists of an `Encoder` and `Decoder` class, while the VAE extends the autoencoder with a `VAE_Encoder` class and a modified `Autoencoder` class that includes a reparameterization trick and a KL divergence term in the loss function.

The models can be trained and tested using the provided scripts. The training script includes a unit test to ensure the output of the `forward` method has the same shape as the input.

## Usage

To train the models, run the following command:

```bash
python train.py
```

## Reference:

- [VAE_WIKI](https://en.wikipedia.org/wiki/Variational_autoencoder)
- [AE & VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)




