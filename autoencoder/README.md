# Autoencoder and Variational Autoencoder (VAE)

This repository provides implementations of autoencoders and variational autoencoders (VAEs) using PyTorch. Autoencoders and VAEs are popular deep learning models used for unsupervised learning and generative modeling tasks.

## Introduction

### Autoencoder

An autoencoder is a type of artificial neural network used for unsupervised learning of efficient codings. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction, and then reconstruct the original data from the encoding. It consists of an encoder network that maps the input data to a latent space representation and a decoder network that reconstructs the input data from the latent space representation.

### Variational Autoencoder (VAE)

A variational autoencoder (VAE) is a generative model that combines the framework of autoencoders with variational inference. Unlike traditional autoencoders, VAEs learn a probabilistic distribution over the latent space, allowing for the generation of new data samples. VAEs aim to maximize the likelihood of generating the input data while simultaneously regularizing the latent space to follow a prior distribution, typically a Gaussian distribution.

## Implementation



