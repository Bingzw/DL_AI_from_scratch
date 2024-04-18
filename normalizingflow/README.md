# Normalizing Flow Implementation

This repository provides an implementation of Normalizing Flows using PyTorch and PyTorch Lightning. Normalizing Flows are a class of models that provide a flexible approach for probabilistic modeling.

## Introduction

Normalizing Flows are a type of generative model that allows for complex, learnable transformations of simple random variables to model complex data distributions. They are particularly useful in unsupervised learning and generative modeling tasks.

The key idea behind Normalizing Flows is to start with a simple distribution (like a multivariate Gaussian) and apply a series of invertible transformations (the "flows") to transform it into a more complex distribution that can model the data more accurately.

## Implementation

The Normalizing Flow model is implemented in Python using the PyTorch and PyTorch Lightning libraries. The model consists of a series of transformations that are applied to a base distribution to form a more complex distribution. The transformations are parameterized by neural networks, allowing the model to learn complex data distributions.

The model can be trained and tested using the provided scripts. The training script includes a unit test to ensure the output of the `forward` method has the same shape as the input.

## Usage

To train the model, run the following command:

```bash
python train.py
```

## References

- [Normalizing Flows: An Introduction and Review](https://arxiv.org/abs/1908.09257)
- [Normalizing Flows for Probabilistic Modeling and Inference](https://arxiv.org/abs/1912.02762)
- [Flow Models](https://lilianweng.github.io/posts/2018-10-13-flow-models/)

