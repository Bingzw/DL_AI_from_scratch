# Vision Transformer (ViT)

This project implements a Vision Transformer (ViT) model in PyTorch. The Vision Transformer is a model that applies the transformer architecture, typically used for natural language processing tasks, to computer vision tasks.

## Introduction

The Vision Transformer (ViT) is a recent advancement in the field of computer vision. Unlike traditional convolutional neural networks (CNNs) that use convolutional layers to process image data, the ViT uses transformer layers, which were originally designed for natural language processing tasks.

The ViT model divides an input image into a sequence of fixed-size patches, linearly embeds each of them, adds positional embeddings, and feeds the resulting sequence of vectors to a standard Transformer encoder. In the output layer, a classification head is attached to the representation corresponding to the first input token (the class token), as in BERT.

The ViT model in this project is implemented in the `vitnet.py` file. It includes the `VisionTransformer` class, which is the main model, and the `ViT` class, which is a PyTorch Lightning module that includes the training, validation, and testing steps.


## Usage

The main model is implemented in the `vitnet.py` file. You can import the `VisionTransformer` class from this file and use it in your own projects.

Here's a basic example of how to use the `VisionTransformer`:

```python
from vit.vitnet import VisionTransformer
import torch

# Define the parameters
embed_dim = 64
hidden_dim = 128
num_channels = 3
num_heads = 4
num_layers = 2
num_classes = 10
patch_size = 4
num_patches = 49
dropout = 0.1

# Create a random input tensor
x = torch.randn(1, num_channels, 28, 28)

# Instantiate the VisionTransformer
model = VisionTransformer(embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size,
                          num_patches, dropout)

# Pass the input tensor through the model
out = model(x)
```

## Reference
Sure, here are some references about the Vision Transformer (ViT):
1. [Google AI Blog: Transformers for Image Recognition at Scale](https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html): This blog post from Google AI provides an overview of the research behind Vision Transformers and their application to image recognition tasks.
