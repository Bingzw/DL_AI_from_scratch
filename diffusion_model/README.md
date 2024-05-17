# Diffusion Model


## Introduction

### Denoising Diffusion Probabilistic Models

A Diffusion Model is a type of probabilistic model that simulates the process of diffusion. It is often used in the field of machine learning and artificial intelligence, particularly in the area of generative modeling. 

The core idea behind a Diffusion Model is to represent data as the endpoint of a diffusion process. This process starts from a simple prior distribution and gradually transforms it into the target data distribution through a series of small diffusion steps. 

In the context of machine learning, these models are used to generate new data samples that are similar to the ones in the training set. They have been successfully applied to various types of data, including images and text.

The Denoising Diffusion Probabilistic Models, as mentioned here, are a specific type of Diffusion Models. They introduce a reverse process to the diffusion, which is a denoising operation. This operation aims to recover the original data from the diffused data, and it is where the learning happens. The model learns to denoise the data in each diffusion step, and in doing so, it learns about the data distribution.


## Implementation
- ddpm.py: Contains the structure of the Denoising Diffusion Probabilistic Model (DDPM) using PyTorch.
- denoise_network.py: Contains the structure of the denoising network used in the DDPM.

## Usage

To train the models, run the following command:

```bash
cd diffusion_model
python train.py
```

## Reference:
- [Diffusion Models](https://arxiv.org/abs/2006.11239)
- [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Generating images with DDPMs: A PyTorch Implementation](https://medium.com/@brianpulfer/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1)