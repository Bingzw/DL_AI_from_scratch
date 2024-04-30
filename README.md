# DL_genAI_practicing

This repo is for practicing deep learning and generative AI. It basically covers the popular deep learning models in CV, NLP, genAI etc.
The goal of this repo is to provide a comprehensive and practical understanding of deep learning and generative AI. The repo is structured in such a way that it covers the following topics:

- Prediction models: used for regression, classification etc.
  - googlenet
  - resnet
  - densenet
  - recommendation models
    - dlrm
    - deepfm
    - mmoe
    - bst
- Generative models: used to generate new data
  - Autogressive models
    - pixelCNN
  - VAE
  - energy based model
  - normalizing flow
- Discovery models: used to learn the hidden state of the data for downstream task
  - AE
  - self-supervised learning:
    - contrastive learning: SimCLR
  - ViT

The above list did not include all revolutionary works, like GAN, diffusion model are not included. I am on my way of hands on practicing and would keep adding them later.

All models are coded using pytorch and lighting, we also tried tuning hyperparameters using ray tune. Users who have interest may refer to the README.md in each model folder for more details.