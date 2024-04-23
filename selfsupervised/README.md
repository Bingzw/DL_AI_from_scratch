# Self-Supervised Learning Module

This module implements the SimCLR self-supervised learning algorithm for image data. It includes utilities for contrastive transformations and preparing data features.

## Files

- `simclr.py`: Contains the implementation of the SimCLR model, which is a PyTorch Lightning module. It includes the definition of the model architecture, the optimizer, and the InfoNCE loss function.

- `logisticregression.py`: This file contains the implementation of the Logistic Regression model used for the downstream classification task. The model is implemented as a PyTorch module and includes the definition of the model architecture and the optimizer. It is used to evaluate the quality of the representations learned by the SimCLR model.

- `util.py`: Contains utility functions and classes for the module. This includes the `ContrastiveTransformations` class for applying contrastive transformations to the input data, and functions for preparing data features and getting a smaller dataset.

- `simclr_test.py`: Contains unit tests for the SimCLR model.

- `train.py`: Contains the main script for training the SimCLR model and evaluating the downstream classification task.



## Downstream Classification Task

After training the SimCLR model, we also performed a downstream classification task to evaluate the quality of the learned representations. This task was performed on the STL10 dataset.

We used a simple logistic regression model for the classification task. The features for this model were prepared by passing the STL10 images through the trained SimCLR model.

We also experimented with different sizes of the training set for this task, ranging from 10 to 500 images per label. The results of these experiments are stored in a dictionary and printed at the end of the `train.py` script.

To run the downstream classification task, use the `train_logreg` function in the `train.py` script.

## Run the code
To run the code, simply run the below command in the terminal:
```bash
python train.py
```

## Reference
- [SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
