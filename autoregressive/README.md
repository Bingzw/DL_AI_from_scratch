# Auto-Regressive Model with PixelCNN

This project is an implementation of an auto-regressive model using PixelCNN.

## Overview

Auto-regressive models are a type of statistical model used for predicting future values based on previous ones. They are widely used in time series forecasting and other types of sequential data.

In this project, we use PixelCNN, a type of auto-regressive model, for generating images. PixelCNN models the joint distribution of pixels in an image and generates pixels one at a time, conditioning on the previously generated pixels.

## Implementation

The implementation is done in Python using PyTorch, a popular deep learning library. The main model file is `pixelcnn.py`, which contains the implementation of the PixelCNN model.

## Usage

To run the model, use the following command:

```bash
python train.py
```

## Testing

Unit tests for the PixelCNN model are provided in `pixelcnn_test.py`. To run the tests, use the following command:

```bash
python -m unittest pixelcnn_test.py
```
