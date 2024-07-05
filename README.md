# Pneumonia Detection using Convolutional Neural Networks

This repository contains the implementation of multiple convolutional neural network (CNN) models to classify chest X-ray images as either pneumonia or normal. The dataset used is from the Chest X-ray Images (Pneumonia) dataset available on Kaggle.

## Table of Contents

- [Dataset](#dataset)
- [Requirements](#requirements)
- [Model Architectures](#model-architectures)
- [Training and Evaluation](#training-and-evaluation)

## Dataset

The dataset used for training and evaluation is the Chest X-ray Images (Pneumonia) dataset. The dataset consists of 5,856 images divided into training and test sets.

- Training set: 5,232 images
- Test set: 624 images

The dataset is organized into two folders: `train` and `test`, each containing subfolders for pneumonia and normal images.

## Requirements

The following Python packages are required to run the code:

- torch
- torchvision
- matplotlib
- torchinfo
- PIL

You can install the required packages using `pip`:

```sh
pip install torch torchvision matplotlib torchinfo pillow
```

## Model Architectures

Three different model architectures were used in this project:

1. **Custom CNN Models (`pneumoniav0`, `pneumoniav1`, `pneumoniav2`)**
   - These models were built from scratch with varying depths and number of convolutional layers.

2. **Transfer Learning with EfficientNet B0 and B7**
   - Pre-trained EfficientNet models were used with their final layers modified for binary classification.

### `pneumoniav0`

- 4 convolutional layers
- 1 fully connected layer

### `pneumoniav1`

- 5 convolutional layers
- 1 fully connected layer

### `pneumoniav2`

- 6 convolutional layers
- 1 fully connected layer

### EfficientNet B0 and B7

- Pre-trained models with their classifiers modified for binary output.

## Training and Evaluation

### Custom Models

Custom models were trained for 10 epochs each with the following parameters:

- Optimizer: SGD
- Learning Rate: 0.1
- Loss Function: BCEWithLogitsLoss
- Batch Size: 32

### Transfer Learning Models

EfficientNet models were trained for 10 epochs each with the following parameters:

- Optimizer: SGD
- Learning Rate: 0.1
- Loss Function: BCEWithLogitsLoss
- Batch Size: 32

### Results

Results for each model were evaluated on the test set and the following metrics were recorded:

- Train Loss
- Train Accuracy
- Test Loss
- Test Accuracy
