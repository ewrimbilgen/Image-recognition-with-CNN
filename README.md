# Chasing Harry Potter: LEGO Minifigure Classification with CNN

**Author:** Evrim Bilgen

## About

This project builds a Convolutional Neural Network (CNN) to classify 30 Harry Potter LEGO minifigure characters from images. Two model architectures were compared: EfficientNetB6 and DenseNet121, both using transfer learning from ImageNet weights.

## Problem

Given an image of a Harry Potter LEGO minifigure, predict which character it is across 30 classes.

## Dataset

- Harry Potter LEGO minifigure image dataset
- 30 character classes
- Train/validation split provided via metadata
- Images resized to 224x224 or 256x256 depending on model

## Approach

### Data Preparation
- Loaded images using OpenCV, converted BGR to RGB
- Normalized pixel values to [0, 1]
- Merged class metadata to attach character names to image paths

### Data Augmentation
Applied using both `albumentations` and Keras `ImageDataGenerator`:
- Random rotation, horizontal/vertical flip
- Brightness and contrast adjustment
- Zoom, width/height shift
- Cutout regularization

### Models

**Model 1: DenseNet121**
- Pretrained on ImageNet, top layer replaced
- Added Dropout (0.5) before final Dense layer
- Optimizer: Adam (lr=0.0001)
- Loss: sparse categorical crossentropy
- Epochs: 50, batch size: 4

**Model 2: EfficientNetB6**
- Pretrained on ImageNet, global average pooling
- Optimizer: SGD
- Loss: categorical crossentropy
- Epochs: 25, batch size: 15

## Results

Training accuracy improved significantly across both models. However, validation loss increased over time while training loss decreased, indicating **overfitting** — a known limitation given the relatively small dataset size per class.

## Key Findings

- Transfer learning from ImageNet provides a strong starting point even for domain-specific tasks like LEGO classification
- Data augmentation helps but is not sufficient to fully overcome overfitting on small datasets
- More training images per class or a lighter architecture would likely improve generalization

## Tools & Libraries

- Python, Jupyter Notebook
- TensorFlow / Keras
- EfficientNet (tfkeras)
- OpenCV, albumentations
- scikit-learn, pandas, numpy
- matplotlib, seaborn
