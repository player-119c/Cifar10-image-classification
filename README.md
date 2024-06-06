# Cifar10-image-classification
Basic  image classification on 10 classes using can architecture


# CIFAR-10 Image Classification using Convolutional Neural Networks

This repository contains code for training a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Results](#results)

## Introduction

The CIFAR-10 dataset is a collection of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. This project uses a CNN to classify these images into their respective classes.

## Dataset

The dataset consists of the following classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The data is split into 50,000 training images and 10,000 testing images.

## Model Architecture

The CNN model used in this project has the following architecture:
- Convolutional Layer with 32 filters, kernel size 3x3, ReLU activation
- MaxPooling Layer with pool size 2x2
- Convolutional Layer with 64 filters, kernel size 3x3, ReLU activation
- MaxPooling Layer with pool size 2x2
- Flatten Layer
- Dense Layer with 64 units, ReLU activation
- Dense Layer with 10 units, softmax activation

## Installation

To run this project, you need to have Python and the following libraries installed:

- TensorFlow
- NumPy
- Matplotlib

## Results
- As cnn performs exceptionaly well on images after 100 epochs out model reached an accuracy of 92%


