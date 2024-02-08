# image-caption-generator

## Overview

This project implements an image caption generator using a ResNet backbone as an encoder and a transformer decoder. The model is trained on the Flicker8k dataset, allowing it to generate captions for images.

## Components

### 1. Encoder (ResNet Backbone)

The encoder utilizes a ResNet architecture to extract features from input images. These features are then passed to the decoder for caption generation.

### 2. Decoder (Transformer)

The decoder employs a transformer architecture to generate captions based on the features extracted by the encoder. The transformer decoder takes as input the image features and produces textual descriptions.

### 3. Modules

The download_data.py downloads the dataset.
The dataset.py defines the Dataset Pytorch Dataset class.
The model.py defines the model.
The train.py script can be executed from the terminal to train the image caption generator model. It loads the Flicker8k dataset, prepares the data for training, initializes the model, trains the model using the specified hyperparameters, and saves the trained model weights.
The inference.py script contains the inference function for generating captions from input images. Given an image as input, it utilizes the trained model to generate a textual description of the image.

## Requirements

Python 3.x
PyTorch
Other required libraries (specified in requirements.txt)

## Getting Started

- Clone the repository.
- Install the required dependencies using `pip install -r requirements.txt.`
- Download the Flicker8k dataset using the download_data.py. Can be run from terminal using `python download_data.py`
- Train the model using train.py. Can be run from terminal using `python train.py`
- Import infer.py to generate captions for new images.

## Additional info

Hyperparameters are `img_size`, `cnn_backbone`, `seq_len`, `d_model`, `n_decode`, `n_head`, `fc_dim`,`dropout`. 
Can be changed in train.py
Can be trained on colab. Check out train_image_capgen.ipynb file.
