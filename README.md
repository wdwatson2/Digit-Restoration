# Restoring Hand Written Digits 
## Using Residual Networks and Homography Transformations

This is a repository to document my experiments with an image restoration project. 

The motivation is to create model that can generally restore a warped handwritten digit back to some standard. Its similar to an image registration problem, but no reference image is used to align the moving image with. I chose to create the training data through a constrained homography transformation, and then the network is forced to output 9 numbers that correspond to a predicted homography transformation. The network is trained on the MSE between the 9 entries from the actual homography matrix and the 9 entries of the predicted homography matrix. 

## Environment Setup

To ensure compatibility and simplify dependency management, you can use the provided `environment.yml` file to create a Conda environment. This environment setup includes CPU-based PyTorch packages. If you intend to use a CUDA GPU, it's recommended to install packages manually following the instructions [here](https://pytorch.org/get-started/locally/).

Follow these steps to set up the environment:

1. Open a terminal or command prompt and navigate to the directory where the `environment.yml` file is located.
2. Run the following command to create the Conda environment from the YAML file:
`conda env create -f environment.yml`
3. After the environment is created, activate it using the following command:
`conda activate digit-restoration-env`

## Overview

- `flask_app`
  - For a simple flask app to draw your own digits and feed to model. (See directions below to do this)
- `homography`
  - Documentation for how I created the homography model.
- `rotation`
  - A simpler problem but provides further documentation into my methods.

## Running Flask App

1. Open a terminal or command prompt and navigate to the `flask_app` directory.
2. Run the following command to begin the session `python app.py`.
3. The click on one of the URL's to be brought to a webpage.

   
<img src="img/Screenshot%202024-05-01%20091725.png" width="650">
