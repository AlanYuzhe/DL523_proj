# Human Face Inpainting:

This project focuses on working with a dataset of images, applying masks for inpainting tasks, and leveraging pretrained encoder and decoder models for image processing. The goal is to provide an easy-to-follow setup for researchers or developers interested in experimenting with image manipulation and enhancement techniques.

## Getting Started:

These instructions will guide you through setting up the project environment and downloading the necessary datasets and pretrained models.

### Prerequisites:

Before you begin, ensure you have Python installed on your machine. Additionally, you may need libraries such as NumPy, Pandas, and PyTorch, depending on your processing and model training needs.

### Installation:

#### Step 1: Download the Image Dataset

Download the CelebA image dataset by visiting the following link:

- [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

#### Step 2: Download the Mask Dataset:

For the mask dataset, essential for the inpainting tasks, use the link below:

- [Partial Convolution Mask Dataset](https://nv-adlr.github.io/publication/partialconv-inpainting)

#### Step 3: Download the Pretrained Encoder and Decoder:

Access pretrained models for your encoder and decoder through this Google Drive link:

- [Pretrained Models](https://drive.google.com/file/d/1nPVW9eMBTHQ4SYuYPfIRgRSFAvjKnmVP/view?usp=sharing)

### Setup:

After downloading the datasets and models, create folders to store your training logs and models by running the following commands in your terminal:

mkdir model

mkdir log

### Train CNN model:
You can run the cnninpainting.py file directly to start the training of the cnn model, remember to check that the address of the image file in it corresponds to your address.

### Train PD-GAN model:
You can run the train.py file directly to start the training of the PD-GAN model, remember to check that the address of the image file in it corresponds to your address.

### Test Models:
First model with implemented PD-GAN
We provide you with the trained model (https://drive.google.com/file/d/1vDjVcrQ9Pn1J5tVCbUozw7iNfsxQz4sK/view?usp=sharing), which you can unzip into the model folder and run the test.py file to test the performance of the model (please set your own folder directory in test.py)

Second model with CNN network
different models in https://drive.google.com/file/d/1ups1x25f6pYj04bowq6GcA631WS_0cci/view?usp=sharing,
You can run the image_inpainting.py file directly after resetting the test image directory, and it will generate the results for you directly.
### Introduction of all files:
1.SPDNorm.py: Created a normalization method specifically for dealing with symmetric positive definite matrices;

2.blocks: Realized image restoration and neural network construction

3.cnninpainting.py: Built the cnn model, trained the model and saved it locally

4.data.py: Pre-processed image data, customized loading, pre-processing and enhancement of image data

5.diffusion.py: Created diffusion model, trained the model and saved it locally

6.image_inpainting.py:test then model trained by cnninpainting.py

7.pd_gan.py: This code defines a generator with a discriminator for PD-GAN 

8.pretrained_cnn.py and pretrained_loss.py: These two pieces of code provide a pre-trained cnn model and a pre-trained loss function for image preprocessing, respectively.

9.psnr.py:This code calculates the PSNR of the repaired image and the original image and is used to compare the quality of the repair

10.test.py:This code imports the trained model and fixes the masked image

11.train.py: This code imports the dataset and initiates the training of the PD_GAN model.
### Acknowledgement:
We reuse the following codebases:
The code and model of Pretrained Encoder-Decoder for building PD-GAN are adapted from the following sources:
The code for data preprocessing:
https://github.com/RenYurui/StructureFlow/blob/master/src/data.py

The code and model of Pretrained Encoder-Decoder: https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/net.py
https://github.com/KumapowerLIU/PD-GAN/blob/main/models/network/pconv.py

The code for loss function using pretrained VGG-16: https://github.com/KumapowerLIU/Rethinking-Inpainting-MEDFE/blob/master/models/loss.py
The code for multi-scale discriminator: https://github.com/yuan-yin/UNISST
