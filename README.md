# Human Face Impainting

This project focuses on working with a dataset of images, applying masks for inpainting tasks, and leveraging pretrained encoder and decoder models for image processing. The goal is to provide an easy-to-follow setup for researchers or developers interested in experimenting with image manipulation and enhancement techniques.

## Getting Started

These instructions will guide you through setting up the project environment and downloading the necessary datasets and pretrained models.

### Prerequisites

Before you begin, ensure you have Python installed on your machine. Additionally, you may need libraries such as NumPy, Pandas, and PyTorch, depending on your processing and model training needs.

### Installation

#### Step 1: Download the Image Dataset

Download the CelebA image dataset by visiting the following link:

- [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

#### Step 2: Download the Mask Dataset

For the mask dataset, essential for the inpainting tasks, use the link below:

- [Partial Convolution Mask Dataset](https://nv-adlr.github.io/publication/partialconv-inpainting)

#### Step 3: Download the Pretrained Encoder and Decoder

Access pretrained models for your encoder and decoder through this Google Drive link:

- [Pretrained Models](https://drive.google.com/drive/folders/1o9reT5_lFzGKBsrLlvck545nNInIAlPe)

### Setup

After downloading the datasets and models, create folders to store your training logs and models by running the following commands in your terminal:

mkdir model

mkdir log

