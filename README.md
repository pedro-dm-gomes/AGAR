# AGAR

Project under submission

[[Project]](https://github.com/pedro-dm-gomes/AGAR) [[Paper]](https://arxiv.org/abs/2307.09936)     



### Overview
AGAR: Attention Graph-RNN for Adaptative Motion Prediction of Point Clouds of Deformable Objects


<img src="https://github.com/pedro-dm-gomes/AGAR/blob/main/Figures/teaser_figure.png" scale="0.6">


This github proposes an improved architecture for point
cloud prediction of deformable 3D objects, able to control the composition of local and global motions
for each point, enabling the network to model complex motions in deformable 3D objects more effectively.

### Setup

Install <a href="https://www.tensorflow.org/get_started/os_setup" target="_blank">TensorFlow</a>. The code has been tested with Python 3.6, TensorFlow 1.15.0, CUDA 9.0 and cuDNN 7.21

Compile the code. You will need to select the correct CUDA version and Tensorflow instaled in your computer. For that edit the Makefiles to the paths of your Cuda and Tensorflow directories.
The Makefiles to compile the code are in `modules/tf_ops`


## Datasets

## Usage
To train a model using AGAR on the Mixamo Dataset

    python train-mixamo.py

To evaluate the model

    python eval-mixamo.py




Code is currently being uploaded
