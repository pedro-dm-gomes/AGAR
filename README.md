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

For more details on how to compile these we refer the reader to the original PointNet++ [2] paper.


## Datasets

The models were evaluated with the following datasets:
1. [Mixamo - Human Bodies](https://drive.google.com/drive/folders/14SRpLT0X7yQPKJV7YDiZXEGJVnw1SkHN?usp=sharing) &emsp;

More information about the dataset (e.g, how to create/modify) can be find in the following github [https://github.com/pedro-dm-gomes/Human_Bodies_Dataset]



## Usage
To train a model using AGAR on the Mixamo Dataset

    python train-mixamo.py  --version <model_version>

To evaluate the model

    python eval-mixamo.py  --version <model_version> 


Code is currently being uploaded

## Citation
Please cite this paper if you want to use it in your work,

	@article{gomes2023agar,
	title={AGAR: Attention Graph-RNN for Adaptative Motion Prediction of Point Clouds of Deformable Objects},
	  author={Pedro Gomes and Silvia Rossi and Laura Toni},
	  year={2023},
	  journal={arXiv preprint arXiv:2307.09936},
	  }




## Acknowledgement
The parts of this codebase is borrowed from Related Repos:

### Related Repos
1. PointRNN TensorFlow implementation: https://github.com/hehefan/PointRNN
2. PointNet++ TensorFlow implementation: https://github.com/charlesq34/pointnet2
3. Dynamic Graph CNN for Learning on Point Clouds https://github.com/WangYueFt/dgcnn
4. Temporal Interpolation of Dynamic Point Clouds using Convolutional Neural Networks https://github.com/jelmr/pc_temporal_interpolation
5. Spatio-temporal Graph-RNN for Point Cloud Prediction https://github.com/pedro-dm-gomes/Graph-RNN 


