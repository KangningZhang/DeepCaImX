# DeepCaImX
## Introduction
#### Two-photon calcium imaging provides large-scale recordings of neuronal activities at cellular resolution. A robust, automated and high-speed pipeline to simultaneously segment the spatial footprints of neurons and extract their temporal activity traces while decontaminating them from background, noise and overlapping neurons is highly desirable to analyze calcium imaging data. In this paper, we demonstrate DeepCaImX, an end-to-end deep learning method based on an iterative shrinkage-thresholding algorithm and a long-short-term-memory neural network to achieve the above goals altogether at a very high speed and without any manually tuned hyper-parameters. DeepCaImX is a multi-task, multi-class and multi-label segmentation method composed of a compressed-sensing-inspired neural network with a recurrent layer and fully connected layers. It represents the first neural network that can simultaneously generate accurate neuronal footprints and extract clean neuronal activity traces from calcium imaging data. We trained the neural network with simulated datasets and benchmarked it against existing state-of-the-art methods with in vivo experimental data. DeepCaImX outperforms existing methods in the quality of segmentation and temporal trace extraction as well as processing speed. DeepCaImX is highly scalable and will benefit the analysis of mesoscale calcium imaging. 
![alt text](https://github.com/KangningZhang/DeepCaImX/blob/main/imgs/Fig1.png)

## System and Environment Requirements
#### 1. Both CPU and GPU are supported to run the code of DeepCaImX. A CUDA compatible GPU is preferred. 
* In our demo of full-version, we use a GPU of Quadro RTX8000 48GB to accelerate the training speed.
* In our demo of mini-version, at least 6 GB momory of GPU/CPU is required.
#### 2. Python 3.9 and Tensorflow 2.10.0
#### 3. Virtual environment: Anaconda Navigator 2.2.0
#### 4. Matlab 2023a

## Demo and installation
#### 1 (_Optional_) GPU environment setup. We need a Nvidia parallel computing platform and programming model called _CUDA Toolkit_ and a GPU-accelerated library of primitives for deep neural networks called _CUDA Deep Neural Network library (cuDNN)_ to build up a GPU supported environment for training and testing our model. The link of CUDA installation guide is https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html and the link of cuDNN installation guide is https://docs.nvidia.com/deeplearning/cudnn/installation/overview.html. 
#### 2 Install Anaconda. Link of installation guide: https://docs.anaconda.com/free/anaconda/install/index.html
#### 3 Launch Anaconda prompt and install Python 3.x and Tensorflow 2.9.0 as the virtual environment.
#### 4 Open the virtual environment, and then  pip install mat73, opencv-python, python-time and scipy.
#### 5 Download the "DeepCaImX_training_demo.ipynb" in folder "Demo (full-version)" for a full version and the simulated dataset via the google drive link. Then, create and put the training dataset in the path "./Training Dataset/". If there is a limitation on your computing resource or a quick test on our code, we highly recommand download the demo from the folder "Mini-version", which only requires around 6.3 GB momory in training. 
#### 6 Run: Use Anaconda to launch the virtual environment and open "DeepCaImX_training_demo.ipynb" or "DeepCaImX_testing_demo.ipynb". Then, please check and follow the guide of "DeepCaImX_training_demo.ipynb" or or "DeepCaImX_testing_demo.ipynb" for training or testing respectively.
#### Note: Every package can be installed in a few minutes.

## Run DeepCaImX
#### 1. Mini-version demo
* Download all the documents in the folder of "Demo (mini-version)".
* Adding training and testing dataset in the sub-folder of "Training Dataset" and "Testing Dataset" separately.
* (Optional) Put pretrained model in the the sub-folder of "Pretrained Model"
* Using Anaconda Navigator to launch the virtual environment and opening "DeepCaImX_training_demo.ipynb" for training or "DeepCaImX_testing_demo.ipynb" for predicting.

#### 2. Full-version demo
* Download all the documents in the folder of "Demo (full-version)".
* Adding training and testing dataset in the sub-folder of "Training Dataset" and "Testing Dataset" separately.
* (Optional) Put pretrained model in the the sub-folder of "Pretrained Model"
* Using Anaconda Navigator to launch the virtual environment and opening "DeepCaImX_training_demo.ipynb" for training or "DeepCaImX_testing_demo.ipynb" for predicting.

## Data Tailor
#### A data tailor developed by Matlab is provided to support a basic data tiling processing. In the folder of "Data Tailor", we can find a "tailor.m" script and an example "test.tiff". After running "tailor.m" by matlab, user is able to choose a "tiff" file from a GUI as loading the sample to be tiled. Settings include size of FOV, overlapping area, normalization option, name of output file and output data format. The output files can be found at local folder, which is at the same folder as the "tailor.m".

## Simulated Dataset
#### 1. Dataset generator (FISSA Version): The algorithm for generating simulated dataset is based on the paper of FISSA (_Keemink, S.W., Lowe, S.C., Pakan, J.M.P. et al. FISSA: A neuropil decontamination toolbox for calcium imaging signals. Sci Rep 8, 3493 (2018)_) and SimCalc repository (https://github.com/rochefort-lab/SimCalc/). For the code used to generate the simulated data, please download the documents in the folder "Simulated Dataset Generator". 
#### Training dataset: https://drive.google.com/file/d/1WZkIE_WA7Qw133t2KtqTESDmxMwsEkjJ/view?usp=share_link
#### Testing Dataset: https://drive.google.com/file/d/1zsLH8OQ4kTV7LaqQfbPDuMDuWBcHGWcO/view?usp=share_link

#### 2. Dataset generator (NAOMi Version): The algorithm for generating simulated dataset is based on the paper of NAOMi (_Song, A., Gauthier, J. L., Pillow, J. W., Tank, D. W. & Charles, A. S. Neural anatomy and optical microscopy (NAOMi) simulation for evaluating calcium imaging methods. Journal of neuroscience methods 358, 109173 (2021)_). For the code use to generate the simulated data, please go to this link: https://bitbucket.org/adamshch/naomi_sim/src/master/code/
## Experimental Dataset
#### We used the samples from ABO dataset:
https://github.com/AllenInstitute/AllenSDK/wiki/Use-the-Allen-Brain-Observatory-%E2%80%93-Visual-Coding-on-AWS.
#### The segmentation ground truth can be found in the folder "Manually Labelled ROIs". 
#### The segmentation ground truth of depth 175, 275, 375, 550 and 625 um are manually labeled by us. 
#### The code for creating ground truth of extracted traces can be found in "Prepro_Exp_Sample.ipynb" in the folder "Preprocessing of Experimental Sample".
