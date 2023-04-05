# LSTM-ISTA-for-Calcium-Imaging-Data
## Introduction
#### Two-photon calcium imaging provides large-scale recordings of neuronal activities in cellular resolution. A robust, automated and high speed pipeline to segment the spatial footprints of neurons, extract the temporal activity traces and decontaminate them from background, noise and overlapping neurons is highly desirable to analyze the calcium imaging data. In this paper, we demonstrate Long-Short-Term-Memory based Iterative Shrinkage-Thresholding Algorithm (LSTM-ISTA), an end-to-end deep learning method to achieve the above goals altogether in a very high speed and without any manual-tuned hyper-parameter. LSTM-ISTA is a multi-task, multi-class and multi-label segmentation method, composed of a compressed-sensing-inspired neural network with a recurrent layer and fully connected layers. It represents the first neural network that can simultaneously generate accurate neuronal footprints and extract clean neuronal activity traces from calcium imaging data. We trained the neural network through simulation dataset, and benchmarked it against existing state-of-the-art methods through in-vivo experimental data. LSTM-ISTA outperforms existing methods in the quality of segmentation and temporal traces extraction, and processing speed. LSTM-ISTA is highly scalable and will benefit the analysis of mesoscale calcium imaging. 
![alt text](https://github.com/KangningZhang/LSTM-ISTA-for-Calcium-Imaging-Data/blob/main/Figures/Fig1.png)

## System Requirements
#### 1. Quadro RTX8000 48GB. A CUDA compatible GPU is preferred.
#### 2. Tensorflow 2.9.0
#### 3. Anaconda

## Demo
#### Please check and follow the guide of LSTM-ISTANet.ipynb in folder Demo

## Simulated Dataset
#### Training dataset: https://drive.google.com/file/d/1WZkIE_WA7Qw133t2KtqTESDmxMwsEkjJ/view?usp=share_link
#### Testing Dataset: https://drive.google.com/file/d/1zsLH8OQ4kTV7LaqQfbPDuMDuWBcHGWcO/view?usp=share_link
## Experimental Dataset
#### We used the samples from ABO dataset from
https://github.com/AllenInstitute/AllenSDK/wiki/Use-the-Allen-Brain-Observatory-%E2%80%93-Visual-Coding-on-AWS.
#### The segmentation ground truth of layers 375, 550, 625 are manually labeled by us. The segmentation ground truth of layers 175 and 275 are based on https://github.com/soltanianzadeh/STNeuroNet/tree/master/Markings/ABO.
