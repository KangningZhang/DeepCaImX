# DeepCaImX
## Introduction
#### Two-photon calcium imaging provides large-scale recordings of neuronal activities at cellular resolution. A robust, automated and high-speed pipeline to simultaneously segment the spatial footprints of neurons and extract their temporal activity traces while decontaminating them from background, noise and overlapping neurons is highly desirable to analyze calcium imaging data. In this paper, we demonstrate DeepCaImX, an end-to-end deep learning method based on an iterative shrinkage-thresholding algorithm and a long-short-term-memory neural network to achieve the above goals altogether at a very high speed and without any manually tuned hyper-parameters. DeepCaImX is a multi-task, multi-class and multi-label segmentation method composed of a compressed-sensing-inspired neural network with a recurrent layer and fully connected layers. It represents the first neural network that can simultaneously generate accurate neuronal footprints and extract clean neuronal activity traces from calcium imaging data. We trained the neural network with simulated datasets and benchmarked it against existing state-of-the-art methods with in vivo experimental data. DeepCaImX outperforms existing methods in the quality of segmentation and temporal trace extraction as well as processing speed. DeepCaImX is highly scalable and will benefit the analysis of mesoscale calcium imaging. 
![alt text](https://github.com/KangningZhang/DeepCaImX/blob/main/imgs/Fig1.png)
#### Please feel free to contact us if you have any concerns (knzhang@ucdavis.edu or knzhang@stanford.edu)
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

## Training dataset preparing
#### We highly recommend to use NAOMi method for generating simulated training and testing dataset. Please check the code in the folder "Simulated Dataset Generator/NAOMi". We high recommand you create your own dataset according to your experiment settings, because the specific DeepCaImX model owns a critical requirement on the types of dataset. The principle of data with ground truth generation are as follows:
#### 1. Settings of calcium imaging depth in NAOMi code is very important. Please follow your experiment situation. Besides, the signal-to-background ratio is also important. Please feel free to tune the factor of the spike intensity in the NAOMi code to match your experiment situations.
#### 2. In our project, the generation of denoise, background-suppressed data should merely keep the traces activity of soma in the temporal information (i.e, no background and dentrite activity included), then run the scanning section in the NAOMi code. This output will give you the ground truth of denoised, background-suppressed data.
#### 3. In this paper, the input of DeepCaImX (which is the raw recordings) should be normalized to 0~1, the denoised BG-suppressed ground truth should be normalized based on the factors of the normalization from the raw recordings.
#### 4. The order of the ROIs segmentation ground truth should critically obey the order when each indivial neuron first appears, since DeepCaImX is a multi-class segmentation.
#### 5. The ground truth of the neuron activity traces should not contain any baseline, and merely represent the activity of soma.
#### Note: more operations for generator code can be found in the section of Dataset generator (NAOMi Version).

### Notice: The extraction of the ground truth ROIs and activity traces from NAOMi training dataset could be obtained from the code in "Simulated Data Generator/NAOMi", but we highly encourage you to train DeepCaImX based on your only experiment settings or objects, even if your experiment are quit different from the principles of NAOMi. To create an innovative, inclusive, and instructive environment, we also plan to release an unsupervised learning approach attached on the framework of DeepCaImX to obtain the output of ROIs or traces without training on ROIs or traces ground truth in the case that you hardly obtain these ground truths from your experiment on Oct 15. We will keep our toolboxs maintained and updated. Please stay tuned!

## Run DeepCaImX
#### 1. Mini-version demo
* Download all the documents in the folder of "Demo (mini-version)".
* Adding training and testing dataset in the sub-folder of "Training Dataset" and "Testing Dataset" separately.
* (Optional) Put pretrained model in the the sub-folder of "Pretrained Model"
* Using Anaconda Navigator to launch the virtual environment and opening "DeepCaImX_training_demo.ipynb" for training or "DeepCaImX_testing_demo.ipynb" for predicting.
* Note: the pre-trained model and testing are based on simulated data via FISSA method.

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

#### How to run our modified code
#### (1) Run ‘Naomi_275mm_demo/TPM_Simulation_Script.m’. It’s used to generate raw video
#### (2) Run ‘Naomi_275mm_demo/ generate_soma_video.m’. It’s used to generate corresponding videos with soma only. For each neuron trace, no baseline is included.
#### (3) Run ‘generate_training_pairs.m’. It’s used to resize each raw video and soma video to be 64x64x400. Then generate the corresponding mask ground truth for each training pairs.
#### (4) Run ‘generate_mask_clips_training_pairs.m’. Generate the trace ground truth by multiplying mask with soma video.
#### (5) Run ‘sort_mask_trace2.m’. Sorting the mask and trace based on which one comes the first spike. By the end of this step, you will get the final mask and trace files used for training.
#### (6) Run ‘generate_video_clips_mat’. Save the raw video and it’s corresponding denoised ground truth. By the end of this step, you will get the final raw video and clean video files used for training.

#### Training dataset: https://drive.google.com/drive/folders/1ANOOVpyqoAQrv0FFetnZB2PDwlDM0C7l?usp=sharing
#### Testing dataset: https://drive.google.com/drive/folders/1O6dGCelX0lti1ohvwf3xYpGOeSJkf4KH?usp=sharing

## Experimental Dataset
#### We used the samples from ABO dataset:
https://github.com/AllenInstitute/AllenSDK/wiki/Use-the-Allen-Brain-Observatory-%E2%80%93-Visual-Coding-on-AWS.
#### The segmentation ground truth can be found in the folder "Manually Labelled ROIs". 
#### The segmentation ground truth of depth 175, 275, 375, 550 and 625 um are manually labeled by us. 
#### The code for creating ground truth of extracted traces can be found in "Prepro_Exp_Sample.ipynb" in the folder "Preprocessing of Experimental Sample".


## Acknowledge
#### Please receive my GREATEST RESPECT for Yifei's cleaning up job and Dr. Yang's supervision work. Please also let me express my appreciated to everyone who share me your precious comments on our work.
