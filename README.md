RUMpy (**R**estoration toolbox developed by the **U**niversity of **M**alta)  
=============
[![Platform Support](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-blue)](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-blue) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

About 
-------------------
RUMpy is a PyTorch-based toolbox for blind image super-resolution (SR), with a variety of SR deep learning architectures and degradation predictors available for use.  In particular, RUMpy provides:  

- An easy-to-use CLI for:
  - Generating low-resolution (LR) images from a high-resolution (HR) dataset, with various types of blurring, noise addition and compression available.
  - Training or fine-tuning of SR models and degradation predictors.
  - Qualitative and quantitative evaluation of SR results.
  - Tools for analyzing, moving and curating models.
- Various SR and degradation prediction architectures, with customizable settings.
- Straightforward pipeline for developing and integrating new models in the framework.
- A GUI for quick evaluation of models, including the cropping and direct SR of video frames.
- Integration with [Aim](https://github.com/aimhubio/aim) (Mac & Linux only) for training monitoring.

![Quick overview of how the GUI application works](Documentation/GUI_quick_look.gif)

RUMpy has been used to test and combine a variety of blind degradation prediction systems with high-performing SR architectures.  Our results are available in our 2022 Sensors journal paper [here](https://doi.org/10.3390/s23010419).  The basic concept of the 'Best of Both Worlds' framework is illustrated below:

![bobw_framework.png](Documentation%2Fbobw_framework.png)

A snapshot of our blind SR results on a real-world image are available below (more details in the full paper):

![lincoln_sr.png](Documentation%2Flincoln_sr.png)

This also acts as the main code repository for the Deep-FIR [project](https://www.um.edu.mt/projects/deep-fir/) (University of Malta).

Developers/Researchers:

- [Matthew Aquilina](https://www.linkedin.com/in/matthewaq/)
- [Keith George Ciantar](https://www.linkedin.com/in/keith-george-ciantar/)
- [Christian Galea](https://cnpgs.wordpress.com)

Project Management:

- [John Abela](https://www.um.edu.mt/profile/johnabela)
- [Kenneth Camilleri](https://www.um.edu.mt/profile/kennethcamilleri)
- [Reuben Farrugia](https://www.um.edu.mt/profile/reubenfarrugia)

Installation
--------------------
### Python and Virtual Environments

If installing from scratch, it is first recommended to set up a new Python virtual environment prior to installing this code.  With [Conda](https://docs.conda.io/en/latest/), this can be achieved through the following:

```conda create -n *environment_name* python=3.7 ```  (Python 3.7-3.8 recommended but not essential).

```conda activate *environment_name*```

Code testing was conducted in Python 3.7 & Python 3.8, but the code should work well with Python 3.6+.

### Local Installation

Run the following commands from the repo base directory to fully install the package and all requirements:

1. **If using CPU only:**

   Install main requirements via:

   ```conda install --file requirements.txt --channel pytorch --channel conda-forge```

   **If using CPU + GPU:** 

   First install Pytorch and Cudatoolkit for your specific configuration using instructions [here](https://pytorch.org/get-started/locally/).  Then, install requirements as above.
2. Install pip packages via: ```pip install -r pip_requirements.txt```
3. **If using [Aim](https://github.com/aimhubio/aim)** for metrics logging, install via ```pip install aim```.  The Aim GUI does not work on Windows, but metrics should still be logged in the .aim folder.
4.    ```pip install -e .```  This installs the toolbox, but will also auto-update if any changes to the code are made (this is ideal for those seeking to make their own custom changes to the code).

All functionality has been tested on Linux (CPU & GPU), Mac OS (CPU) and Windows (CPU & GPU). 

Requirements installation is only meant as a guide and all requirements can be installed using alternative means (e.g. using ```pip``` for all packages).

GUI Installation and Usage
------------------------------

Details provided in `GUI/README.md`.

Guidelines for Generating SR Data
-----------------
All details on generating LR data are provided in `Documentation/data_prep.md`.

###  DIV2K/Flickr2K Datasets
DIV2K training/validation downloadable from [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/).  
Flickr2K dataset downloadable from [here](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar).

### SR testing Datasets
All SR testing datasets are available for download from the LapSRN main page [here](http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_testing_datasets.zip).  Generate LR versions of each image using the same commands as used for the DIV2K/Flickr2K datasets. 

### CelebA Datasets (for our 2021 Signal Processing Letters paper)

Please refer to details [here](https://github.com/um-dsrg/Super-Resolution-Meta-Attention-Networks).  

Training/Evaluating Models
----------------
### Training
To train models, prepare a configuration file (details in `Documentation/model_training.md`) and run:

```train_sisr --parameters *path_to_config_file*```
### Evaluation
Similarly, for evaluation, prepare an eval config file (details in `Documentation/sisr_model_eval.md`) and run:

```eval_sisr --config *path_to_config_file*```

### Contrastive Model Evaluation
Additional functionality for evaluating contrastive models is discussed in `Documentation/contrastive_model_eval.md`.

### Standard SISR models available (code for each adapted from their official repository - linked within source code):
1. [SRCNN](https://arxiv.org/abs/1501.00092)
2. [VDSR](https://arxiv.org/abs/1511.04587)
3. [EDSR](https://arxiv.org/abs/1707.02921)
4. [RCAN](https://arxiv.org/abs/1807.02758)
5. [ESRGAN](https://arxiv.org/abs/1809.00219) - 4x only
6. [Real-ESRGAN](https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Wang_Real-ESRGAN_Training_Real-World_Blind_Super-Resolution_With_Pure_Synthetic_Data_ICCVW_2021_paper.pdf) - 4x only
6. [Wavelet-SRNet](https://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Wavelet-SRNet_A_Wavelet-Based_ICCV_2017_paper.pdf)
7. [Wavelet-SRGAN](https://link.springer.com/article/10.1007/s11263-019-01154-8) (WIP)
8. [SPARNet](https://arxiv.org/abs/2012.01211)
9. [DICNET](https://openaccess.thecvf.com/content_CVPR_2020/html/Ma_Deep_Face_Super-Resolution_With_Iterative_Collaboration_Between_Attentive_Recovery_and_CVPR_2020_paper.html) (not fully validated) - 4x only
10. [SFTMD](https://arxiv.org/abs/1904.03377)
11. [SRMD](https://arxiv.org/abs/1712.06116)
12. [SAN](https://openaccess.thecvf.com/content_CVPR_2019/html/Dai_Second-Order_Attention_Network_for_Single_Image_Super-Resolution_CVPR_2019_paper.html)
13. [HAN](https://arxiv.org/abs/2008.08767)
14. [ELAN](https://arxiv.org/abs/2203.06697)

### Standard Blind SISR models available:

1. [IKC](https://arxiv.org/abs/1904.03377)
2. [DAN v1 & v2](https://github.com/greatlog/DAN)
3. [DASR](https://arxiv.org/abs/2104.00416)

### Degradation Predictors available:

1. DAN (v1)
2. MoCo (algorithm from DAN)
3. SupMoCo (algorithm from [here](https://arxiv.org/abs/2101.11058))
4. WeakCon (algorithm from [here](https://doi.org/10.1016/j.knosys.2022.108984))
   
### SR/Degradation Predictors integrated within our framework:
1. RCAN (DAN & Contrastive Encoders)
2. HAN (DAN & Contrastive Encoders)
3. ELAN (DAN & Contrastive Encoders)
4. Real-ESRGAN (DAN & Contrastive Encoders)
5. SAN (Contrastive Encoders)
6. EDSR (Contrastive Encoders)

### Pre-Trained Model Weights
- IEEE Signal Processing Letters models (baseline models + meta-attention models): [link](https://doi.org/10.5281/zenodo.5551061)
- Sensors 2022 models (all Best of Both Worlds models): [link](https://doi.org/10.5281/zenodo.7488458)

Once downloaded, models from the above links can be used directly with the eval command (```eval_sisr``)  or with the GUI.


Additional/Advanced Setup
--------------
### Setting up VGGFace for face recognition metrics (Tensorflow)
Install the required packages using ```pip install -r special_requirements.txt```.  Model weights are automatically installed to ```~./keras``` when first used.

### Setting up VGGFace (Pytorch)
Download pre-trained weights for the VGGFace model from [here](http://www.robots.ox.ac.uk/~albanie/pytorch-models.html) (scroll to VGGFace).  Place the weights file in the directory ```./external_packages/VGGFace/```.  The weights file should be called ```vgg_face_dag.pth```.

### Setting up lightCNN
Download pre-trained weights for the lightCNN model from [here](https://github.com/AlfredXiangWu/LightCNN) (LightCNN-29 v1).  Place the weights file in the directory ```./external_packages/LightCNN/```.  The weights file should be called ```LightCNN_29Layers_checkpoint.pth.tar```.

### Setting up the YOLO face detector
Download pre-trained weights for the YOLO model from [here](https://github.com/sthanhng/yoloface).  Place the weights file in the directory ```./Code/utils/yolo_detection```.  The weights file should be called ```yolov3-wider_16000.weights```.

### Setting up the Bisenet segmentation network
Download pre-trained weights for the Bisenet model trained on Celeba-HQ from [here](https://github.com/zllrunning/face-parsing.PyTorch).  Place the weights file in the directory ```./Code/utils/face_segmentation```.  The weights file should be called ```weights.pth```.

### <a name="jm-install"></a> Setting up JM (for compressing images)
Download the reference software from [here](http://iphome.hhi.de/suehring/tml/).  Place the software in the directory ```./JM```.  cd into this directory and compile the software using the commands ```. unixprep.sh``` and ```make```.  Some changes might be required for different OS versions.   

### Setting up the Face Aligner
Install the [face alignment package](https://github.com/1adrianb/face-alignment) using ```conda install -c 1adrianb face_alignment```.

> ℹ️ **NOTE:**  Currently, this package doesn't get installed properly if using Python 3.8/CUDA 10.0.

> ℹ️ **NOTE:**  Landmarks generated by this method vary slightly if using older versions of the package.

Creating Custom Models
------------------------
Information on how to develop and train your own models is available in `Documentation/framework_development.md`.

Full List of Commands Available
-------------------------------
The entire list of commands available with this repository is:

- `train_sisr` - main model training function.
- `eval_sisr` - main model evaluation function.
- `image_manipulate` - main bulk image converter.
- `find_faces` - Helper function for using YOLO face detector to detect faces in an input image directory.
- `face_segment` - Helper function to segment face images and save output map for downstream use.
- `images_to_video` - Helper function to convert a folder of images into a video.
- `extract_best_model` - Helper function to extract model config and best model checkpoint from a folder to a target location.
- `clean_models` - Helper function to remove unnecessary model checkpoints.
- `model_report` - Helper function to report on models available in specified directory.

Each command can be run with the `--help` parameter, which will print out the available options and docstrings.

Uninstall
-----------------
Simply run:

```pip uninstall rumpy```

from any directory, with the relevant virtual environment activated.

Citation
-----------------
The main paper to cite for this repository is our 2023 [paper](https://doi.org/10.3390/s23010419):
```bibtex
@Article{RUMpy,
    AUTHOR = {Aquilina, Matthew and Ciantar, Keith George and Galea, Christian and Camilleri, Kenneth P. and Farrugia, Reuben A. and Abela, John},
    TITLE = {The Best of Both Worlds: A Framework for Combining Degradation Prediction with High Performance Super-Resolution Networks},
    JOURNAL = {Sensors},
    VOLUME = {23},
    YEAR = {2023},
    NUMBER = {1},
    ARTICLE-NUMBER = {419},
    URL = {https://www.mdpi.com/1424-8220/23/1/419},
    ISSN = {1424-8220},
    ABSTRACT = {To date, the best-performing blind super-resolution (SR) techniques follow one of two paradigms: (A) train standard SR networks on synthetic low-resolution&ndash;high-resolution (LR&ndash;HR) pairs or (B) predict the degradations of an LR image and then use these to inform a customised SR network. Despite significant progress, subscribers to the former miss out on useful degradation information and followers of the latter rely on weaker SR networks, which are significantly outperformed by the latest architectural advancements. In this work, we present a framework for combining any blind SR prediction mechanism with any deep SR network. We show that a single lightweight metadata insertion block together with a degradation prediction mechanism can allow non-blind SR architectures to rival or outperform state-of-the-art dedicated blind SR networks. We implement various contrastive and iterative degradation prediction schemes and show they are readily compatible with high-performance SR networks such as RCAN and HAN within our framework. Furthermore, we demonstrate our framework&rsquo;s robustness by successfully performing blind SR on images degraded with blurring, noise and compression. This represents the first explicit combined blind prediction and SR of images degraded with such a complex pipeline, acting as a baseline for further advancements.},
    DOI = {10.3390/s23010419}
}
```
An earlier version of this framework has also been used for our 2021 Signal Processing Letters [paper](https://doi.org/10.1109/LSP.2021.3116518) introducing meta-attention.  A checkpoint containing this earlier version is available [here](https://github.com/um-dsrg/Super-Resolution-Meta-Attention-Networks), with the associated paper available to cite as follows:
```bibtex
@ARTICLE{Meta-Attention,
  author={Aquilina, Matthew and Galea, Christian and Abela, John and Camilleri, Kenneth P. and Farrugia, Reuben A.},
  journal={IEEE Signal Processing Letters}, 
  title={Improving Super-Resolution Performance Using Meta-Attention Layers}, 
  year={2021},
  volume={28},
  number={},
  pages={2082-2086},
  doi={10.1109/LSP.2021.3116518}}
```
License
--------------------------
This code has been released via the GNU GPLv3 open-source license.  However, this code can also be made available via an alternative closed, permissive license.  Third-parties interested in this form of licensing should contact us separately.  

Usages of code from other repositories is properly referenced within the code itself and the licenses of these repositories are available under Documentation/external_licenses.  
