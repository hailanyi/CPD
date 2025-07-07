# Commonsense Prototype for Outdoor Unsupervised 3D Object Detection (CVPR 2024)

This is the codebase of our [CVPR 2024 paper](http://arxiv.org/abs/2404.16493).

## Overview
- [Abstract](#abstract)
- [Environment](#environment)
- [Prepare Dataset](#prepare-daraset)
- [Getting Started](#getting-started)
- [Model Zoo](#model-zoo)
- [Citation](#citation)
  
## Abstract
**CPD** (**C**ommonsense **P**rototype-based **D**etector)  is a high-performance unsupervised 3D object detection framework. CPD first constructs Commonsense Prototype (CProto) characterized by high-quality bounding box and dense points, based on commonsense intuition. Subsequently, CPD refines the low-quality pseudo-labels by leveraging the size prior from CProto. Furthermore, CPD enhances the detection accuracy of sparsely scanned objects by the geometric knowledge from CProto. CPD outperforms state-of-the-art unsupervised 3D detectors on the Waymo Open Dataset (WOD), and KITTI datasets by a large margin.
![image](https://github.com/hailanyi/CPD/assets/75151571/45d42484-216c-4144-9675-d0886934626d)

## Environment
```bash
conda create -n spconv2 python=3.9
conda activate spconv2
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy==1.19.5 protobuf==3.19.4 scikit-image==0.19.2 waymo-open-dataset-tf-2-5-0 nuscenes-devkit==1.0.5 spconv-cu111 numba scipy pyyaml easydict fire tqdm shapely matplotlib opencv-python addict pyquaternion awscli open3d pandas future pybind11 tensorboardX tensorboard Cython prefetch-generator
```

Environment we tested:
```
Ubuntu 18.04
Python 3.9.13
PyTorch 1.8.1
Numba 0.53.1
Spconv 2.1.22 # pip install spconv-cu111
NVIDIA CUDA 11.1
4x 3090 GPUs
```

## Prepare Dataset
#### Waymo Dataset
* Please download the official [Waymo Open Dataset](https://waymo.com/open/download/), 
    including the training data `training_0000.tar~training_0031.tar` and the validation 
    data `validation_0000.tar~validation_0007.tar`.
* Unzip all the above `xxxx.tar` files to the directory of `data/waymo/raw_data` as follows (You could get 798 *train* tfrecord and 202 *val* tfrecord ):  
```
CPD
├── data
│   ├── waymo
│   │   │── ImageSets
│   │   │── raw_data
│   │   │   │── segment-xxxxxxxx.tfrecord
|   |   |   |── ...
|   |   |── waymo_processed_data_train_val_test
│   │   │   │── segment-xxxxxxxx/
|   |   |   |── ...
│   │   │── pcdet_waymo_track_dbinfos_train_cp.pkl
│   │   │── waymo_infos_test.pkl
│   │   │── waymo_infos_train.pkl
│   │   │── waymo_infos_val.pkl
├── cpd
├── tools
```

Then, generate dataset information:
```
python3 -m cpd.datasets.waymo_unsupervised.waymo_unsupervised_dataset --cfg_file tools/cfgs/dataset_configs/waymo_unsupervised/waymo_unsupervised_cproto.yaml
```

#### KITTI Dataset

* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):

```
CasA
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── cpd
├── tools
```

Run following command to create dataset infos:
```
python3 -m cpd.datasets.kitti.kitti2waymo_dataset create_kitti_infos tools/cfgs/dataset_configs/waymo_unsupervised/kitti2waymo_dataset.yaml
```

### Training
Train using scripts
```bash
cd tools
sh dist_train.sh {cfg_file}
```
The log infos are saved into log.txt. You can run cat log.txt to view the test results.

or run directly
```
cd tools
python train.py 
```

### Evaluation
```bash
cd tools
sh dist_test.sh {cfg_file}
```
The log infos are saved into log-test.txt You can run cat log-test.txt to view the test results.



## Model Zoo
<table text-align="center">
    <tr>
        <td rowspan="2"><b>Model</td>
        <td colspan="2"><b>Vehicle 3D AP</td>
        <td colspan="2"><b>Pedestrian 3D AP</td>
        <td colspan="2"><b>Cyclist 3D AP</td>
        <td rowspan="2"><b>Download</td>
        <tr>
         	<td><b>L1</td>
         	<td><b>L2</td>
         	<td><b>L1</td>
          	<td><b>L2</td>
            <td><b>L1</td>
         	<td><b>L2</td>
     	<tr>
    <tr>
    <tr>
        <td><b>DBSCAN-single-train</td>
        <td>2.65</td>
        <td>2.29</td>
        <td>0</td>
        <td>0</td>
        <td>0.25</td>
        <td>0.20</td>
        <td><b><a href="https://www.../">---</a></td>
     <tr>
       <tr>
        <td><b>OYSTER-single-train</td>
        <td><b>7.91</td>
        <td><b>6.78</td>
        <td><b>0.03</td>
        <td><b>0.02</td>
        <td><b>4.65</td>
        <td><b>4.05</td>
        <td><b><a href="https://drive.google.com/file/d/1rHySNcUnwXkMNNRgf2D2GQ-hPnaa7ejz/view?usp=sharing">oyster_pretrained</a></td>
    <tr>
    <tr>
       <tr>
        <td><b>CPD</td>
        <td><b>38.74</td>
        <td><b>33.37</td>
        <td><b>16.53</td>
        <td><b>13.72</td>
        <td><b>4.28</td>
        <td><b>4.13</td>
        <td><b><a href="https://drive.google.com/file/d/1_6iFzGfnwGZYD8pqQBRa-Nz1ZLGZ27fZ/view?usp=sharing">cpd_pretrained</a></td>
    <tr>
</table>

The thresholds for evaluating these three categories are respectively set to $IoU_{0.7}$, $IoU_{0.5}$, and $IoU_{0.5}$.

## Citation
```
@inproceedings{CPD,
    title={Commonsense Prototype for Outdoor Unsupervised 3D Object Detection},
    author={Wu, Hai and Zhao, Shijia and Huang, Xun and Wen, Chenglu and Li, Xin and Wang, Cheng},
    booktitle={CVPR},
    year={2024}
}
```

