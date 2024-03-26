# Commonsense Prototype for Outdoor Unsupervised 3D Object Detection

This is the codebase of our CVPR 2024 paper.

## Overview
- [Abstract](#abstract)
- [Environment](#environment)
- [Prepare Dataset](#prepare-daraset)
- [Getting Started](#getting-started)
- [Model Zoo](#model-zoo)
- [Citation](#citation)
  
## Abstract
**CPD** (**C**ommonsense **P**rototype-based **D**etector) is a fully unsupervised three-dimensional object detection framework. We propose a detection method based on CProto, which significantly addresses the errors in pseudo-labels caused by the sparsity of LiDAR data, and outperforms state-of-the-art unsupervised 3D detectors on the Waymo Open Dataset (WOD)PandaSet, and KITTI datasets by a large margin.
![image](https://github.com/hailanyi/CPD/assets/75151571/45d42484-216c-4144-9675-d0886934626d)

## Environment
```bash
conda create --name cpd python=3.8
conda activate cpd
pip install -r requirements.txt

#Install PCD
python setup.py develop
```

## Prepare Dataset
#### Waymo Dataset
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
├── pcdet
├── tools
```
## Getting Started
### Training
Train using scripts
```bash
cd tools
sh dist_train.sh {cfg_file}
```
or run directly
```
cd tools
python train.py 
```

### Evaluation
```bash
Train using scripts
cd tools
sh dist_test.sh {cfg_file}
```
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
        <td><b>DBSCAN</td>
        <td>2.65</td>
        <td>2.29</td>
        <td>0</td>
        <td>0</td>
        <td>0.25</td>
        <td>0.20</td>
        <td><b><a href="https://www.../">dbscan_20epoch</a></td>
    <tr>
       <tr>
        <td><b>CPD</td>
        <td><b>38.74</td>
        <td><b>33.37</td>
        <td><b>16.53</td>
        <td><b>13.72</td>
        <td><b>4.28</td>
        <td><b>4.13</td>
        <td><b><a href="https://drive.google.com/file/d/1_6iFzGfnwGZYD8pqQBRa-Nz1ZLGZ27fZ/view?usp=sharing">cpd_20epoch</a></td>
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

