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
**CPD** (**C**ommonsense **P**rototype-based **D**etector)  represents a high-performance unsupervised 3D detection framework. CPD first constructs Commonsense Prototype (CProto) characterized by high-quality bounding box and dense points, based on commonsense intuition. Subsequently, CPD refines the low-quality pseudo-labels by leveraging the size prior from CProto. Furthermore, CPD enhances the detection accuracy of sparsely scanned objects by the geometric knowledge from CProto, and outperforms state-of-the-art unsupervised 3D detectors on the Waymo Open Dataset (WOD)PandaSet, and KITTI datasets by a large margin.
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
├── pcdet
├── tools
```
Install the official `waymo-open-dataset` by running the following command: 
```
# tf 2.0.0
pip install waymo-open-dataset-tf-2-5-0 --user
```
Then, generate dataset information:
```
cd pcdet/datasets/waymo_unsupervised
python waymo_unsupervised_dataset.py --cfg_file{...}
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
        <td><b><a href="https://www.../">---</a></td>
     <tr>
       <tr>
        <td><b>OYSTER</td>
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
        <td><b>4.49</td>
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

