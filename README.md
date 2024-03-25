# Commonsense Prototype for Outdoor Unsupervised 3D Object Detection

This is the codebase of our CVPR 2024 paper.

## Abstract
CPD (Commonsense Prototype for Outdoor Unsupervised 3D Object Detection) is a fully unsupervised three-dimensional object detection framework. We propose a detection method based on CProto, which significantly addresses the errors in pseudo-labels caused by the sparsity of LiDAR data, and outperforms state-of-the-art unsupervised 3D detectors on the Waymo Open Dataset (WOD)PandaSet, and KITTI datasets by a large margin.
![image](https://github.com/hailanyi/CPD/assets/75151571/45d42484-216c-4144-9675-d0886934626d)

## Environment
```bash
conda create --name cpd python=3.8
conda activate cpd
pip install -r requirements.txt

#Install PCD
python setup.py develop
```

## Training
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

## Evaluation
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
        <td><b><a href="https://www.../">20epoch</a></td>
    <tr>
       <tr>
        <td><b>CPD</td>
        <td><b>37.40</td>
        <td><b>32.13</td>
        <td><b>16.31</td>
        <td><b>13.22</td>
        <td><b>5.06</td>
        <td><b>4.87</td>
        <td><b><a href="https://www.../">20epoch</a></td>
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

