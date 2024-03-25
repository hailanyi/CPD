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


## Citation
```
@inproceedings{CPD,
    title={Commonsense Prototype for Outdoor Unsupervised 3D Object Detection},
    author={Wu, Hai and Zhao, Shijia and Huang, Xun and Wen, Chenglu and Li, Xin and Wang, Cheng},
    booktitle={CVPR},
    year={2024}
}
```

