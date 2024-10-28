[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/target-before-shooting-accurate-anomaly/anomaly-detection-on-mvtec-ad)](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad?p=target-before-shooting-accurate-anomaly)
# CPR
Official PyTorch implementation of [CPR](https://paperswithcode.com/paper/target-before-shooting-accurate-anomaly)

## Datasets

We use the [MVTec AD](https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz) dataset for experiments. And use [DTD](https://thor.robots.ox.ac.uk/datasets/dtd/dtd-r1.0.1.tar.gz) data set to simulate anomalous image.

The data directory is as follows:
```
data
├── dtd
│   ├── images
│   ├── imdb
│   └── labels
└── mvtec
    ├── bottle
    │   ├── ground_truth
    │   ├── license.txt
    │   ├── readme.txt
    │   ├── test
    │   └── train
    ...
    └── zipper
        ├── ground_truth
        ├── license.txt
        ├── readme.txt
        ├── test
        └── train
```

## Installation

> [pytorch](https://pytorch.org/)

`conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`

`pip install -r requirements.txt`

## generate foreground and global retrieval result

```
python tools/generate_foreground.py
python tools/generate_retrieval.py
```

## Training

**generate synthetic data**

`python tools/generate_synthetic_data.py -fd log/foreground_mvtec_DenseNet_features.denseblock1_320`

`bash train.sh`

## Testing

`python test.py -fd log/foreground_mvtec_DenseNet_features.denseblock1_320/ --checkpoints weights/{category}.pth`

## Pretrained Checkpoints

Download pretrained checkpoints [here](https://github.com/flyinghu123/CPR/releases) and put the checkpoints under <project_dir>/weights/.

Baidu Netdisk: https://pan.baidu.com/s/1FTE4b2G8nVZt4lUyaP-kIQ?pwd=ky7j

## Acknowledgement
We borrow some codes from [PatchCore](https://github.com/amazon-science/patchcore-inspection), [MemSeg](https://github.com/TooTouch/MemSeg) and [SuperPoint](https://github.com/eric-yyjau/pytorch-superpoint)

## Citation
```
@ARTICLE{10678861,
  author={Li, Hanxi and Hu, Jianfei and Li, Bo and Chen, Hao and Zheng, Yongbin and Shen, Chunhua},
  journal={IEEE Transactions on Image Processing}, 
  title={Target Before Shooting: Accurate Anomaly Detection and Localization Under One Millisecond via Cascade Patch Retrieval}, 
  year={2024},
  volume={33},
  number={},
  pages={5606-5621},
  keywords={Accuracy;Anomaly detection;Measurement;Standards;Prototypes;Image retrieval;Image reconstruction;Anomaly detection;image patch retrieval;metric learning},
  doi={10.1109/TIP.2024.3448263}
}

```
