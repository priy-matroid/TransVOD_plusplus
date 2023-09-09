# TransVOD++
**by [Qianyu Zhou](https://qianyuzqy.github.io/), [Xiangtai Li](https://lxtgh.github.io/), [Lu He](https://github.com/SJTU-LuHe)**, [Yibo Yang](), [Guangliang Cheng](), [Yunhai Tong](), [Lizhuang Ma](https://dmcv.sjtu.edu.cn/people/), [Dacheng Tao]()

**[[Arxiv]](https://arxiv.org/pdf/2201.05047.pdf)**
**[[Paper]](https://ieeexplore.ieee.org/document/9960850)**


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transvod-end-to-end-video-object-detection/video-object-detection-on-imagenet-vid)](https://paperswithcode.com/sota/video-object-detection-on-imagenet-vid?p=transvod-end-to-end-video-object-detection)

(TPAMI 2023) [TransVOD:End-to-End Video Object Detection with Spatial-Temporal Transformers](https://ieeexplore.ieee.org/document/9960850).

:bell: We are happy to announce that TransVOD was accepted by **IEEE TPAMI**. 

:bell: We are happy to announce that our method is the first work that achieves 90% mAP on ImageNet VID dataset.


## Updates
- (January 2023) Checkpoints of pretrained models are scheduled to release. 
- (January 2023) Code of TransVOD++ are released. 

## Citing TransVOD
If you find TransVOD useful in your research, please consider citing:
```bibtex
@article{zhou2022transvod,
 author={Zhou, Qianyu and Li, Xiangtai and He, Lu and Yang, Yibo and Cheng, Guangliang and Tong, Yunhai and Ma, Lizhuang and Tao, Dacheng},  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},   
 title={TransVOD: End-to-End Video Object Detection With Spatial-Temporal Transformers},   
 year={2022},   
 pages={1-16},  
 doi={10.1109/TPAMI.2022.3223955}}


@inproceedings{he2021end,
  title={End-to-End Video Object Detection with Spatial-Temporal Transformers},
  author={He, Lu and Zhou, Qianyu and Li, Xiangtai and Niu, Li and Cheng, Guangliang and Li, Xiao and Liu, Wenxuan and Tong, Yunhai and Ma, Lizhuang and Zhang, Liqing},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={1507--1516},
  year={2021}
}
```


## Main Results
Our proposed method TransVOD++, achieving the best tradeoff between the speed and accuracy with different backbones. SwinB, SwinS and SwinT mean Swin Base, Small and Tiny.

![Comparison Results](fig/sota.png)



*Note:*
1. All models of TransVOD++ are trained  with pre-trained weights on COCO dataset.


## Installation

The codebase is built on top of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [TransVOD](https://github.com/SJTU-LuHe/TransVOD).

### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n TransVOD++ python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate TransVOD++
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/)

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    
    Note that you need to install mmcv-full and mmdetection before using the TransVOD++.
    ```bash
    pip install -r requirements.txt
    ```

* Build MultiScaleDeformableAttention
    ```bash
    cd ./models/ops
    sh ./make.sh
    ```

## Usage

### Checkpoints

Below, we provide checkpoints, training logs and inference logs of TransVOD++ in this link

[DownLoad Link of Google Drive](https://drive.google.com/drive/folders/1qXq6jz2-uvnUfa-IO2CoxsvEHnk93wee?usp=share_link)

[DownLoad Link of Baidu Netdisk](https://pan.baidu.com/s/1_8hCRWCXCSvqD4fsUYPnKA)(password:shm7)


## Recreate CBP Results
Refer to Google Drive (Matroid interns 2023 / Video object detection) for checkpoints and annotations files mentioned below. 

### Dataset preparation
Place CBP video frame data and single image data in path/to/data/directory. Also place annotation files for CBP video frame data and single image data in path/to/data/directory. Annotation files should be in COCO format with added video id and frame id information. Refer to CBP_coco_train.json, train_DET.json for example format of video data and single image data respectively. 

In the file datasets/vid_single.py, make sure the variable PATHS (at the bottom of the file) contains the correct names of the annotations files. 

Also in the file datasets/vid_multi.py, make sure the variable PATHS (at the bottom of the file) contains the correct names of the annotations files. 

### Training single frame modules
First we will train the single frame modules. Place the single_pretrain_checkpoint.pth (from the google drive) in path/to/pretrain/single. Alter the run_vid_single_exp.py: 
1. Make sure --data_root refers to the path/to/data/directory
2. Make sure --resume refers to path/to/pretrain/single/single_pretrain_checkpoint.pth
3. Make sure --num_classes is the number of classes plus 1. So if the classes are person and car then --num_classes should be 2+1=3. This is because we need to include the background class. 

#### Single GPU Training
```
python run_vid_single_exp.py
```

#### Distributed Training
Make sure to copy the arguments in run_vid_single_exp.py to swinb_train_single.sh. 

 ```
GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 configs/swinb_train_single.sh
```

### Training multi frame modules
After training the single frame modules, save the desired checkpoint to path/to/single/checkpoint.pth
1. Make sure --data_root refers to the path/to/data/directory
2. Make sure --resume refers to path/to/single/checkpoint.pth

#### Single GPU Training
```
python run_vid_multi_exp.py
```

#### Distributed Training
Make sure to copy the arguments in run_vid_multi_exp.py to swinb_train_multi.sh. 

 ```
GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 configs/swinb_train_multi.sh
```

### Evaluate
1. Make sure --resume refers to path/to/model.pth

In run_vid_multi_exp.py, uncomment the --eval. 

```
python run_vid_multi_exp.py
```

## Acknowledgements

This project is based on the following open-source projects. We thank their
authors for making the source code publically available.

* [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)


## License

This project is released under the [Apache License 2.0](LICENSE), while some 
specific features in this repository are with other licenses. Please refer to 
[LICENSES.md](LICENSES.md) for the careful check, if you are using our code for 
commercial matters.




