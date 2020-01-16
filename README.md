# DLAAE
《Discriminative Latent Attribute Autoencoder for Zero-Shot Learning》CCIS2018。论文代码很大程度参考[SAE](https://github.com/Elyorcv/SAE)和[LAD](http://vipl.ict.ac.cn/resources/codes)。

## Dateset

* AwA(https://cvml.ist.ac.at/AwA/)

## Requirements

* Matlab==2017a

## Usage

1. Download the initial mat from https://drive.google.com/drive/folders/1gLnAgZGLtUGjpKcXxjL5VWWRkkBCaQqa, and then put it in `./datasets/`
2. Download the vgg feature of AwA and put those in `./datasets/AwA/`. Then `datasets` will be like this:
```
datasets
│  initial_awa_ADS.mat
│
└─AwA
        feat-imagenet-vgg-verydeep-19.mat
        predicateMatrixContinuous.mat
        trainTestSplit.mat
```
3. run `main.m`

## Description

1. 读取数据
2. 归一化
3. PCA降维
4. 初始化
5. 计算相似性空间
6. 计算one-hot标签
7. 计算出 $Y_s$
8. 根据字典 $D$ 与 $Y_s$ 用阿格朗日对偶问题计算 $W$
9. 根据字典 $D$ 与 $Y_s$ 用SAE计算 $U$
10. 迭代优化
11. 测试


## References 

* Kodirov E, Xiang T, Gong S. Semantic Autoencoder for Zero-Shot Learning[C]//2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2017: 4447-4456.
* Jiang H, Wang R, Shan S, et al. Learning Discriminative Latent Attributes for Zero-Shot Classification[C]//2017 IEEE International Conference on Computer Vision (ICCV). IEEE, 2017: 4233-4242.