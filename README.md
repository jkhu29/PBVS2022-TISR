# PBVS @ CVPR 2022

[Thermal Images Super-resolution Challenge](https://codalab.lisn.upsaclay.fr/competitions/1990#learn_the_details)

requirements:

1. timm
2. tfrecord

## usage

### SwinIR + WGAN (PSNR: 33.15, SSIM: 0.9133)

SwinIR + WGAN, use L1Loss & TVLoss & SSIMLoss

```shell
python make_dataset.py
# if you want to see the images in tfrecord files
# python -W ignore test_data.py

# for paired x4
sh train.sh

# for paired x2, add --scale 2 to train.sh

# for unpaired x2
sh train_cycle.sh
```

### Our Method(PSNR: 34.2, SSIM: 0.9249)

coming soon...

## pretrain models

SwinIR x4: [Google Drive](https://drive.google.com/file/d/1l0RlJgfo9pPdjF54T1-53s0yQENm3JTB/view?usp=sharing) or [Baidu Pad p: wcl1](https://pan.baidu.com/s/1-NIP2a9KOngDNG_wDndchw)

put it in "./models/x4", then `python test.py`

> test on ubuntu20.04LTS pytorch-1.7.0 RTX3090
