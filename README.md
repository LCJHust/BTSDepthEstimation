# From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation 

## Introduction
This is a PyTorch(0.4.1) implementation of [From Big to Small: Multi-Scale Local Planar Guidance
for Monocular Depth Estimation](https://arxiv.org/abs/1907.10326).

## Usage
1. For training:
```
python train.py config/bts_cfg.py
```

2. For inference or eval with pretrained ckpt:
The pretrained ckpt is at: **node01: /home/caojia/Projects/ckpts/bts_best_kitti.pth**
```
python eval.py
```

## Performance
| **D1**   | **D2**      | **D3**      | **AbsRel**      | **SquaRel** | **RMSE** | **RMSE_log** |**log10**|  
| :------: | :---------: | :---------: | :-------------: |  :--------: | :------: | :----------: |  :----: |     
|  0.9313 |  0.9874     |  0.9968     |   0.0731        |  0.3393     |  3.1952  |    --  |  0.0326  | 
