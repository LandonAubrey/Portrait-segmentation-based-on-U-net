# 人像分割项目

## 配置要求

### 硬件要求
- 建议使用的GPU内存大于16GB，否则运行时会出现内存不够的报错
- 如果使用PaddlePaddle平台，建议选用V100 32GB
- 内存不足，可以适当调低batch_size,改为8或者4
- 如果还是出现内存报错，建议重启环境再运行一遍代码

### 训练时间
⚠️ **注意：因为数据集较大，模型训练时间可能会较长(5个小时左右)**

## 数据集

### 数据源
数据集源自：https://github.com/aisegmentcn/matting_human_datasets

### 下载方式
**方式一：百度网盘下载**
- 链接：https://pan.baidu.com/s/1R9PJJRT-KjSxh-2-3wCGxQ
- 提取码：dzsn

**方式二：PaddlePaddle平台下载**
- 链接：https://aistudio.baidu.com/datasetdetail/338285

下载后可以使用ipynb文件中的命令行代码进行解压。

## 环境依赖

### 安装依赖包
```bash
!pip install beautifulsoup4 -t /home/aistudio/external-libraries --no-user
```

### 导入所需的库
```python
import os
import zipfile
import random
import json
import paddle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from paddle.io import Dataset, DataLoader
import cv2
import paddle.vision.transforms as T
import time
from tqdm.notebook import tqdm  # 使用notebook版本的tqdm
from visualdl import LogWriter
import paddle.nn as nn
import paddle.nn.functional as F
```

## 路径配置

⚠️ **如果不在PaddlePaddle平台上运行，请修改以下路径为本地数据集的路径**

### 查看数据集样本模块
```python
data_root = '/home/aistudio/work/matting_human_half'
```

### 主函数模块
```python
# 数据集根目录
data_root='./work/matting_human_half'
# 输出目录        
output_dir='./output'
```

### config参数配置
```python
'data_root': '/home/aistudio/work/matting_human_half',
'output_dir': './output',
'test_image': '/home/aistudio/work/matting_human_half/test.jpg'  # 测试图片路径
```

### 测试模块
```python
output_path = os.path.join('./output', 'visualization2.png')
visualize_results(
    model_path='./output/models/best_model.pdparams',
    test_image_path='./work/matting_human_half/test2.jpg',
    output_path=output_path,
    img_size=(512,384)
)
```
