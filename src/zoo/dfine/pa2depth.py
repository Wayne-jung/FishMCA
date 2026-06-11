from __future__ import absolute_import, division, print_function
import os
import argparse
from tqdm import tqdm
import yaml
import numpy as np
import cv2
import matplotlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
_PATH = '/root/autodl-tmp/PanDA-main'
import sys
if _PATH not in sys.path:
    sys.path.insert(0, _PATH)
import copy

models = {}

from networks.models import *

with open('/root/autodl-tmp/PanDA-main/config/inference/panda_large.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    print('config loaded.')
def pa2depth():
    model_path = config["load_weights_dir"]
    model_dict = torch.load(model_path, map_location='cpu') # 建议加 map_location='cpu' 防止直接加载到错误显卡
    model = make(config['model'])

    # --- 删除这两行 (DELETE THESE) ---
    # if any(key.startswith('module') for key in model_dict.keys()):
    #     model = nn.DataParallel(model)
    # -------------------------------

    # --- 新增：清洗 key 的逻辑 (ADD THIS) ---
    # 如果权重文件包含 'module.' 前缀，手动去掉它，而不是包裹模型
    new_state_dict = {}
    for k, v in model_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 去掉 'module.' (7个字符)
        else:
            new_state_dict[k] = v
    
    # 使用清洗后的 dict 加载
    model_state_dict = model.state_dict()
    # 过滤掉形状不匹配或不存在的 key
    model.load_state_dict({k: v for k, v in new_state_dict.items() if k in model_state_dict}, strict=False)

    # model.cuda()  <-- 建议注释掉这行。
    # 原因：作为子模块，它应该由主模型 (DFINE) 统一控制设备。
    # 如果这里写死 cuda()，可能会导致它被强行放到 cuda:0，而 DDP 希望它在 cuda:1, cuda:2...
    
    model.eval()
    return model