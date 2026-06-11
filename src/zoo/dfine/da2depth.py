import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
# 添加 DA360 路径到 sys.path（如果不在同一目录）
_DA360_PATH = '/root/autodl-tmp/DA-2'
import sys
if _DA360_PATH not in sys.path:
    sys.path.insert(0, _DA360_PATH)


import os
import torch
from contextlib import nullcontext
from tqdm import tqdm
from da2 import (
    prepare_to_run,
    load_model
)
from eval.utils import run_evaluation
config, accelerator, output_dir = prepare_to_run()
model = load_model(config, accelerator)
class DA2DepthPredictor(nn.Module):
    def __init__(
        self, 
        model_path: str, 
        height: int = 490, 
        width: int = 2058, 
        dinov2_encoder: str = 'vits', 
        device: torch.device = torch.device('cuda'),
        freeze_params: bool = True
    ):
        super().__init__()
        self.device = device
        self.target_h = height
        self.target_w = width
        
        # 1. 模拟 config 对象以适配 load_model 函数
        # 注意：这里的字段名需要根据你的 configs/infer.json 结构微调
        class MockConfig:
            def __init__(self):
                self.model = {
                    "backbone": dinov2_encoder,
                    "pretrained": model_path
                }
                # 如果 load_model 内部需要更多参数，在此添加
                self.checkpoint = model_path 
        
        # config = MockConfig()
        
        # # 2. 加载模型 (这里假设 accelerator 可以传 None 或简单的对象)
        # # 如果 load_model 内部强制要求 accelerator，请根据实际情况传入
        # class MockAccelerator:
        #     def __init__(self, device):
        #         self.device = device
        #     def prepare(self, model):
        #         return model
        
        # accel = MockAccelerator(device)
        self.model = model
        self.model.to(device)
        self.model.eval()

        if freeze_params:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(
        self, 
        x: torch.Tensor, 
        return_features: bool = False
    ) -> torch.Tensor:
        """
        x: 输入张量 [B, 3, H, W]
        """
        # 确保输入在正确的设备上
        if x.device != self.device:
            x = x.to(self.device)

        # 记录原始尺寸用于还原
        orig_h, orig_w = x.shape[-2:]

        # # 1. 缩放到符合模型要求的对齐尺寸 (如 14 的倍数)
        # if (orig_h != self.target_h) or (orig_w != self.target_w):
        #     x_input = F.interpolate(x, size=(self.target_h, self.target_w), mode="bilinear", align_corners=False)
        # else:
        x_input = x

        # 2. 混合精度设置
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            # 自动识别是 cuda 还是 cpu 的 autocast
            autocast_ctx = torch.autocast(device_type=self.device.type)

        # 3. 推理
        with autocast_ctx, torch.no_grad():
            # 这里对应原代码中的 distances = model(x)
            # 根据 DA360 返回值结构，通常返回的是 dict 或 Tensor
            outputs = self.model(x_input)
            
            # 处理可能的字典返回格式
            if isinstance(outputs, dict):
                pred_disp = outputs.get("pred_disp", outputs.get("distance"))
            else:
                pred_disp = outputs

        # # 5. 移除通道维度，返回 [B, H, W]
        pred_depth = pred_disp*10

        return pred_depth