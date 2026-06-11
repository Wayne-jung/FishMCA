"""
DA360 深度预测模块
参考 dfine.py 中对 VideoDepthAnything 的使用方式
提供可直接导入使用的 DA360 深度预测器
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import torch.nn.functional as F

# 添加 DA360 路径到 sys.path（如果不在同一目录）
_DA360_PATH = '/root/autodl-tmp/DA360'
if _DA360_PATH not in sys.path:
    sys.path.insert(0, _DA360_PATH)

try:
    import networks
    from networks.da360 import DA360
except ImportError as e:
    raise ImportError(
        f"无法导入 DA360 网络模块。请确保 DA360 项目路径正确。\n"
        f"当前尝试路径: {_DA360_PATH}\n"
        f"错误信息: {e}"
    )


class DA360DepthPredictor(nn.Module):
    """
    DA360 深度预测器封装类
    参考 dfine.py 中 VideoDepthAnything 的使用方式
    
    使用示例:
        # 初始化
        depth_predictor = DA360DepthPredictor(
            model_path='path/to/checkpoint.pth',
            height=518,
            width=1036,
            dinov2_encoder='vits',
            device='cuda'
        )
        
        # 推理
        with torch.no_grad():
            depth_map, depth_features = depth_predictor(input_image)
    """
    
    def __init__(
        self,
        model_path: str,
        height: int = 518,
        width: int = 1036,
        dinov2_encoder: str = 'vits',
        device: Optional[torch.device] = None,
        freeze_params: bool = True,
        **kwargs
    ):
        """
        初始化 DA360 深度预测器
        
        Args:
            model_path: 模型权重文件路径 (.pth)
            height: 输入图像高度，默认 518
            width: 输入图像宽度，默认 1036
            dinov2_encoder: DINOv2 编码器类型 ('vits', 'vitb', 'vitl', 'vitg')，默认 'vits'
            device: 计算设备，默认自动选择
            freeze_params: 是否冻结模型参数，默认 True
            **kwargs: 其他传递给 DA360 的参数
        """
        super().__init__()
        
        self.height = height
        self.width = width
        self.dinov2_encoder = dinov2_encoder
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 加载模型配置
        model_dict = self._load_model_dict(model_path)
        
        # 从 model_dict 中获取配置（如果存在）
        if 'height' in model_dict:
            self.height = model_dict['height']
        if 'width' in model_dict:
            self.width = model_dict['width']
        if 'dinov2_encoder' in model_dict:
            self.dinov2_encoder = model_dict['dinov2_encoder']
        elif 'net' in model_dict and model_dict['net'] != 'DA360':
            # 如果指定了不同的网络类型，使用默认值
            pass
        
        # 创建 DA360 模型
        self.depth_pred = DA360(
            equi_h=self.height,
            equi_w=self.width,
            dinov2_encoder=self.dinov2_encoder,
            **kwargs
        )
        
        # 加载权重
        self._load_weights(model_path, model_dict)
        
        # 冻结参数（如果启用）
        if freeze_params:
            for param in self.depth_pred.parameters():
                param.requires_grad = False
        
        # 设置为评估模式
        self.depth_pred.eval()
        
        # 移动到指定设备
        self.depth_pred.to(self.device)
    
    def _load_model_dict(self, model_path: str) -> Dict:
        """加载模型字典"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model_dict = torch.load(model_path, map_location='cpu')
        
        # 如果直接是 state_dict，返回空字典
        if not isinstance(model_dict, dict) or 'net' not in model_dict:
            return {}
        
        return model_dict
    
    def _load_weights(self, model_path: str, model_dict: Dict):
        """加载模型权重"""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 提取 state_dict
        if isinstance(checkpoint, dict):
            # 尝试从 checkpoint 中提取 state_dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # 假设整个字典就是 state_dict，但需要过滤掉非模型参数
                model_state_dict = self.depth_pred.state_dict()
                state_dict = {k: v for k, v in checkpoint.items() 
                             if k in model_state_dict}
        else:
            state_dict = checkpoint
        
        # 加载权重（允许部分匹配）
        model_state_dict = self.depth_pred.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                              if k in model_state_dict}
        
        if len(filtered_state_dict) == 0:
            raise ValueError(
                f"无法从模型文件中加载权重。\n"
                f"模型文件中的键: {list(state_dict.keys())[:5]}...\n"
                f"模型期望的键: {list(model_state_dict.keys())[:5]}..."
            )
        
        self.depth_pred.load_state_dict(filtered_state_dict, strict=False)
    

    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # 1. 确保设备一致
        if x.device != self.device:
            x = x.to(self.device)
        
        # 2. 处理输入维度 [C, H, W] -> [B, C, H, W]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        # 记录原始尺寸，用于最后还原
        orig_h, orig_w = x.shape[-2:]
        
        # 3. 计算符合 14 倍数的尺寸
        patch_size = 14
        new_h = ((orig_h + patch_size - 1) // patch_size) * patch_size
        new_w = ((orig_w + patch_size - 1) // patch_size) * patch_size
        
        # 4. 缩放图像到对齐尺寸 (使用双线性插值)
        if new_h != orig_h or new_w != orig_w:
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        
        with torch.no_grad():
            # DA360 模型前向传播
            outputs = self.depth_pred(x)
        
        # 5. 提取视差并转为深度
        pred_disp = outputs["pred_disp"]  # 假设形状为 [B, 1, new_H, new_W]
        pred_depth = 1.0 / (pred_disp + 1e-8)
        
        # 6. 缩放回原始尺寸
        # 注意：先保持 [B, 1, H, W] 的 4D 结构进行插值，更准确
        if pred_depth.dim() == 3:
            pred_depth = pred_depth.unsqueeze(1)
            
        if pred_depth.shape[-2:] != (orig_h, orig_w):
            pred_depth = F.interpolate(pred_depth, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        
        # 7. 移除通道维度 [B, 1, H, W] -> [B, H, W]
        pred_depth = pred_depth.squeeze(1)
        
        # 处理返回值
        if return_features:
            return pred_depth, None
        
        return pred_depth
    
    def predict_depth(
        self,
        image: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        预测深度（便捷方法）
        
        Args:
            image: 输入图像 [B, 3, H, W] 或 [3, H, W]
            normalize: 是否归一化深度值，默认 True
        
        Returns:
            depth_map: 深度图 [B, H, W] 或 [H, W]
        """
        depth = self.forward(image, return_features=False)
        
        # 移除单通道维度（如果存在）
        if depth.dim() == 4 and depth.shape[1] == 1:
            depth = depth.squeeze(1)
        
        # 归一化
        if normalize:
            depth = depth / (depth.min() + 1e-8)
        
        return depth
    
    def to_device(self, device: torch.device):
        """移动到指定设备"""
        self.device = device
        self.depth_pred.to(device)
        return self


def create_da360_predictor(
    model_path: str,
    height: int = 518,
    width: int = 1036,
    dinov2_encoder: str = 'vits',
    device: Optional[torch.device] = None,
    freeze_params: bool = True,
    **kwargs
) -> DA360DepthPredictor:
    """
    创建 DA360 深度预测器的便捷函数
    
    Args:
        model_path: 模型权重文件路径
        height: 输入图像高度
        width: 输入图像宽度
        dinov2_encoder: DINOv2 编码器类型
        device: 计算设备
        freeze_params: 是否冻结参数
        **kwargs: 其他参数
    
    Returns:
        DA360DepthPredictor 实例
    """
    return DA360DepthPredictor(
        model_path=model_path,
        height=height,
        width=width,
        dinov2_encoder=dinov2_encoder,
        device=device,
        freeze_params=freeze_params,
        **kwargs
    )


# 使用示例
if __name__ == "__main__":
    # 示例：如何在自己的模型中使用 DA360DepthPredictor
    
    # 1. 初始化（类似 dfine.py 中的方式）
    depth_predictor = DA360DepthPredictor(
        model_path='/path/to/da360/checkpoint.pth',
        height=518,
        width=1036,
        dinov2_encoder='vits',
        device=torch.device('cuda'),
        freeze_params=True
    )
    
    # 2. 在模型中使用（类似 dfine.py 的 forward 方法）
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.depth_pred = depth_predictor  # 或者直接在这里初始化
        
        def forward(self, x, targets=None):
            if targets is not None:
                # 预测深度
                depth_map, depth_features = self.depth_pred(x, return_features=True)
                # 使用深度信息...
                pass
    
    print("DA360DepthPredictor 模块加载成功！")

