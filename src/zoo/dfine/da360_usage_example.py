"""
DA360 深度预测器使用示例
展示如何在你的模型中使用 DA360DepthPredictor（参考 dfine.py 中 VideoDepthAnything 的使用方式）
"""

import torch
import torch.nn as nn
from .da360_depth_predictor import DA360DepthPredictor, create_da360_predictor


# ========== 示例 1: 在模型类中直接使用（类似 dfine.py） ==========

class MyModelWithDA360(nn.Module):
    """
    示例：在你的模型中使用 DA360DepthPredictor
    参考 dfine.py 中 VideoDepthAnything 的使用方式
    """
    
    def __init__(
        self,
        backbone,
        encoder,
        decoder,
        da360_model_path: str = './DA360/checkpoints/your_model.pth',
        height: int = 518,
        width: int = 1036,
        dinov2_encoder: str = 'vits',
        device: torch.device = None
    ):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        
        # ========== 关键部分：初始化 DA360 深度预测器 ==========
        # 参考 dfine.py 第 85-97 行的方式
        self.depth_pred = DA360DepthPredictor(
            model_path=da360_model_path,
            height=height,
            width=width,
            dinov2_encoder=dinov2_encoder,
            device=device,
            freeze_params=True  # 冻结参数，不参与梯度更新
        )
        
        # 注意：DA360DepthPredictor 内部已经设置了：
        # - param.requires_grad = False（如果 freeze_params=True）
        # - model.eval()
        # 所以不需要再手动设置
    
    def forward(self, x, targets=None):
        """
        前向传播
        参考 dfine.py 第 100-109 行的方式
        """
        if targets is not None:
            b, c, h, w = x.shape
            
            # ========== 关键部分：使用 DA360 预测深度 ==========
            # 参考 dfine.py 第 108 行：y, depth_fea = self.depth_pred(x.unsqueeze(dim=0).detach())
            
            # 方式1：直接调用（推荐）
            depth_map = self.depth_pred(x.detach())  # [B, H, W] 或 [B, 1, H, W]
            
            # 方式2：如果需要特征（如果模型支持）
            # depth_map, depth_features = self.depth_pred(x.detach(), return_features=True)
            
            # 方式3：使用便捷方法
            # depth_map = self.depth_pred.predict_depth(x.detach(), normalize=True)
            
            # 后续处理...
            x = self.backbone(x)
            x = self.encoder(x)
            x = self.decoder(x, targets)
            
            # 使用深度信息...
            # 例如：将深度信息融合到检测结果中
            
            return x, depth_map
        
        # 推理模式
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)
        return x


# ========== 示例 2: 使用便捷函数创建 ==========

def create_model_with_da360(
    backbone,
    encoder,
    decoder,
    da360_model_path: str,
    **da360_kwargs
):
    """
    使用便捷函数创建模型
    """
    model = MyModelWithDA360(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        da360_model_path=da360_model_path,
        **da360_kwargs
    )
    return model


# ========== 示例 3: 单独使用 DA360 进行深度预测 ==========

def example_standalone_usage():
    """
    示例：单独使用 DA360 进行深度预测
    """
    # 创建预测器
    depth_predictor = create_da360_predictor(
        model_path='/path/to/da360/checkpoint.pth',
        height=518,
        width=1036,
        dinov2_encoder='vits',
        device=torch.device('cuda'),
        freeze_params=True
    )
    
    # 准备输入（需要归一化的图像）
    # 假设你有一个图像张量 [B, 3, H, W]，已经归一化
    # 归一化方式：mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    input_image = torch.randn(1, 3, 518, 1036).cuda()
    
    # 预测深度
    with torch.no_grad():
        depth_map = depth_predictor.predict_depth(input_image, normalize=True)
        # depth_map shape: [B, H, W]
    
    print(f"深度图形状: {depth_map.shape}")
    return depth_map


# ========== 示例 4: 替换 dfine.py 中的 VideoDepthAnything ==========

def example_replace_videodepthanything():
    """
    示例：如何将 dfine.py 中的 VideoDepthAnything 替换为 DA360
    
    原来的代码（dfine.py 第 85-97 行）：
        self.depth_pred = VideoDepthAnything(**{'encoder': 'vits', ...})
        self.depth_pred.load_state_dict(torch.load(...), strict=True)
        for param in self.depth_pred.parameters():
            param.requires_grad = False
        self.depth_pred.eval()
    
    替换为：
        self.depth_pred = DA360DepthPredictor(
            model_path='./DA360/checkpoints/your_model.pth',
            height=518,
            width=1036,
            dinov2_encoder='vits',
            freeze_params=True  # 自动冻结参数
        )
        # 注意：不需要手动设置 requires_grad 和 eval()，已经自动完成
    """
    pass


# ========== 示例 5: 处理输入图像归一化 ==========

def prepare_input_for_da360(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    准备输入图像（归一化）
    
    DA360 需要归一化的输入：
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    
    Args:
        image_tensor: [B, 3, H, W] 或 [3, H, W]，值域 [0, 1]
    
    Returns:
        归一化后的图像张量
    """
    from torchvision import transforms
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    normalized_image = normalize(image_tensor)
    return normalized_image


if __name__ == "__main__":
    print("DA360 使用示例")
    print("=" * 50)
    print("\n1. 在模型类中使用（参考 dfine.py）:")
    print("   - 在 __init__ 中初始化 DA360DepthPredictor")
    print("   - 在 forward 中调用 self.depth_pred(x)")
    print("\n2. 单独使用:")
    print("   - 创建预测器实例")
    print("   - 调用 predict_depth() 方法")
    print("\n3. 注意事项:")
    print("   - 输入图像需要归一化（ImageNet 标准）")
    print("   - 默认冻结参数，不参与梯度更新")
    print("   - 自动设置为 eval 模式")

