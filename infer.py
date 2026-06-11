"""
推理脚本（适配 DA-2 风格）
用于使用 dfine 模型进行推理
"""

import os
import torch
from contextlib import nullcontext
from tqdm import tqdm
from pathlib import Path

from src.utils.accelerate_utils import (
    prepare_to_run,
    load_model,
    get_autocast_context
)
from src.solver.det_engine import evaluate as det_evaluate


def infer_model(model, solver, config, accelerator, output_dir):
    """
    使用模型进行推理
    
    Args:
        model: 模型实例
        solver: DetSolver 实例
        config: 配置字典
        accelerator: Accelerator 实例
        output_dir: 输出目录
    """
    model.eval()
    
    if accelerator.is_main_process:
        # 获取推理配置
        infer_config = config.get('inference', {})
        
        # 获取 autocast 上下文
        autocast_ctx = get_autocast_context(accelerator)
        
        with autocast_ctx, torch.no_grad():
            # 准备数据加载器
            # 这里需要根据实际情况实现数据加载
            # 暂时使用 val_dataloader 作为示例
            
            if hasattr(solver, 'val_dataloader') and solver.val_dataloader is not None:
                dataloader = solver.val_dataloader
            else:
                # 如果没有 val_dataloader，需要从配置中创建
                raise ValueError("需要提供推理数据加载器")
            
            # 创建输出目录
            infer_output_dir = os.path.join(output_dir, 'inference_results')
            os.makedirs(infer_output_dir, exist_ok=True)
            
            # 推理循环
            results = []
            for i, (samples, targets) in enumerate(tqdm(dataloader, desc='Predicting')):
                # 移动到设备
                samples = samples.to(accelerator.device)
                targets = [{k: (v.to(accelerator.device) if isinstance(v, torch.Tensor) else v) 
                           for k, v in t.items()} for t in targets]
                
                # 模型推理
                outputs, depth_map, _ = model(samples)
                
                # 后处理
                orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                det_results = solver.postprocessor(outputs, orig_target_sizes)
                
                # 保存结果
                # 这里可以根据需要保存检测结果、深度图等
                img_paths = [t.get("image_path", f"image_{i}_{j}.jpg") 
                            for j, t in enumerate(targets)]
                
                # 保存检测结果（如果需要）
                if infer_config.get('save_detections', True):
                    from src.solver.utils import save_results_with_depth
                    save_results_with_depth(
                        det_results,
                        img_paths,
                        os.path.join(infer_output_dir, 'detections')
                    )
                
                results.append({
                    'detections': det_results,
                    'depth_maps': depth_map,
                    'image_paths': img_paths
                })
            
            # 保存汇总结果
            if infer_config.get('save_summary', True):
                summary_path = os.path.join(infer_output_dir, 'summary.txt')
                with open(summary_path, 'w') as f:
                    f.write(f"Total images processed: {len(results)}\n")
                    f.write(f"Output directory: {infer_output_dir}\n")
                
                config['env']['logger'].info(f"Inference completed. Results saved to {infer_output_dir}")


if __name__ == '__main__':
    # 准备运行环境
    config, accelerator, output_dir = prepare_to_run()
    
    # 加载模型
    model, solver = load_model(config, accelerator, config_path=config.get('config_path'))
    
    # 推理
    infer_model(model, solver, config, accelerator, output_dir)

