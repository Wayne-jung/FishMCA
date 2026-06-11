"""
评估脚本（适配 DA-2 风格）
用于评估 dfine 模型
"""

import os
import torch
from contextlib import nullcontext
from tqdm import tqdm

from src.utils.accelerate_utils import (
    prepare_to_run,
    load_model,
    get_autocast_context
)
from src.solver.det_engine import evaluate as det_evaluate
from src.data import CocoEvaluator


def eval_model(model, solver, config, accelerator, output_dir):
    """
    评估模型
    
    Args:
        model: 模型实例
        solver: DetSolver 实例
        config: 配置字典
        accelerator: Accelerator 实例
        output_dir: 输出目录
    """
    model = model.eval()
    
    if accelerator.is_main_process:
        # 获取评估配置
        eval_config = config.get('evaluation', {})
        eval_datasets = eval_config.get('datasets', {})
        
        # 获取 autocast 上下文
        autocast_ctx = get_autocast_context(accelerator)
        
        with autocast_ctx, torch.no_grad():
            # 遍历评估数据集
            for dataset_name in eval_datasets.keys():
                # 这里需要根据实际情况实现数据集加载
                # 暂时使用 solver 的 val 方法
                if hasattr(solver, 'val'):
                    # 使用 solver 的评估方法
                    solver.val()
                else:
                    # 使用 det_engine 的 evaluate 函数
                    # 需要准备 dataloader 和 evaluator
                    val_dataloader = solver.val_dataloader
                    evaluator = solver.evaluator
                    
                    metrics = det_evaluate(
                        model=solver.model,
                        criterion=solver.criterion,
                        postprocessor=solver.postprocessor,
                        data_loader=val_dataloader,
                        coco_evaluator=evaluator,
                        device=accelerator.device,
                        epoch=-1,
                        use_wandb=False,
                        output_dir=output_dir,
                    )
                    
                    # 打印指标
                    if config.get('env', {}).get('verbose', True):
                        logger = config['env']['logger']
                        for metric_name, metric_value in metrics.items():
                            if isinstance(metric_value, list):
                                metric_value = metric_value[0] if len(metric_value) > 0 else 0
                            logger.info(f"\033[92mEVAL --> {dataset_name}: {metric_name} = {metric_value}.\033[0m")


if __name__ == '__main__':
    # 准备运行环境
    config, accelerator, output_dir = prepare_to_run()
    
    # 加载模型
    model, solver = load_model(config, accelerator, config_path=config.get('config_path'))
    
    # 评估
    eval_model(model, solver, config, accelerator, output_dir)

