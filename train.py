"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse

from src.core import YAMLConfig, yaml_utils
from src.misc import dist_utils
from src.solver import TASKS
from pprint import pprint

debug = False

if debug:
    def custom_repr(self):
        return f"{{Tensor:{tuple(self.shape)}}} {original_repr(self)}"

    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr


def safe_get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def main(args) -> None:
    """main"""

    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
# 2. 获取 rank
    rank = safe_get_rank()

    # 3. 强制配置日志 (传入 output_dir)
    # 如果 args.output_dir 为空，默认用当前目录
    output_dir = args.output_dir if args.output_dir else "./"
    setup_force_logger(output_dir, rank)
    
    # 4. 打印测试
    if rank == 0:
        logging.info(f"主程序启动 - Log configured successfully in {output_dir}")
    assert not all(
        [args.tuning, args.resume]
    ), "Only support from_scrach or resume or tuning at one time"

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update(
        {
            k: v
            for k, v in args.__dict__.items()
            if k
            not in [
                "update",
            ]
            and v is not None
        }
    )

    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume or args.tuning:
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if safe_get_rank() == 0:
        print("cfg: ")
        pprint(cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()
import logging

def setup_force_logger(output_dir, rank):
    # 1. 获取 Root Logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清理旧 Handler (仅 Rank 0 清理，防止重复)
    if rank == 0 and logger.hasHandlers():
        logger.handlers.clear()

    # ------------------ 定义过滤器 ------------------
    class UselessLogFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            # 在这里添加你想屏蔽的任何关键词
            keywords = [
                "Image embeddings computed",
                "For numpy array image",
                "Computing image embeddings",
            ]
            for k in keywords:
                if k in msg:
                    # 返回 False 表示丢弃这条日志
                    return False
            return True
    
    # 实例化过滤器
    my_filter = UselessLogFilter()
    # -----------------------------------------------

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 2. 配置 Handler (仅 Rank 0)
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "training.log")
        
        # --- 文件输出 ---
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.addFilter(my_filter) # <--- 给 Handler 也加上过滤器
        logger.addHandler(file_handler)
        
        # --- 屏幕输出 ---
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(my_filter) # <--- 给 Handler 也加上过滤器
        logger.addHandler(console_handler)

    # 3. 给 Logger 本身也加上过滤器 (双重保险)
    logger.addFilter(my_filter)
    
    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, help="resume from checkpoint")
    parser.add_argument("-t", "--tuning", type=str, help="tuning from checkpoint")
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        help="device",
    )
    parser.add_argument("--seed", type=int, help="exp reproducibility")
    parser.add_argument("--use-amp", action="store_true", help="auto mixed precision training")
    parser.add_argument("--output-dir", type=str, help="output directoy")
    parser.add_argument("--summary-dir", type=str, help="tensorboard summry")
    parser.add_argument(
        "--test-only",
        action="store_true",
        default=False,
    )

    # priority 1
    parser.add_argument("-u", "--update", nargs="+", help="update yaml config")

    # env
    parser.add_argument("--print-method", type=str, default="builtin", help="print method")
    parser.add_argument("--print-rank", type=int, default=0, help="print rank id")

    parser.add_argument("--local-rank", type=int, help="local rank id")

    parser.add_argument("-v", '--visual', action="store_true",help="visual the result for evaluate")
    # parser.add_argument("--d", type=bool, default=False, help="visual for depth map")
    parser.add_argument("-track",'--track', action="store_true",help="tracking using bytetrack")

    args = parser.parse_args()

    main(args)
