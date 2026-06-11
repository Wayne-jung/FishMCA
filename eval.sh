#!/bin/bash
# 评估启动脚本（适配 DA-2 风格）
# 用于启动 dfine 模型的评估

export CUDA=0
export CONFIG_PATH="configs/eval.yaml"  # 或 eval.json

accelerate launch \
  --config_file=configs/accelerate/${CUDA}.yaml \
  --mixed_precision="fp16" \
  --main_process_port="12345" \
  eval.py --config_path=${CONFIG_PATH}




