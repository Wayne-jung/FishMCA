#!/bin/bash
# 推理启动脚本（适配 DA-2 风格）
# 用于启动 dfine 模型的推理

export CUDA=0
export CONFIG_PATH="configs/infer.yaml"  # 或 infer.json

accelerate launch \
  --config_file=configs/accelerate/${CUDA}.yaml \
  --mixed_precision="fp16" \
  --main_process_port="12345" \
  infer.py --config_path=${CONFIG_PATH}




