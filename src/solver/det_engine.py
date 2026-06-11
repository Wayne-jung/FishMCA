"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
"""
from torch._tensor import Tensor
from ByteTrack.yolox.tracker.byte_tracker import BYTEDTracker
import math
import sys
import os
import argparse
from typing import Dict, Iterable, List, Tuple, Any
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.amp
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
from contextlib import contextmanager

from ..data import CocoEvaluator
from ..data.dataset import mscoco_category2label
from ..misc import MetricLogger, SmoothedValue, dist_utils, save_samples
from ..optim import ModelEMA, Warmup
from .validator import Validator, scale_boxes
from .utils import *
from ..zoo.dfine.dfine_utils import plot_distributions

# 导入 motmetrics 用于跟踪评估
try:
    import motmetrics as mm
    MOTMETRICS_AVAILABLE = True
except ImportError:
    MOTMETRICS_AVAILABLE = False
    print("警告: motmetrics 未安装，无法进行跟踪评估。请安装: pip install motmetrics")



# ----------------------------
# 评估时临时把滑窗 window_len 设置为 1
# ----------------------------
@contextmanager
def _temp_window1(dataloader):
    """
    在 with 块中把滑窗长度临时改为 1；退出时恢复。
    仅当 dataloader.dataset 是 SlidingWindowView 并且其 base 有 window_len 时生效。
    """
    ds = getattr(dataloader, "dataset", None)
    base = getattr(ds, "base", None)  # _SlidingWindowView.base -> 你的 CocoDetection
    if base is None or not hasattr(base, "window_len"):
        # 非滑窗或拿不到 window_len，直接运行
        yield None
        return

    old_len = int(base.window_len)
    try:
        base.window_len = 1          # ✅ eval 只取单帧
        # 重新构建 sample_begin_frames
        if hasattr(base, "set_epoch"):
            # 用 dataloader 记录的 epoch，拿不到就用 0
            epoch = getattr(dataloader, "epoch", 0)
            base.set_epoch(epoch)
        yield old_len
    finally:
        # 恢复
        base.window_len = old_len
        if hasattr(base, "set_epoch"):
            epoch = getattr(dataloader, "epoch", 0)
            base.set_epoch(epoch)


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool,
    max_norm: float = 0,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    epochs = kwargs.get("epochs", None)
    header = "Epoch: [{}]".format(epoch) if epochs is None else "Epoch: [{}/{}]".format(epoch, epochs)

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)
    losses = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "train", normalized=True, box_fmt="cxcywh")

        samples = samples.to(device)
        targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)

        # ===== 前向（兼容 AMP 与多返回）=====
        if scaler is not None:
            with torch.autocast(device_type=device.type, cache_enabled=True, dtype=torch.float16, enabled=True):
                outputs, pred_depth, depth_fea , depth,gt_distribution = model(samples, targets=targets)
                # = _split_model_outputs(raw_out)

                # # 训练首个 step 做一次调试打印
                # if i == 0 and (not dist_utils.is_dist_available_and_initialized() or dist_utils.is_main_process()):
                #     tb = targets[0]["boxes"]
                #     pb = outputs.get("pred_boxes", None) if isinstance(outputs, dict) else None
                #     print("[debug] image size (HxW):", samples.shape[-2], "x", samples.shape[-1])
                #     print("[debug] targets range:", float(tb.min()), float(tb.max()))
                #     if isinstance(outputs, dict) and pb is not None:
                #         print("[debug] preds   range:", float(pb.detach().min()), float(pb.detach().max()))
                #         print("[debug] pred_boxes shape:", tuple(outputs["pred_boxes"].shape))
                #     if isinstance(outputs, dict) and "pred_logits" in outputs:
                #         pl = outputs["pred_logits"].detach()
                #         print("[debug] pred_logits shape:", tuple(pl.shape), "range:", float(pl.min()), float(pl.max()))
                #     print("[debug] tgt_boxes  shape:", tuple(tb.shape))
                #     print("[debug] criterion box_fmt:", getattr(criterion, "box_fmt", "<no-box_fmt>"))

                # NaN/Inf 保护
                if isinstance(outputs, dict) and "pred_boxes" in outputs:
                    if torch.isnan(outputs["pred_boxes"]).any() or torch.isinf(outputs["pred_boxes"]).any():
                        print(outputs["pred_boxes"])
                        state = model.state_dict()
                        new_state = {}
                        for key, value in state.items():
                            new_key = key.replace("module.", "")
                            new_state[new_key] = value
                        dist_utils.save_on_master({"model": new_state}, "./NaN.pth")

                # 兼容 criterion 是否需要 fea/depth_fea
                fea = outputs["fea"]
                try:
                    loss_dict = criterion(outputs, targets, pred_depth, fea, depth_fea, depth, gt_distribution,**metas)
                except TypeError:
                    # 老签名：不带 fea/depth_fea
                    try:
                        loss_dict = criterion(outputs, targets, pred_depth, fea, depth_fea, depth, gt_distribution, **metas)
                    except TypeError:
                        loss_dict = criterion(outputs, targets, pred_depth, fea, depth_fea, depth, gt_distribution,**metas)

                loss = sum(loss_dict.values())
            # 反传
            scaler.scale(loss).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # 非 AMP
            outputs, pred_depth, depth_fea , depth,gt_distribution = model(samples, targets=targets)
            
            fea = outputs["fea"]
            try:
                loss_dict = criterion(outputs, targets, pred_depth, fea, depth_fea,depth, gt_distribution, **metas)
            except TypeError:
                try:
                    loss_dict = criterion(outputs, targets, pred_depth,depth, **metas)
                except TypeError:
                    loss_dict = criterion(outputs, targets, **metas)

            loss: torch.Tensor = sum(loss_dict.values())
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        # EMA & warmup
        if ema is not None:
            ema.update(model)
        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        # 统计
        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        losses.append(loss_value.detach().cpu().numpy())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", v.item(), global_step)

    if use_wandb:
        wandb.log({"lr": optimizer.param_groups[0]["lr"], "epoch": epoch, "train/loss": np.mean(losses)})
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
import time  # 引入时间库
from datetime import datetime  #
import logging
from .tracker import evaluate_dancetrack
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
    epoch: int,
    use_wandb: bool,
    is_visual=False,
    is_track=False,
    **kwargs,
):
    if use_wandb:
        import wandb
    logging.info("#####################################evaluate############################### \n")
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"
    iou_types = coco_evaluator.iou_types

    total_compute_time = 0.0
    total_frames = 0

    gt: List[Dict[str, torch.Tensor]] = []
    preds: List[Dict[str, torch.Tensor]] = []
    seen_img_ids = set()  # ✅ 每张图只评一次，避免 COCO 重复

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)
    
    if is_track:
        from types import SimpleNamespace
        # 请确保 BYTEDTracker 已正确导入
        tracker = BYTEDTracker(args = argparse.Namespace(**{
            "aspect_ratio_thresh":1.6,
            "min_box_area":10,
            "track_thresh": 0.5,
            "track_buffer": 30,
            "match_thresh": 0.8,
            "mot20": False,
            }))

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    track_save_path = f'./visual_out/track/{time_str}/data'
    os.makedirs(track_save_path,exist_ok=True)

    # === 关键：评估时把 window_len 临时改为 1 ===
    last_scene_id = None 
# [时间统计] 初始化变量
    total_compute_time = 0.0
    total_frames = 0

    with _temp_window1(data_loader):
        for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            
            global_step = epoch * len(data_loader) + i
            id_map = {}
            next_local_id = 1
            if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
                save_samples(samples, targets, output_dir, "val", normalized=False, box_fmt="xyxy")

            samples = samples.to(device)
            targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]
            # [FPS统计] 计时开始 (在数据送入 GPU 后，模型运行前)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_t = time.perf_counter()
            # 模型推理
            outputs, depth_map, _ = model(samples)

            # 尺寸还原
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessor(outputs, orig_target_sizes)

            # 获取当前 Batch 所有图片的路径和场景 ID
            img_paths = [t["image_path"] for t in targets]
            current_scenes = [p.split('/')[-3] for p in img_paths] 
            if is_visual:
                save_results_with_depth(results, img_paths, os.path.join('./visual_out', "detection_results"))

            # ==========================================================
            # [Tracking 逻辑] 逐帧处理
            # ==========================================================
            if is_track:
                for idx, (res, tgt, scene_id, path) in enumerate(zip(results, targets, current_scenes, img_paths)):
                    frame_name = os.path.splitext(os.path.basename(img_paths[idx]))[0]
                    seq = img_paths[idx].split('/')[-3]
                    frame_idx =  int(frame_name)
                    res_file = track_save_path+f"/{seq}.txt"

                    save_results = []
                    # 1. 检查场景是否切换
                    if scene_id != last_scene_id:

                        if hasattr(tracker, 'reset'):
                            tracker.reset()
                        else:
                             # 如果 tracker 没有 reset，尝试 clear 或重新初始化
                            if hasattr(tracker, 'clear'): tracker.clear()
                        last_scene_id = scene_id 
                    # if res[-1] == 1
                    # 2. 构造单帧输入
                    single_res_list = [res] 
                    single_info_img = tgt["orig_size"].unsqueeze(0).cpu()
                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    # 3. 执行跟踪
                    online_target = track(single_res_list, tracker, info_imgs=single_info_img)[0]
                    for t in online_target:
                        tlwh = t.tlwh
                        tid = t.track_id
                        if tid not in id_map:
                            id_map[tid] = next_local_id
                            next_local_id += 1

                        # if tlwh[2] * tlwh[3] > 10 and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        # save results
                        save_results.append(
                            f"{frame_idx},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )

                    with open(res_file, 'a+') as f:
                        f.writelines(save_results)
# [FPS统计] 计时结束 (当前 Batch 全部处理完)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_t = time.perf_counter()
            
            # 累加
            batch_time = end_t - start_t
            batch_size = len(targets)
            total_compute_time += batch_time
            total_frames += batch_size
                # if is_visual:
                #     visualize_track_results(
                #         img_paths=img_paths,
                #         track_results=online_target,
                #         output_dir="./visual_out",  # 跟踪可视化结果保存目录
                #         class_names=kwargs.get("class_names", None),  # 类别名称列表（可选）
                #         depth_maps=depth_map
                #     )

            # ==========================================================

            # [时间统计] 结束计时 (当前 Batch 处理完毕)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_t = time.perf_counter()
            
            # 累加时间与帧数
            batch_time = end_t - start_t
            batch_size = len(targets)
            total_compute_time += batch_time
            total_frames += batch_size
            # ==========================================================
            
            # [COCO Detection 逻辑]
            batch_res = {}
            for idx, (tgt, resi) in enumerate[tuple[dict[Any, Tensor | Any], Any]](zip(targets, results)):
                img_id = int(tgt["image_id"].item())
                if img_id in seen_img_ids:
                    continue
                seen_img_ids.add(img_id)
                batch_res[img_id] = resi

                gt.append({
                    "boxes": scale_boxes(
                        tgt["boxes"],
                        (tgt["orig_size"][1], tgt["orig_size"][0]),
                        (samples[idx].shape[-1], samples[idx].shape[-2]),
                    ),
                    "labels": tgt["labels"],
                })
                
                if getattr(postprocessor, "remap_mscoco_category", False):
                    labs = torch.tensor(
                        [mscoco_category2label[int(x.item())] for x in resi["labels"].flatten()],
                        device=resi["labels"].device
                    ).reshape(resi["labels"].shape)
                else:
                    labs = resi["labels"]
                preds.append({"boxes": resi["boxes"], "labels": labs, "scores": resi["scores"]})

            if coco_evaluator is not None and len(batch_res) > 0:
                coco_evaluator.update(batch_res)
    # ==========================================================
    # [FPS 结果输出] 循环结束后计算
    # ==========================================================
    avg_fps = total_frames / total_compute_time if total_compute_time > 0 else 0.0
    avg_latency = (total_compute_time / total_frames * 1000) if total_frames > 0 else 0.0
    
    fps_msg = (
        f"\n------------------------------------------------------------\n"
        f" [Inference Speed Analysis] \n"
        f" Total Frames: {total_frames}\n"
        f" Total Time  : {total_compute_time:.4f}s (Model + PostProcess + Tracking)\n"
        f" Average FPS : {avg_fps:.2f} FPS\n"
        f" Latency     : {avg_latency:.2f} ms/frame\n"
        f"------------------------------------------------------------\n"
    )
    logging.info(fps_msg)
    
    # ==========================================================
    # [Evaluation 1] MOT Tracking Metrics (HOTA, IDF1, etc.)
    # ==========================================================
    if is_track and dist_utils.is_main_process():
        # 运行 TrackEval
        # 注意：ensure 'track_save_path' points to the tracker folder, not 'data' subfolder depending on your implementation
        output_res, output_msg = evaluate_dancetrack(track_save_path.replace('/data', ''))
        
        # --- 鲁棒地提取 MOT 关键指标 ---
        try:
            # TrackEval 的输出结构很深，通常是: dataset -> data -> COMBINED_SEQ -> class -> metric_group
            # 1. 动态获取数据集名称 (例如 'MotChallenge2DBox')
            dataset_key = list(output_res.keys())[0]
            
            # 2. 定位到汇总结果 (COMBINED_SEQ -> pedestrian)
            # 如果你的类别不是 'pedestrian'，TrackEval 可能会输出 'all_classes' 或其他
            # 这里做一个简单的兼容性检查
            seq_res = output_res[dataset_key]['data']['COMBINED_SEQ']
            class_key = 'pedestrian' if 'pedestrian' in seq_res else list(seq_res.keys())[0]
            combined_res = seq_res[class_key]
    
            # 3. 提取各个指标组
            hota_group = combined_res.get('HOTA', {})
            clear_group = combined_res.get('CLEAR', {})
            id_group    = combined_res.get('Identity', {})
    
            # 4. 提取我们关心的 5 大金刚指标
            # 注意：HOTA/DetA/AssA 在 HOTA 组，MOTA 在 CLEAR 组，IDF1 通常在 Identity 组
            mot_metrics = {
                "HOTA": float(hota_group.get("HOTA", 0.0).mean().item()),
                "DetA": float(hota_group.get("DetA", 0.0).mean().item()),
                "AssA": float(hota_group.get("AssA", 0.0).mean().item()),
                "MOTA": float(clear_group.get("MOTA", 0.0).item()),
                "IDF1": float(id_group.get("IDF1", 0.0).item()) # 有些版本 IDF1 也在 CLEAR 里，视情况调整
            }
    
            # 5. 打印 Console 日志 & 加入 WandB
            log_lines = ["\n[Tracking] MOT Evaluation Results:"]
            # 表头
            log_lines.append(f"  {'Metric':<10} | {'Score':<10}")
            log_lines.append(f"  {'-'*25}")
            
            for k, v in mot_metrics.items():
                # 打印: HOTA       | 0.562
                log_lines.append(f"  {k:<10} | {v:.3f}")
    
    
    
            logging.info("\n".join(log_lines))

        except Exception as e:
            logging.error(f"Failed to extract MOT metrics: {e}")
            # 如果解析失败，把原始字典打出来方便调试
            logging.debug(f"Raw TrackEval output: {output_res}")

    # ==========================================================
    # [Evaluation 2] Custom Validator Metrics
    # ==========================================================
    # 假设 gt, preds 是你之前收集的
    metrics = Validator(gt, preds).compute_metrics()
    logging.info(f"[Custom] Validator Metrics: {metrics}")
    
    # ==========================================================
    # [Evaluation 3] COCO Detection Metrics
    # ==========================================================
    metric_logger.synchronize_between_processes()
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    
    # 定义标准名称
    coco_metric_names = [
        "AP (IoU=0.50:0.95) area=all", "AP (IoU=0.50)      area=all", "AP (IoU=0.75)      area=all",
        "AP (IoU=0.50:0.95) area=small", "AP (IoU=0.50:0.95) area=medium", "AP (IoU=0.50:0.95) area=large",
        "AR (IoU=0.50:0.95) area=all      maxDets=1", "AR (IoU=0.50:0.95) area=all      maxDets=10", 
        "AR (IoU=0.50:0.95) area=all      maxDets=100", "AR (IoU=0.50:0.95) area=small    maxDets=100",
        "AR (IoU=0.50:0.95) area=medium   maxDets=100", "AR (IoU=0.50:0.95) area=large    maxDets=100",
    ]

    if coco_evaluator is not None:
        # --- BBox ---
        if "bbox" in iou_types and coco_evaluator.coco_eval.get("bbox") is not None:
            bbox_stats = coco_evaluator.coco_eval["bbox"].stats.tolist()
            stats["coco_eval_bbox"] = bbox_stats
            
            log_lines = ["\n[Detection] COCO BBox Results:"]
            for name, value in zip(coco_metric_names, bbox_stats):
                log_lines.append(f"  {name:<40}: {value:.3f}")
                # 简化 key 名字存入 wandb: coco/AP_all, coco/AP50_all
                short_key = name.split(')')[0].replace(' (IoU=', '_').replace(':', '-') 

            logging.info("\n".join(log_lines))

        # --- Mask ---
        if "segm" in iou_types and coco_evaluator.coco_eval.get("segm") is not None:
            mask_stats = coco_evaluator.coco_eval["segm"].stats.tolist()
            stats["coco_eval_masks"] = mask_stats
            
            log_lines = ["\n[Detection] COCO Mask Results:"]
            for name, value in zip(coco_metric_names, mask_stats):
                log_lines.append(f"  {name:<40}: {value:.3f}")

            logging.info("\n".join(log_lines))
        # if is_track and dist_utils.is_main_process():
        # stats["MOTA"] =  float(clear_group.get("MOTA", 0.0).item())

    return stats, coco_evaluator