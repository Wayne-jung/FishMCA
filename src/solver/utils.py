import math
import sys
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.amp
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
import argparse
from ..data import CocoEvaluator
from ..data.dataset import mscoco_category2label
from ..misc import MetricLogger, SmoothedValue, dist_utils, save_samples
from ..optim import ModelEMA, Warmup
from .validator import Validator, scale_boxes
import os
import cv2
from ..zoo.dfine.box_ops import box_cxcywh_to_xyxy
from PIL import Image
import matplotlib.pyplot as plt


def save_results_with_depth(results, img_paths, output_dir,  class_names=None):
    """
    将检测结果以 MOT 格式 + 深度保存到场景 txt 文件中
    results: List[Dict]  # postprocessor 输出
    img_paths: List[str] # 当前 batch 的图像路径
    depth_map: torch.Tensor  # 当前 batch 的深度图 (B,H,W)
    frame_idx: int  # 当前帧编号
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for res, img_path in zip(results, img_paths):
        frame_name = os.path.splitext(os.path.basename(img_path))[0]
        # 转换为 int
        frame_idx = int(frame_name)-1

        scne = img_path.split("/")[-3]  # 场景名
        save_path = os.path.join(output_dir, f"{scne}.txt")
        box_depth = res['depths'].cpu().numpy()
        boxes = res["boxes"].cpu().numpy()
        scores = res["scores"].cpu().numpy()
        # labels = res["labels"].cpu().numpy()

        with open(save_path, "a") as f:
            for box, score, depth in zip(boxes, scores, box_depth):
                # if score>=0.5:
                # xyxy -> xywh
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                x, y = x1, y1

                # MOT 格式: frame_id, track_id, x, y, w, h, score, -1, -1, -1, depth
                line = f"{frame_idx}, -1, {x:.2f}, {y:.2f}, {w:.2f}, {h:.2f}, {score:.4f}, 1, 1, 1, {depth:.4f}\n"
                f.write(line)


def draw_track_results(img, online_targets, depth_map=None, class_names=None, show_depth_overlay=False):
    """
    绘制跟踪框和深度信息，可选择是否叠加深度图
    Args:
        img: 输入图像，PIL.Image 或 torch.Tensor 或 np.array
        online_targets: 当前帧跟踪目标列表
        depth_map: 可选，深度图 (H, W) np.array
        class_names: 可选，类别名称列表
        show_depth_overlay: 是否在图像上叠加深度图
    Returns:
        绘制后的图像 (RGB)
    """
    # 转为 np.array
    if isinstance(img, Image.Image):
        img = np.array(img)
    elif isinstance(img, torch.Tensor):
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)

    H, W = img.shape[:2]

    # 叠加深度图
    if depth_map is not None and show_depth_overlay:
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        img = cv2.addWeighted(img, 0.6, depth_colored, 0.4, 0)

    # 随机颜色
    np.random.seed(42)
    colors = {i: (int(np.random.randint(0, 255)),
                  int(np.random.randint(0, 255)),
                  int(np.random.randint(0, 255))) for i in range(1000)}

    for target in online_targets:
        track_id = target.track_id
        x, y, w, h = target.tlwh
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(W - 1, int(x + w))
        y2 = min(H - 1, int(y + h))

        color = colors[track_id % 1000]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label_name = "person"
        score = getattr(target, "depth", 1.0)*25
        text = f"ID:{track_id} {label_name} {score:.2f}"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        ty1 = max(0, y1 - th - 4)
        cv2.rectangle(img, (x1, ty1), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img

def visualize_track_results(img_paths, track_results, depth_maps=None, output_dir="./track_vis", class_names=None, show_depth_overlay=False):
    """
    批量可视化跟踪结果和深度图
    Args:
        img_paths: 图像路径列表
        track_results: 跟踪目标列表
        depth_maps: 可选，深度图列表 (与 img_paths 对应)
        output_dir: 保存目录
        class_names: 类别名称列表（可选）
        show_depth_overlay: 是否叠加深度图到跟踪图
        save_depth_map: 是否单独保存深度图
    """
    os.makedirs(output_dir, exist_ok=True)
    depth_dir = os.path.join(output_dir, "depth_maps")
    track_dir = os.path.join(output_dir,'track_vis')
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(track_dir, exist_ok=True)
    if depth_maps is not None:
        save_depth_map = True
        depth_maps=depth_maps.cpu().detach().numpy()
    else:
        save_depth_map = False
    if save_depth_map:
        os.makedirs(depth_dir, exist_ok=True)

    for idx, (img_path, targets) in enumerate(zip(img_paths, track_results)):
        if not os.path.exists(img_path):
            print(f"Warning: 图像路径不存在 {img_path}，跳过可视化")
            continue

        img = Image.open(img_path).convert("RGB")
        depth_map = None
        if depth_maps is not None:
            depth_map = depth_maps[idx]

        # 绘制跟踪框
        drawn_img = draw_track_results(img, targets, depth_map=depth_map, class_names=class_names, show_depth_overlay=show_depth_overlay)
        scne=img_path.split('/')[-3]
        img_name = os.path.basename(img_path)
        save_path = os.path.join(track_dir ,scne, img_name)
        os.makedirs(os.path.join(track_dir ,scne), exist_ok=True)
        drawn_img = cv2.cvtColor(drawn_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, drawn_img)
        print(f"跟踪+深度可视化已保存: {save_path}")

        # 单独保存深度图
        if save_depth_map and depth_map is not None:
            depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            depth_save_path = os.path.join(depth_dir,scne, img_name)
            cv2.imwrite(depth_save_path, depth_colored)
            print(f"深度图已单独保存: {depth_save_path}")
if not hasattr(np, 'asfarray'):
    np.asfarray = np.asarray

import motmetrics as mm

def compute_mot_metrics(pred_tracks, gt_targets):
    """
    使用 motmetrics 库直接在内存中评估跟踪指标。
    兼容 List[STrack] 对象列表输入。
    """
    # 1. 初始化累加器
    acc = mm.MOTAccumulator(auto_id=True)

    for frame_idx, (preds, gt) in enumerate(zip(pred_tracks, gt_targets)):
        # --- 解析 GT (Ground Truth) ---
        gt_boxes = gt["boxes"].cpu().numpy().copy()

        if len(gt_boxes) > 0:
            gt_boxes[:, 2] = gt_boxes[:, 2] - gt_boxes[:, 0]
            gt_boxes[:, 3] = gt_boxes[:, 3] - gt_boxes[:, 1]
        else:
            gt_boxes = np.array([])
        if "track_ids" in gt:
            gt_ids = gt["track_ids"].cpu().numpy().astype(int)
        elif "obj_ids" in gt:
            gt_ids = gt["obj_ids"].cpu().numpy().astype(int)
        else:
            gt_ids = np.arange(len(gt_boxes))

        # --- 解析 Prediction (预测结果) ---
        pred_boxes = np.array([])
        pred_ids = np.array([])

        if len(preds) > 0:
            # [新增修复] 情况 A: preds 是 STrack 对象列表 (你的 ByteTrack 输出)
            # 检查第一个元素是否具有 tlbr 和 track_id 属性
            first_item = preds[0]
            if hasattr(first_item, 'tlbr') and hasattr(first_item, 'track_id'):
                # 提取 STrack 对象中的数据
                temp_boxes = []
                temp_ids = []
                for t in preds:
                    # 1. 过滤掉未激活的 (ByteTrack 中通常用 is_activated)
                    #    注意：有些版本的 STrack 用 state == TrackState.Activated
                    is_active = False
                    if hasattr(t, 'is_activated'):
                        is_active = t.is_activated
                    elif hasattr(t, 'state'): 
                        # 如果你导入了 TrackState，可以用 t.state == TrackState.Activated
                        # 这里的 1 通常代表 Activated
                        is_active = (t.state == 1) 
                    
                    # 2. 过滤掉置信度过低的 (可选，但推荐，通常 0.1 或 0.2)
                    #    TrackEval 某些配置默认会切掉低分框
                    min_score = 0.1 
                    
                    if is_active and t.score > min_score:
                        temp_boxes.append(t.tlwh)
                        temp_ids.append(t.track_id)
                pred_boxes = np.array(temp_boxes)
                pred_ids = np.array(temp_ids).astype(int)
            
            # 情况 B: preds 已经是数值数组 (例如 [[x1,y1,x2,y2,id], ...])
            else:
                if isinstance(preds, torch.Tensor):
                    preds_arr = preds.cpu().numpy()
                else:
                    preds_arr = np.array(preds)
                
                # 确保是 2D 数组且列数足够
                if preds_arr.ndim == 2 and preds_arr.shape[1] >= 5:
                    pred_boxes = preds_arr[:, :4]
                    pred_ids = preds_arr[:, 4].astype(int)

        # --- 计算距离矩阵 (IoU) ---
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            # motmetrics 计算的是距离 (1 - IoU)，max_iou 是匹配阈值
            dists = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        else:
            dists = np.array([])

        # --- 更新累加器 ---
        acc.update(
            gt_ids,      # Ground Truth IDs
            pred_ids,    # Hypothesis (Prediction) IDs
            dists        # Distance matrix
        )

    # 2. 计算指标
    mh = mm.metrics.create()
    metrics_list = ['mota' ,'idf1', 'idp', 'idr', 'recall', 'precision', 'num_switches']
    summary = mh.compute(acc, metrics=metrics_list, name='acc')
    
# 3. 美化打印 (使用 rich 库)
    # ==========================================================
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()
        
        # 创建表格对象
        table = Table(title="🚀 MOT Evaluation Results", box=box.ROUNDED, header_style="bold cyan")
        
        # 定义需要展示的指标及其格式
        # Key: dataframe中的列名, Label: 显示名称, Fmt: 格式化字符串
        columns_config = [
            ("mota", "MOTA", "{:.1%}"),
            ("idf1", "IDF1", "{:.1%}"),
            ("recall", "Rcll", "{:.1%}"),
            ("precision", "Prcn", "{:.1%}"),
            ("num_switches", "ID Sw.", "{:d}"),
            ("idp", "IDP", "{:.1%}"),
            ("idr", "IDR", "{:.1%}"),
        ]

        # 添加表头
        for _, label, _ in columns_config:
            table.add_column(label, justify="center")

        # 提取数据并填充表格
        # summary 是一个 DataFrame，通常只有一行 (index='acc')
        row_data = []
        for key, _, fmt in columns_config:
            val = summary.loc['acc', key]
            # 根据格式化字符串处理数值
            formatted_val = fmt.format(val)
            
            # 给重要指标添加颜色高亮
            if key in ['mota', 'idf1']:
                formatted_val = f"[bold green]{formatted_val}[/bold green]"
            elif key == 'num_switches':
                formatted_val = f"[red]{formatted_val}[/red]"
                
            row_data.append(formatted_val)
        
        table.add_row(*row_data)

        print("\n")
        console.print(table)
        print("\n")

    except ImportError:
        # 如果没有安装 rich，回退到 Pandas 的原生 markdown 格式，也比 motmetrics 默认的好看
        print("\n" + "="*80)
        print("MOT Evaluation Results (Install 'rich' for better visualization):")
        # 筛选关键列并重命名
        rename_map = {
            'mota': 'MOTA', 'idf1': 'IDF1', 'recall': 'Recall', 
            'precision': 'Prec', 'num_switches': 'ID Sw.', 
            'idp': 'IDP', 'idr': 'IDR'
        }
        # 仅显示存在的列
        cols_to_show = [c for c in rename_map.keys() if c in summary.columns]
        display_df = summary[cols_to_show].rename(columns=rename_map)
        
        # 格式化百分比
        format_dict = {
            'MOTA': '{:.1%}', 'IDF1': '{:.1%}', 'Recall': '{:.1%}', 
            'Prec': '{:.1%}', 'IDP': '{:.1%}', 'IDR': '{:.1%}'
        }
        print(display_df.to_string(formatters={k: v.format for k, v in format_dict.items() if k in display_df.columns}))
        print("="*80 + "\n")

    return summary.to_dict('records')[0]

def track(outputs, tracker, info_imgs):
    online_targets=[]
    for i, dets in enumerate(outputs):
        info_img=np.array(info_imgs[i])
        # np.array(img_size[i])
        online_target = tracker.update( dets['bytesort_input'].cpu(), info_img, info_img)
        online_targets.append(online_target)
        
        # else:
        #     # 如果没有检测结果，传入空列表
        #     online_targets = tracker.update([], info_imgs, img_size)

    return online_targets


def get_box_depth(targets, pred_depth):

    for i in range(len(targets)):
        # pred_depth[i] shape: [H, W] or [1, H, W]
        depth_map = pred_depth[i]
        if depth_map.ndim == 3:
            depth_map = depth_map.squeeze(0)  # 去掉通道维度 if [1, H, W]

        depth_list = []
        boxes = box_cxcywh_to_xyxy(targets[i]["boxes"])
        for t, box in enumerate(boxes):  # 应该是 [x1, y1, x2, y2]
            x1, y1, x2, y2 = [coord for coord in box]

            # 边界合法性检查，防止越界
            h, w = depth_map.shape
            x1 = int(max(0, min(x1*w, w - 1)))
            x2 = int(max(0, min(x2*w, w)))
            y1 = int(max(0, min(y1*h, h - 1)))
            y2 = int(max(0, min(y2*h, h)))

            if x2 <= x1 or y2 <= y1:
                continue  # 跳过无效 box

            # 获取边界框区域的深度
            region = depth_map[y1:y2, x1:x2]
            if region.numel() == 0:
                continue  # 空区域，跳过

            # 计算深度均值（也可以选择中值）
            mean_depth = region.mean()

            # 更新 target 中的 depth 值
            depth_list.append(mean_depth.item())
        depth_list=torch.tensor(depth_list, device=depth_map.device)
        targets[i]["depth"] = depth_list
        targets[i]['bytesort_input'] =torch.cat([targets[i]['boxes'], torch.ones_like(depth_list).unsqueeze(1), depth_list.unsqueeze(1)],dim=1)
    return targets



from collections import defaultdict
def visualize_depth_tracks(all_online_targets, save_path="depth_tracks.png"):
    """
    可视化跟踪实例的边界框中心 x 坐标 vs 深度值 (2D 折线图)
    
    Args:
        all_online_targets: list，每一帧的 online_targets（BYTETracker.update 的输出）
        save_path: 保存图片路径
    """
    # 保存每个 track_id 的轨迹点
    track_data = defaultdict(list)  # {track_id: [(center_x, depth), ...]}
        
    track_data = defaultdict(list)  # {track_id: [(frame_idx, depth), ...]}
    frame_idx=0
    for batch_targets in enumerate(all_online_targets):
        for frame_targets in batch_targets:
            frame_idx+=1
            for t in frame_targets:
                depth = getattr(t, "depth", None) 
                if depth is None:
                    continue
                track_data[t.track_id].append((frame_idx, depth))

    plt.figure(figsize=(10, 6))
    for tid, points in track_data.items():
        points = sorted(points, key=lambda p: p[0])  # 按帧号排序
        xs = [p[0] for p in points]  # frame_id
        ys = [p[1] for p in points]  # depth
        plt.plot(xs, ys, marker='o', markersize=2,label=f'ID {tid}')


    plt.xlabel("Bounding Box Center X")
    plt.ylabel("Depth")
    plt.title("Track Instance Depth Visualization")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 可视化保存到 {save_path}")

def visualize_depth_video(all_online_targets, save_path="./depth_scatter.mp4", fps=10):
    """
    生成视频，展示每一帧中目标中心 x 与深度值的散点图, 可视化在不同帧中深度值的变化轨迹

    Args:
        all_online_targets: list，每一帧的 online_targets（BYTETracker.update 的输出）
        save_path: 保存视频路径
        fps: 输出视频的帧率
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 先统计帧数
    num_frames = len(all_online_targets)

    # 画布大小
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.close(fig)

    # 用 cv2.VideoWriter 输出视频
    writer = None

    for frame_idx, batch_targets in enumerate(all_online_targets):
        ax.clear()
        ax.set_xlim(0, 1920)   # 这里假设图像宽度 1920，你可以改成实际大小
        ax.set_ylim(0, 50.0)    # 假设深度值范围 [0,1]，可根据实际改
        ax.set_xlabel("Bounding Box Center X")
        ax.set_ylabel("Depth")
        ax.set_title(f"Frame {frame_idx}")

        # 在当前帧画点
        for frame_targets in batch_targets:
            for t in frame_targets:
                depth = getattr(t, "depth", None)
                if depth is None:
                    continue
                x, y, w, h = t.tlwh
                center_x = x + w / 2
                ax.scatter(center_x, depth, label=f"ID {t.track_id}", s=40)

        # 保存 matplotlib 图到 numpy 数组
        fig.canvas.draw()
        # 正确的转换步骤
        img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        # 先转为4通道格式
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # 去掉Alpha通道，并转换为RGB格式
        img = img[:, :, 1:]  # 保留RGB通道，丢弃Alpha

        # 初始化 VideoWriter
        if writer is None:
            h, w, _ = img.shape
            writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        # 转换颜色 (matplotlib RGB → OpenCV BGR)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(img_bgr)

    if writer is not None:
        writer.release()

    print(f"🎥 深度可视化视频已保存到 {save_path}")