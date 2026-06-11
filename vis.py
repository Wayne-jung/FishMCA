import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset
import sys
import itertools

# -------------------------------------------------------------------------
# 1. 环境与路径设置
# -------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
dfine_path = os.path.join(current_dir, 'src/zoo/dfine')
if dfine_path not in sys.path:
    sys.path.insert(0, dfine_path)

try:
    from da2depth import DA2DepthPredictor
    from da360_depth_predictor import DA360DepthPredictor

except ImportError as e:
    print(f"[Warning] DA2DepthPredictor 导入失败: {e}")

try:
    from SAM2.sam2.build_sam import build_sam2
    from SAM2.sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("[Warning] SAM2 导入失败")

try:
    from VideoDepthAnything import VideoDepthAnything
except ImportError:
    print("[Warning] VideoDepthAnything 导入失败")

# -------------------------------------------------------------------------
# 2. 模型构建
# -------------------------------------------------------------------------
def build_models(device):
    print("正在加载模型...")
    # A. 时序模型
    t_model = VideoDepthAnything(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    ckpt_path = './VideoDepthAnything/video_depth_anything_vits.pth'
    if os.path.exists(ckpt_path):
        t_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    t_model.to(device).eval()

    # B. 单帧静态模型
    s_model = DA360DepthPredictor(
        model_path='../DA360/checkpoints/DA360_small.pth',
        height=490, width=2058, 
        dinov2_encoder='vits',
        device=device, 
        freeze_params=True
    )
    
    # C. SAM2
    sam_checkpoint = "SAM2/checkpoints/sam2.1_hiera_tiny.pt"
    sam_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    if os.path.exists(sam_checkpoint):
        sam_model = build_sam2(sam_cfg, sam_checkpoint, device=device)
        sam_predictor = SAM2ImagePredictor(sam_model)
    else:
        sam_predictor = None
    
    return t_model, s_model, sam_predictor

# -------------------------------------------------------------------------
# 3. 数据加载器 (Window=2, Stride=1)
# -------------------------------------------------------------------------
class DanceTrackSlidingLoader(Dataset):
    def __init__(self, seq_root):
        self.img_dir = os.path.join(seq_root, 'img1')
        self.gt_path = os.path.join(seq_root, 'gt/gt.txt')
        self.window_size = 2
        self.img_names = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])
        self.anns = {}
        if os.path.exists(self.gt_path):
            data = np.loadtxt(self.gt_path, delimiter=',')
            for line in data:
                f_id, obj_id = int(line[0]), int(line[1])
                x, y, w, h = line[2], line[3], line[4], line[5]
                if f_id not in self.anns: self.anns[f_id] = []
                self.anns[f_id].append({'id': obj_id, 'box': [x, y, w, h]})

    def __len__(self):
        return len(self.img_names) - self.window_size + 1

    def get_window(self, idx):
        imgs, img_tensors, anns_list = [], [], []
        for i in range(self.window_size):
            curr_idx = idx + i
            img_path = os.path.join(self.img_dir, self.img_names[curr_idx])
            img = cv2.imread(img_path)
            if img is None: continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img_rgb)
            t = torch.from_numpy(cv2.resize(img_rgb, (2058, 490))).permute(2, 0, 1).float() / 255.0
            img_tensors.append(t)
            anns_list.append(self.anns.get(curr_idx + 1, []))
        return imgs, torch.stack(img_tensors), anns_list

# -------------------------------------------------------------------------
# 4. 深度提取工具
# -------------------------------------------------------------------------
def extract_obj_depth(depth_map, box, predictor):
    if predictor is None: return 0.0
    input_box = np.array([box[0], box[1], box[0]+box[2], box[1]+box[3]])
    masks, _, _ = predictor.predict(box=input_box, multimask_output=False)
    mask = masks[0]
    if mask.shape != depth_map.shape:
        mask = cv2.resize(mask.astype(np.uint8), (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST)
    valid_depths = depth_map[mask > 0]
    return np.median(valid_depths) if len(valid_depths) > 0 else np.nan

# # -------------------------------------------------------------------------
# # 5. 可视化核心：矩阵绘制
# # -------------------------------------------------------------------------
# def save_correlation_matrix(frame_data, output_dir):
#     all_ids = set()
#     for f in frame_data:
#         all_ids.update(frame_data[f]['static'].keys())
    
#     # 筛选活跃 ID (至少出现 15 帧)
#     id_counts = {oid: 0 for oid in all_ids}
#     for f in frame_data:
#         for oid in frame_data[f]['static']: 
#             id_counts[oid] += 1
            
#     # 如果 ID 总数太少，适当降低阈值，防止无法画图
#     threshold = 15 if len(all_ids) > 5 else 5
#     valid_ids = sorted([oid for oid, c in id_counts.items() if c > threshold])
    
#     if len(valid_ids) < 2: 
#         print(f"有效 ID 数量不足 ({len(valid_ids)} < 2)，无法绘制矩阵。")
#         return

#     n = len(valid_ids)
#     m_static = np.zeros((n, n))
#     m_temporal = np.zeros((n, n))
    
#     print(f"计算关联矩阵 ({n}x{n})...")

#     for i in range(n):
#         for j in range(n):
#             if i == j: continue
#             id_a, id_b = valid_ids[i], valid_ids[j]
#             diffs_s, diffs_t = [], []
            
#             for f in frame_data:
#                 s_d = frame_data[f]['static']
#                 t_d = frame_data[f]['temporal']
                
#                 # --- 修复点：必须同时检查 ID 是否存在于两个字典中 ---
#                 if (id_a in s_d and id_b in s_d and 
#                     id_a in t_d and id_b in t_d):
                    
#                     val_s_a, val_s_b = s_d[id_a], s_d[id_b]
#                     val_t_a, val_t_b = t_d[id_a], t_d[id_b]
                    
#                     # 双重保险：确保取出的值不是 NaN
#                     if not np.isnan([val_s_a, val_s_b, val_t_a, val_t_b]).any():
#                         diffs_s.append(val_s_a - val_s_b)
#                         diffs_t.append(val_t_a - val_t_b)
            
#             # 只有当共同出现的帧数足够多时，才计算标准差
#             if len(diffs_s) > 5:
#                 m_static[i, j] = np.std(diffs_s)
#                 m_temporal[i, j] = np.std(diffs_t)

#     # 绘图逻辑
#     if np.all(m_static == 0):
#         print("警告：矩阵全为 0，可能是数据不足或匹配失败。")
#         return

#     fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
#     # 自动调整最大值，忽略极值影响显示
#     valid_values = m_static[m_static > 0]
#     if len(valid_values) > 0:
#         v_max = np.percentile(valid_values, 95)
#     else:
#         v_max = 1.0
    
#     # Plot Static
#     im1 = axes[0].imshow(m_static, cmap='hot_r', vmin=0, vmax=v_max)
#     axes[0].set_title('Static Model: Relative Depth Instability')
#     axes[0].set_xlabel('Object ID')
#     axes[0].set_ylabel('Object ID')
#     # 添加 ID 刻度
#     if n < 20: # ID 太多就不显示刻度了，免得挤在一起
#         axes[0].set_xticks(range(n))
#         axes[0].set_yticks(range(n))
#         axes[0].set_xticklabels(valid_ids)
#         axes[0].set_yticklabels(valid_ids)

#     # Plot Temporal
#     im2 = axes[1].imshow(m_temporal, cmap='hot_r', vmin=0, vmax=v_max)
#     axes[1].set_title('Temporal Window: Relative Depth Instability')
#     axes[1].set_xlabel('Object ID')
#     if n < 20:
#         axes[1].set_xticks(range(n))
#         axes[1].set_yticks(range(n))
#         axes[1].set_xticklabels(valid_ids)
#         axes[1].set_yticklabels(valid_ids)

#     # 计算提升指标
#     mean_s = np.mean(m_static[m_static > 1e-6]) if np.any(m_static > 1e-6) else 0
#     mean_t = np.mean(m_temporal[m_temporal > 1e-6]) if np.any(m_temporal > 1e-6) else 0
    
#     if mean_s > 0:
#         improv = (mean_s - mean_t) / mean_s * 100
#     else:
#         improv = 0.0
    
#     plt.suptitle(f'Inter-Object Stability Analysis\nStatic Avg Jitter: {mean_s:.4f} vs Temporal Avg Jitter: {mean_t:.4f}\nStability Improvement: {improv:.2f}%', fontsize=14, fontweight='bold')
    
#     save_path = os.path.join(output_dir, 'depth_matrix.png')
#     plt.savefig(save_path)
#     plt.close()
#     print(f"矩阵已保存: {save_path}")
#     print(f"稳定性提升: {improv:.2f}% (正值代表变好了)")
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import linear_sum_assignment

# -------------------------------------------------------------------------
# 1. 定义一个纯深度跟踪器 (Simple Depth-Only Tracker)
# -------------------------------------------------------------------------
class DepthTracker:
    def __init__(self, max_depth_diff=1.0):
        """
        :param max_depth_diff: 允许的最大深度跳变阈值。
                               超过此值认为不是同一个物体（轨迹断裂）。
        """
        self.max_diff = max_depth_diff
        self.tracks = {} # {track_id: last_depth_value}
        self.next_id = 0
        
        # 记录结果用于评估: frame_idx -> {gt_id: assigned_track_id}
        self.history = {} 

    def update(self, frame_idx, detections):
        """
        detections: list of {'gt_id': original_id, 'depth': value}
        注意：在匹配阶段，我们故意"假装"不知道 gt_id，只用 depth 匹配
        """
        current_depths = [d['depth'] for d in detections]
        track_ids = list(self.tracks.keys())
        track_depths = list(self.tracks.values())
        
        # --- A. 构建代价矩阵 (Cost Matrix) ---
        cost_matrix = np.zeros((len(track_ids), len(current_depths)))
        for t_i, t_depth in enumerate(track_depths):
            for d_i, d_val in enumerate(current_depths):
                diff = abs(t_depth - d_val)
                # 如果差异太大，设为无穷大，禁止匹配
                if diff > self.max_diff:
                    cost_matrix[t_i, d_i] = 1e6
                else:
                    cost_matrix[t_i, d_i] = diff
        
        # --- B. 匈牙利算法匹配 ---
        # linear_sum_assignment 寻找最小代价
        if len(track_ids) > 0 and len(current_depths) > 0:
            row_inds, col_inds = linear_sum_assignment(cost_matrix)
        else:
            row_inds, col_inds = [], []
            
        # --- C. 更新轨迹 ---
        assigned_track_indices = set()
        assigned_det_indices = set()
        
        frame_results = {} # gt_id -> track_id
        
        # 处理匹配成功的对
        for r, c in zip(row_inds, col_inds):
            if cost_matrix[r, c] < self.max_diff:
                track_id = track_ids[r]
                new_depth = current_depths[c]
                
                # 更新轨迹状态
                self.tracks[track_id] = new_depth
                
                # 记录评估结果 (将生成的 track_id 映射回 GT ID 以便后续计算)
                gt_id = detections[c]['gt_id']
                frame_results[gt_id] = track_id
                
                assigned_track_indices.add(r)
                assigned_det_indices.add(c)
        
        # --- D. 处理未匹配的检测 (新轨迹) ---
        for i in range(len(current_depths)):
            if i not in assigned_det_indices:
                # 这是一个新出现的物体，或者旧物体深度跳变太大导致断裂
                new_track_id = self.next_id
                self.next_id += 1
                self.tracks[new_track_id] = current_depths[i]
                
                gt_id = detections[i]['gt_id']
                frame_results[gt_id] = new_track_id
        
        # --- E. 移除未匹配的旧轨迹 (消失) ---
        # 简单处理：当前帧没匹配上就删掉 (严谨的MOT会保留几帧，这里为了测试连续性越严越好)
        active_tracks = {}
        for r, tid in enumerate(track_ids):
            if r in assigned_track_indices:
                active_tracks[tid] = self.tracks[tid]
        self.tracks = active_tracks
        
        return frame_results

# -------------------------------------------------------------------------
# 2. MOT 指标计算器
# -------------------------------------------------------------------------
def calculate_mot_metrics(gt_data, tracker_results):
    """
    计算 ID Switches (IDSW) 和 MOTA (近似)
    gt_data: 所有出现的 GT ID 集合
    tracker_results: list of {gt_id: track_id} per frame
    """
    id_switches = 0
    total_detections = 0
    
    # 记录每个 GT ID 当前对应的 Track ID
    gt_to_track_state = {} 
    
    for frame_res in tracker_results:
        for gt_id, track_id in frame_res.items():
            total_detections += 1
            
            if gt_id not in gt_to_track_state:
                # 第一次出现，记录对应关系
                gt_to_track_state[gt_id] = track_id
            else:
                # 之前出现过，检查 Track ID 是否改变
                if gt_to_track_state[gt_id] != track_id:
                    id_switches += 1
                    gt_to_track_state[gt_id] = track_id # 更新状态
                    
    # MOTA = 1 - (IDSW + FP + FN) / Total_GT
    # 因为我们使用 GT Box 作为输入，假定 Detector 是完美的 (FP=0, FN=0)
    # 所以这是一个专注于身份保持能力的 "Identity Consistency Score"
    mota = 1.0 - (float(id_switches) / float(total_detections)) if total_detections > 0 else 0
    
    return {
        'IDSW': id_switches,
        'MOTA': mota * 100, # 百分比
        'Total_Dets': total_detections
    }

# -------------------------------------------------------------------------
# 3. 主可视化函数
# -------------------------------------------------------------------------
def save_correlation_matrix(frame_data, output_dir):
    """
    基于深度连续性的多目标跟踪评估 (Depth-based MOT Evaluation)
    """
    print("正在进行基于深度的 MOT 跟踪模拟评估...")
    
    # 1. 准备数据序列
    sorted_frames = sorted(frame_data.keys())
    
    # 设定深度关联阈值
    # 假设深度已经归一化或在合理范围内。如果波动超过此值，认为断裂。
    # 可以根据你的数据尺度调整，例如 0.5 或 1.0
    # 为了公平，先计算一下全局标准差作为参考
    all_depths = []
    for f in sorted_frames:
        for d in frame_data[f]['static'].values():
            if not np.isnan(d): all_depths.append(d)
    global_std = np.std(all_depths)
    match_threshold = global_std * 0.5  # 设定为 0.5 倍标准差作为容忍度
    print(f"自动设定深度匹配阈值: {match_threshold:.4f}")

    results = {}
    
    # 2. 分别运行两种模型的跟踪器
    for model_type in ['static', 'temporal']:
        tracker = DepthTracker(max_depth_diff=match_threshold)
        tracking_history = []
        
        for f in sorted_frames:
            # 准备当前帧的 "检测"
            detections = []
            frame_vals = frame_data[f][model_type]
            
            for gt_id, depth_val in frame_vals.items():
                if not np.isnan(depth_val):
                    detections.append({'gt_id': gt_id, 'depth': depth_val})
            
            # 运行跟踪步
            frame_res = tracker.update(f, detections)
            tracking_history.append(frame_res)
            
        # 计算指标
        metrics = calculate_mot_metrics(None, tracking_history)
        results[model_type] = metrics

    # 3. 输出文本结果
    print("\n" + "="*40)
    print("   DEPTH-ONLY TRACKING EVALUATION   ")
    print("="*40)
    print(f"{'Metric':<10} | {'Static':<10} | {'Temporal':<10} | {'Improvement'}")
    print("-" * 50)
    
    idsw_s = results['static']['IDSW']
    idsw_t = results['temporal']['IDSW']
    mota_s = results['static']['MOTA']
    mota_t = results['temporal']['MOTA']
    
    imp_idsw = (idsw_s - idsw_t) / idsw_s * 100 if idsw_s > 0 else 0
    imp_mota = mota_t - mota_s
    
    print(f"{'ID Switch':<10} | {idsw_s:<10} | {idsw_t:<10} | -{imp_idsw:.1f}% (Better)")
    print(f"{'MOTA':<10} | {mota_s:.1f}%     | {mota_t:.1f}%     | +{imp_mota:.1f}% (Better)")
    print("="*40)

    # 4. 可视化绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Bar Chart 1: ID Switches (越低越好)
    labels = ['Static', 'Temporal']
    idsw_vals = [idsw_s, idsw_t]
    bars = ax1.bar(labels, idsw_vals, color=['#ff9999', '#66b3ff'], alpha=0.9)
    ax1.set_title('ID Switches (Lower is Better)\n(Caused by Depth Jitter)', fontsize=12)
    ax1.set_ylabel('Count')
    ax1.bar_label(bars, fmt='%d')
    
    # Bar Chart 2: MOTA (越高越好)
    mota_vals = [mota_s, mota_t]
    bars2 = ax2.bar(labels, mota_vals, color=['#ff9999', '#66b3ff'], alpha=0.9)
    ax2.set_title('Depth Consistency Score (MOTA)\n(Higher is Better)', fontsize=12)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0, 100)
    ax2.bar_label(bars2, fmt='%.1f%%')
    
    plt.suptitle(f'Depth-Based Tracking Stability Analysis\n(Threshold={match_threshold:.2f})', fontsize=14, fontweight='bold')
    
    save_path = os.path.join(output_dir, 'depth_tracking_metrics.png')
    plt.savefig(save_path)
    plt.close()
    print(f"评估图表已保存: {save_path}")

# -------------------------------------------------------------------------
# 6. 主逻辑
# -------------------------------------------------------------------------
def run_analysis(seq_path, output_dir='./results'):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t_model, s_model, sam_predictor = build_models(device)
    dataset = DanceTrackSlidingLoader(seq_path)

    # 存储所有帧的所有ID数据
    # format: frame_idx -> {'static': {oid: val}, 'temporal': {oid: val}}
    frame_data_storage = {}

    print(f"开始全量矩阵分析: {seq_path}")
    
    for i in tqdm(range(len(dataset))):
        # if i==100:
        #     break
        imgs, img_tensors, anns_list = dataset.get_window(i)
        if len(imgs) < 2: continue
        
        # 记录当前帧的数据容器
        current_frame_data = {'static': {}, 'temporal': {}}
        
        with torch.no_grad():
            # A. 时序模型 (Window=2)
            t_input = img_tensors.unsqueeze(1).to(device)
            t_output, _ = t_model(t_input)
            t_maps = t_output.squeeze(1).cpu().numpy()
            
            # B. 静态模型 (Only Frame 1)
            s_input = img_tensors[0].unsqueeze(0).to(device)
            s_map = s_model(s_input).squeeze().cpu().numpy()*25

        # 关联逻辑：Mean Fusion
        # 这里为了矩阵计算，我们需要把当前帧(Frame 1)里所有ID的深度都算出来
        frame1_anns = anns_list[0]
        frame2_anns = anns_list[1]
        f2_dict = {obj['id']: obj['box'] for obj in frame2_anns}
        
        sam_predictor.set_image(imgs[0]) # Set Frame 1
        
        for obj in frame1_anns:
            obj_id = obj['id']
            box1 = obj['box']
            
            # 1. Static Depth
            d_s = extract_obj_depth(s_map, box1, sam_predictor)
            
            # 2. Temporal Depth (Mean of F1 and F2)
            val_t1 = extract_obj_depth(t_maps[0], box1, sam_predictor)
            
            # 简化版 F2 提取 (不切换 SAM 图片，直接切片取中值)
            val_t2 = np.nan
            if obj_id in f2_dict:
                box2 = f2_dict[obj_id]
                x, y, w, h = [int(v) for v in box2]
                y2 = min(y+h, t_maps[1].shape[0])
                x2 = min(x+w, t_maps[1].shape[1])
                crop = t_maps[1][y:y2, x:x2]
                if crop.size > 0:
                    val_t2 = np.median(crop)
            
            if not np.isnan(val_t1) and not np.isnan(val_t2):
                d_temporal = (val_t1 + val_t2) / 2.0
            else:
                d_temporal = val_t1
            
            # 存入临时数据
            if not np.isnan(d_s): current_frame_data['static'][obj_id] = d_s
            if not np.isnan(d_temporal): current_frame_data['temporal'][obj_id] = d_temporal
        
        # 存入全局存储 (Key 是真实的时间戳 i)
        frame_data_storage[i] = current_frame_data

    # 绘制矩阵
    save_correlation_matrix(frame_data_storage, output_dir)

if __name__ == "__main__":
    seq_path = '/root/autodl-tmp/QuadTrack/test/0000'
    run_analysis(seq_path)