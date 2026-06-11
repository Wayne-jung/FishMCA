
import os.path as osp

from ByteTrack.tools.demo_track_frome_detdtxt import Predictor, image_demo   # 假设你的代码保存成 your_demo_script.py
from DanceTrack.TrackEval.trackeval import Evaluator  # pip install trackeval
import time  # 用于计时
gt_root = r"/root/autodl-tmp/QuadTrack/test"
SEQMAP_FOLDER = '../QuadTrack/seqmaps'

def evaluate_dancetrack( result_root):
    """使用 TrackEval 计算指标"""
    from DanceTrack.TrackEval import trackeval
    eval_config = {
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 4,
        'PRINT_RESULTS': True,
        'PRINT_CONFIG': True,
        'TIME_PROGRESS': True,
        'DISPLAY_LESS_PROGRESS': True,
    }
    dataset_config = {
        'GT_FOLDER': gt_root,         # dancetrack/val
        'TRACKERS_FOLDER': result_root, # 保存的tracking结果
        'TRACKER_SUB_FOLDER':'',
        'SEQMAP_FILE': None,
        'SEQ_INFO': None,
        'BENCHMARK': 'QuadTrack',
        'SEQMAP_FOLDER': SEQMAP_FOLDER,
        'SPLIT_TO_EVAL': 'test'
    }

    metrics_config = {'METRICS': ['HOTA','CLEAR','Identity']}
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
    return output_res, output_msg



    