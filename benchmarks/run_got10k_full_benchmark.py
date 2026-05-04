import os
import cv2
import torch
import numpy as np
import pandas as pd
import motmetrics as mm
from tqdm import tqdm
import sys
import yaml
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.lightglue_tracker import LightGlueTracker
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.bot_sort import BOTSORT

class MockResults:
    def __init__(self, xywh, conf, cls):
        self.xywh = np.array(xywh).reshape(-1, 4) if len(xywh) > 0 else np.zeros((0,4))
        self.conf = np.array(conf).reshape(-1) if len(conf) > 0 else np.zeros((0,))
        self.cls = np.array(cls).reshape(-1) if len(cls) > 0 else np.zeros((0,))
    
    def __len__(self):
        return len(self.xywh)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return MockResults(self.xywh[idx:idx+1], self.conf[idx:idx+1], self.cls[idx:idx+1])
        return MockResults(self.xywh[idx], self.conf[idx], self.cls[idx])

def load_tracker_args(yaml_path):
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    defaults = {
        'track_high_thresh': 0.5,
        'track_low_thresh': 0.1,
        'new_track_thresh': 0.6,
        'track_buffer': 30,
        'match_thresh': 0.8,
        'fuse_score': True,
        'gmc_method': 'none',
        'proximity_thresh': 0.5,
        'appearance_thresh': 0.25,
        'with_reid': False,
        'model': 'auto'
    }
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
    return SimpleNamespace(**cfg)

def evaluate_mot(gt_dir, res_dir, sequences, metrics=['mota', 'num_switches', 'idp', 'idr', 'idf1', 'motp']):
    mh = mm.metrics.create()
    accs = []
    names = []
    
    for seq in sequences:
        gt_file = os.path.join(gt_dir, seq, 'groundtruth.txt')
        res_file = os.path.join(res_dir, f"{seq}.txt")
        
        if not os.path.exists(res_file):
            continue
            
        # GOT-10k groundtruth is usually frame-by-frame starting at frame 1
        # format: x,y,w,h
        gt_raw = np.loadtxt(gt_file, delimiter=',')
        
        # Convert GOT-10k GT to MOT15 format for motmetrics
        # MOT15 columns required by motmetrics: X, Y, Width, Height, Confidence, ClassId, Visibility
        # Index must be (FrameId, Id). For SOT dataset like GOT-10k, id is always 1.
        gt_df = pd.DataFrame(gt_raw, columns=['X', 'Y', 'Width', 'Height'])
        gt_df['FrameId'] = np.arange(1, len(gt_raw) + 1)
        gt_df['Id'] = 1
        gt_df['Confidence'] = 1
        gt_df['ClassId'] = -1
        gt_df['Visibility'] = -1
        gt_df.set_index(['FrameId', 'Id'], inplace=True)
        
        # Load results (already in MOT15 format from our tracker loop)
        ts = mm.io.loadtxt(res_file, fmt='mot15-2D')
        
        # Create accumulator
        acc = mm.utils.compare_to_groundtruth(gt_df, ts, 'iou', distth=0.5)
        accs.append(acc)
        names.append(seq)
        
    if not accs:
        return None
    summary = mh.compute_many(accs, metrics=metrics, names=names, generate_overall=True)
    return summary

def main():
    DATASET_ROOT = "/mnt/hd2/ArtigoLightglue/Datasets_externos/val"
    RESULTS_ROOT = "/home/servidor/ArtigoLightglue/mass_results/got10k_full"
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    
    sequences = sorted([d for d in os.listdir(DATASET_ROOT) if d.startswith('GOT-10k_Val')])
    if not sequences:
        print("Nenhuma sequencia encontrada no GOT-10k.")
        return
        
    byte_args = load_tracker_args("/mnt/hd2/ArtigoLightglue/configs/bytetrack.yaml")
    bot_args = load_tracker_args("/mnt/hd2/ArtigoLightglue/configs/botsort.yaml")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # GOT-10k base is 10 FPS
    strides = [1, 2, 5] # 10 FPS, 5 FPS, 2 FPS
    stats = []
    
    for stride in strides:
        fps_simulated = 10 // stride
        print(f"\n--- Running GOT-10k Benchmark at {fps_simulated} FPS (Stride={stride}) ---")
        
        byte_dir = os.path.join(RESULTS_ROOT, f'bytetrack_s{stride}')
        bot_dir = os.path.join(RESULTS_ROOT, f'botsort_s{stride}')
        lg_dir = os.path.join(RESULTS_ROOT, f'lightglue_s{stride}')
        os.makedirs(byte_dir, exist_ok=True)
        os.makedirs(bot_dir, exist_ok=True)
        os.makedirs(lg_dir, exist_ok=True)
        
        for seq_name in tqdm(sequences, desc="Sequences"):
            byte_res_path = os.path.join(byte_dir, f"{seq_name}.txt")
            bot_res_path = os.path.join(bot_dir, f"{seq_name}.txt")
            lg_res_path = os.path.join(lg_dir, f"{seq_name}.txt")
            
            if os.path.exists(byte_res_path) and os.path.exists(bot_res_path) and os.path.exists(lg_res_path):
                continue
                
            seq_dir = os.path.join(DATASET_ROOT, seq_name)
            img_files = sorted([f for f in os.listdir(seq_dir) if f.endswith('.jpg')])
            gt_path = os.path.join(seq_dir, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',')

            byte_tracker = BYTETracker(byte_args, frame_rate=fps_simulated)
            bot_tracker = BOTSORT(bot_args, frame_rate=fps_simulated)
            
            # For GOT-10k, we use generic mode (no roadside constraints)
            lg_tracker = LightGlueTracker(
                device=device,
                accept_th=1.2,
                motion_weight=0.3,
                max_age=30, 
                roadside_mode=False,
                use_cmc=True # GOT-10k has moving cameras
            )
            
            byte_results = []
            bot_results = []
            lg_results = []
            
            for i in range(0, len(img_files), stride):
                if i >= len(gt): break
                
                img_path = os.path.join(seq_dir, img_files[i])
                frame = cv2.imread(img_path)
                if frame is None: break
                
                # GOT-10k is SOT, but we treat as MOT with 1 object
                x_tl, y_tl, w, h = gt[i]
                
                bbox_lg = np.array([[x_tl, y_tl, x_tl + w, y_tl + h]])
                bbox_yolo = np.array([[x_tl + w/2, y_tl + h/2, w, h]])
                results = MockResults(bbox_yolo, [1.0], [0])
                
                gt_frame_id = i + 1
                
                # ByteTrack
                byte_res = byte_tracker.update(results, frame)
                for r in byte_res:
                    x1, y1, x2, y2, tid = r[:5]
                    byte_results.append(f"{gt_frame_id},{int(tid)},{x1},{y1},{x2-x1},{y2-y1},1,-1,-1,-1\n")
                    
                # BoTSORT
                bot_res = bot_tracker.update(results, frame)
                for r in bot_res:
                    x1, y1, x2, y2, tid = r[:5]
                    bot_results.append(f"{gt_frame_id},{int(tid)},{x1},{y1},{x2-x1},{y2-y1},1,-1,-1,-1\n")
                    
                # LightGlue
                lg_res = lg_tracker.update(bbox_lg, frame, [0], i, stride=stride)
                for r in lg_res:
                    x1, y1, x2, y2 = r['bbox']
                    tid = r['id']
                    lg_results.append(f"{gt_frame_id},{int(tid)},{x1},{y1},{x2-x1},{y2-y1},1,-1,-1,-1\n")
                    
            with open(os.path.join(byte_dir, f"{seq_name}.txt"), 'w') as f:
                f.writelines(byte_results)
            with open(os.path.join(bot_dir, f"{seq_name}.txt"), 'w') as f:
                f.writelines(bot_results)
            with open(os.path.join(lg_dir, f"{seq_name}.txt"), 'w') as f:
                f.writelines(lg_results)
                
        # Evaluate Stride
        byte_sum = evaluate_mot(DATASET_ROOT, byte_dir, sequences)
        bot_sum = evaluate_mot(DATASET_ROOT, bot_dir, sequences)
        lg_sum = evaluate_mot(DATASET_ROOT, lg_dir, sequences)
        
        if byte_sum is not None and bot_sum is not None and lg_sum is not None:
            for m_name, m_sum in [("ByteTrack", byte_sum), ("BoTSORT", bot_sum), ("LightGlue (Proposed)", lg_sum)]:
                stats.append({
                    "Method": m_name,
                    "FPS": fps_simulated,
                    "MOTA": m_sum.loc['OVERALL']['mota'] * 100,
                    "IDF1": m_sum.loc['OVERALL']['idf1'] * 100,
                    "IDSW": m_sum.loc['OVERALL']['num_switches']
                })

    # Summary
    df = pd.DataFrame(stats)
    df = df.sort_values(['FPS', 'Method'], ascending=[False, True])
    print("\n--- RESUMO PARA O ARTIGO (GOT-10k / SOT-to-MOT Mode) ---")
    print(df.to_string(index=False))
    
    latex_file = os.path.join(RESULTS_ROOT, "got10k_table.tex")
    with open(latex_file, 'w') as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{GOT-10k Tracking robustness at variable frame rates.}\n")
        f.write("\\begin{tabular}{l|c|ccc}\n\\hline\n")
        f.write("Method & FPS & MOTA $\\uparrow$ & IDF1 $\\uparrow$ & IDSW $\\downarrow$ \\\\ \\hline\n")
        for _, row in df.iterrows():
            f.write(f"{row['Method']} & {row['FPS']} & {row['MOTA']:.1f} & {row['IDF1']:.1f} & {row['IDSW']} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}")
    print(f"\nLaTeX table saved to {latex_file}")

if __name__ == "__main__":
    main()
