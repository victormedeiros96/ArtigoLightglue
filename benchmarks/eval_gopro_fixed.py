import os
import motmetrics as mm
import pandas as pd

def evaluate_mot(gt_dir, res_dir, sequences):
    mh = mm.metrics.create()
    accs = []
    names = []
    
    for seq in sequences:
        gt_file = os.path.join(gt_dir, seq, 'gt', 'gt.txt')
        res_file = os.path.join(res_dir, f"{seq}.txt")
        
        if not os.path.exists(gt_file) or not os.path.exists(res_file) or os.path.getsize(res_file) == 0:
            print(f"Skipping {seq}: GT or Result missing/empty")
            continue
            
        gt = mm.io.loadtxt(gt_file, fmt='mot15-2D', min_confidence=1)
        ts = mm.io.loadtxt(res_file, fmt='mot15-2D')
        
        acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
        accs.append(acc)
        names.append(seq)
        
    if not accs:
        return None
    summary = mh.compute_many(accs, metrics=['mota', 'num_switches', 'idf1'], names=names, generate_overall=True)
    return summary

DATASET_ROOT = "/mnt/hd2/ArtigoLightglue/Datasets_externos/gopro_benchmark"
RESULTS_ROOT = "/home/servidor/ArtigoLightglue/mass_results/gopro"
sequences = ["2049", "2050", "2053", "2055", "2059"]

for stride in [15, 6]:
    fps = 30 // stride
    print(f"\n=== Results for {fps} FPS (Stride {stride}) ===")
    
    # Eval LightGlue (New CMC)
    lg_dir = os.path.join(RESULTS_ROOT, f'lightglue_s{stride}')
    lg_sum = evaluate_mot(DATASET_ROOT, lg_dir, sequences)
    if lg_sum is not None:
        print("\nLightGlue (with CMC):")
        print(lg_sum) # All sequences
        
    # Eval BoTSORT (Baseline)
    bot_dir = os.path.join(RESULTS_ROOT, f'botsort_s{stride}')
    bot_sum = evaluate_mot(DATASET_ROOT, bot_dir, sequences)
    if bot_sum is not None:
        print("\nBoTSORT (Baseline):")
        print(bot_sum.tail(1)) # Overall
