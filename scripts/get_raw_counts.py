import pandas as pd
import motmetrics as mm
import os

def main():
    gt_dir = '/mnt/hd2/ArtigoLightglue/Datasets_externos/gopro_benchmark'
    res_root = '/home/servidor/ArtigoLightglue/mass_results/gopro'
    mh = mm.metrics.create()
    sequences = sorted([d for d in os.listdir(gt_dir) if os.path.isdir(os.path.join(gt_dir, d))])
    metrics = ['mota', 'num_switches', 'num_misses', 'num_false_positives', 'num_objects', 'idf1']
    
    results = []
    
    for s in [30, 6, 1]:
        fps = 30 // s
        for m in ['bytetrack', 'lightglue']:
            res_dir = os.path.join(res_root, f'{m}_s{s}')
            accs = []
            names = []
            for seq in sequences:
                gtf = os.path.join(gt_dir, seq, 'gt', 'gt.txt')
                resf = os.path.join(res_dir, f'{seq}.txt')
                if os.path.exists(resf):
                    gt = mm.io.loadtxt(gtf, fmt='mot15-2D')
                    ts = mm.io.loadtxt(resf, fmt='mot15-2D')
                    accs.append(mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5))
                    names.append(seq)
            
            if not accs: continue
            
            summary = mh.compute_many(accs, metrics=metrics, names=names, generate_overall=True)
            row = summary.loc['OVERALL']
            
            results.append({
                "FPS": fps,
                "Method": m.upper(),
                "Missed (FN)": int(row.num_misses),
                "ID Switches": int(row.num_switches),
                "Total Objects": int(row.num_objects),
                "MOTA (%)": row.mota * 100,
                "IDF1 (%)": row.idf1 * 100
            })
            
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
