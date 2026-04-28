import os
import subprocess
import pandas as pd
import json

VIDEOS = [
    "/home/servidor/ArtigoLightglue/proxies/GX010084.MP4",
    "/home/servidor/ArtigoLightglue/proxies/GX010083.MP4",
    "/home/servidor/ArtigoLightglue/proxies/GX010069.MP4",
    "/home/servidor/ArtigoLightglue/proxies/GX010076.MP4",
    "/home/servidor/ArtigoLightglue/proxies/GX010086.MP4",
    "/home/servidor/ArtigoLightglue/proxies/GX010067.MP4",
    "/home/servidor/ArtigoLightglue/proxies/GX010080.MP4",
    "/home/servidor/ArtigoLightglue/proxies/GX010081.MP4",
    "/home/servidor/ArtigoLightglue/proxies/GX010085.MP4"
]

OUTPUT_ROOT = "/home/servidor/ArtigoLightglue/mass_results"
JSON_DIR = os.path.join(OUTPUT_ROOT, "jsons")
os.makedirs(JSON_DIR, exist_ok=True)

TRACKERS = ["botsort", "bytetrack"]
STRIDES = {"30fps": 1, "5fps": 6, "1fps": 30}

def run_baselines():
    all_stats = []
    
    for v_path in VIDEOS:
        if not os.path.exists(v_path): continue
        v_name = os.path.basename(v_path).split('.')[0]
        
        for t_name in TRACKERS:
            for fps_label, stride in STRIDES.items():
                name_key = f"{v_name}_{t_name}_{fps_label}"
                print(f"Running Baseline: {name_key}...")
                
                cmd = [
                    "python3", "/home/servidor/ArtigoLightglue/benchmarks/run_single_baseline.py",
                    "--video", v_path,
                    "--model", "/home/servidor/ArtigoLightglue/models/detector_placa.pt",
                    "--tracker", f"/home/servidor/ArtigoLightglue/configs/{t_name}.yaml",
                    "--stride", str(stride),
                    "--name", name_key,
                    "--device", "0"
                ]
                subprocess.run(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
                
                # Ler o JSON gerado pelo run_single_track para pegar stats
                json_path = f"/home/servidor/ArtigoLightglue/mass_results/jsons/{name_key}.json"
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    uids = len(set(d['id'] for d in data))
                    all_stats.append({
                        "Video": v_name, "Metodologia": t_name, "FPS": fps_label,
                        "Unique_IDs": uids, "Sightings": len(data)
                    })
                    
    df = pd.DataFrame(all_stats)
    df.to_csv(os.path.join(OUTPUT_ROOT, "mass_baselines_report.csv"), index=False)
    print("Baselines concluídos.")

if __name__ == "__main__":
    run_baselines()
