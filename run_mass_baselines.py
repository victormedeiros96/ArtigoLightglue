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

# Ultralytics nativos (usam run_single_track.py com .yaml)
TRACKERS_ULT = ["botsort", "bytetrack"]
# boxmot (usam run_single_track_boxmot.py)
TRACKERS_BOXMOT = ["strongsort", "ocsort", "deepocsort", "hybridsort", "boosttrack"]
STRIDES = {"30fps": 1, "5fps": 6, "1fps": 30}

def run_baselines():
    all_stats = []
    
    for v_path in VIDEOS:
        if not os.path.exists(v_path): continue
        v_name = os.path.basename(v_path).split('.')[0]
        
        all_trackers = [(t, "ultralytics") for t in TRACKERS_ULT] + \
                       [(t, "boxmot") for t in TRACKERS_BOXMOT]
        
        for t_name, source in all_trackers:
            for fps_label, stride in STRIDES.items():
                name_key = f"{v_name}_{t_name}_{fps_label}"
                print(f"Running Baseline: {name_key}...")
                
                if source == "ultralytics":
                    cmd = [
                        "python3", "/home/servidor/ArtigoLightglue/run_single_track.py",
                        "--video", v_path,
                        "--model", "/home/servidor/ArtigoLightglue/detector_placa.pt",
                        "--tracker", f"/home/servidor/ArtigoLightglue/{t_name}.yaml",
                        "--stride", str(stride),
                        "--name", name_key,
                        "--device", "0"
                    ]
                else:  # boxmot
                    cmd = [
                        "python3", "/home/servidor/ArtigoLightglue/run_single_track_boxmot.py",
                        "--video", v_path,
                        "--model", "/home/servidor/ArtigoLightglue/detector_placa.pt",
                        "--tracker", t_name,
                        "--stride", str(stride),
                        "--name", name_key,
                        "--device", "0"
                    ]
                subprocess.run(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
                
                # Ler o JSON gerado pelo run_single_track para pegar stats
                json_path = f"/home/servidor/ArtigoLightglue/tracking_results/{name_key}.json"
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    uids = len(set(d['id'] for d in data))
                    all_stats.append({
                        "Video": v_name, "Metodologia": t_name, "FPS": fps_label,
                        "Unique_IDs": uids, "Sightings": len(data)
                    })
                    # Mover para a pasta de massa
                    os.rename(json_path, os.path.join(JSON_DIR, f"{name_key}.json"))
                    
    df = pd.DataFrame(all_stats)
    df.to_csv(os.path.join(OUTPUT_ROOT, "mass_baselines_report.csv"), index=False)
    print("Baselines concluídos.")

if __name__ == "__main__":
    run_baselines()
