import os
import cv2
import json
import torch
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.lightglue_tracker import LightGlueTracker

# Configurações de Massa
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
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "jsons"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "videos"), exist_ok=True)

MODEL_PATH = "/home/servidor/ArtigoLightglue/models/detector_placa.pt"

def run_experiment_on_video(video_path):
    video_name = os.path.basename(video_path).split('.')[0]
    model = YOLO(MODEL_PATH)
    
    results_summary = []
    
    # Diferentes taxas de amostragem
    for fps_target in [30, 5, 1]:
        stride = 30 // fps_target
        cap = cv2.VideoCapture(video_path)
        
        # Parâmetros do tracker adaptados para o artigo
        tracker = LightGlueTracker(
            device='cuda',
            accept_th=1.2,    # Rigoroso para evitar matches falsos
            motion_weight=0.3 # Peso equilibrado para movimento
        )
        
        tracking_data = []
        f_idx = 0
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        pbar = tqdm(total=total_f, desc=f"{video_name}_{fps_target}fps")
        
        print(f"Iniciando loop de frames para {video_name} @ {fps_target}fps...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if f_idx % stride == 0:
                det_res = model.predict(frame, imgsz=1280, verbose=False, classes=list(range(11)))[0]
                
                det_bboxes, det_classes = [], []
                if det_res.boxes is not None:
                    det_bboxes = det_res.boxes.xyxy.cpu().numpy()
                    det_classes = det_res.boxes.cls.cpu().numpy().astype(int)
                
                active_tracks = tracker.update(det_bboxes, frame, det_classes, f_idx, stride=stride)
                
                for trk in active_tracks:
                    tracking_data.append({
                        "frame": f_idx,
                        "id": trk['id'],
                        "box": trk['bbox'].tolist(),
                        "cls": int(trk['cls'])
                    })
            
            f_idx += 1
            pbar.update(1)
            if f_idx >= 3600: break
            
        pbar.close()
        cap.release()
        
        id_counts = {}
        for d in tracking_data:
            id_counts[d['id']] = id_counts.get(d['id'], 0) + 1
        
        # Filtro de estabilidade mínima (0.3s)
        import math
        min_presenca = max(1, math.ceil(0.3 * fps_target))
        clean_data = [d for d in tracking_data if id_counts[d['id']] >= min_presenca]
        
        json_path = os.path.join(OUTPUT_ROOT, "jsons", f"{video_name}_lg_{fps_target}fps.json")
        with open(json_path, 'w') as f:
            json.dump(clean_data, f)
            
        unique_ids = len(set(d['id'] for d in clean_data))
        total_sightings = len(clean_data)
        
        results_summary.append({
            "video": video_name,
            "fps": fps_target,
            "ids": unique_ids,
            "sightings": total_sightings
        })
        
    return results_summary

def main():
    all_stats = []
    for v_path in VIDEOS:
        stats = run_experiment_on_video(v_path)
        all_stats.extend(stats)
        
    df = pd.DataFrame(all_stats)
    df.to_csv(os.path.join(OUTPUT_ROOT, "mass_experiment_report.csv"), index=False)
    print("\n--- EXPERIMENTO DE MASSA CONCLUÍDO ---")
    print(f"Relatório salvo em: {os.path.join(OUTPUT_ROOT, 'mass_experiment_report.csv')}")

if __name__ == "__main__":
    main()
