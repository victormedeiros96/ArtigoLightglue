import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import cv2

# Adicionar caminhos para importar o tracker
sys.path.append('/home/servidor/ArtigoLightglue')
from trackers.lightglue_tracker import LightGlueTracker
from ultralytics import YOLO

MODEL_PATH = "/home/servidor/ArtigoLightglue/detector_placa.pt"
VIDEO_PATH = "/home/servidor/ArtigoLightglue/proxies/GX010069.MP4"
TARGET_IDS = 4 

def run_with_params(params, video_path, fps_list=[30, 5, 1]):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}
    
    cache_file = "/tmp/det_cache_69.json"
    if not os.path.exists(cache_file):
        print("Gerando cache de detecções YOLO...")
        model = YOLO(MODEL_PATH)
        cap = cv2.VideoCapture(video_path)
        all_dets = []
        f_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or f_idx >= 3600: break
            res = model.predict(frame, imgsz=1280, verbose=False, classes=list(range(11)))[0]
            if res.boxes is not None:
                boxes = res.boxes.xyxy.cpu().numpy().tolist()
                clss = res.boxes.cls.cpu().numpy().tolist()
                all_dets.append({"f": f_idx, "b": boxes, "c": clss})
            f_idx += 1
        cap.release()
        with open(cache_file, 'w') as f: json.dump(all_dets, f)
    
    with open(cache_file, 'r') as f: all_dets = json.load(f)
    det_map = {d['f']: d for d in all_dets}

    cap = cv2.VideoCapture(video_path)
    frames_cache = {} 
    
    for fps in fps_list:
        stride = 30 // fps
        tracker = LightGlueTracker(
            device=device,
            min_matches_short=params['th_short'],
            min_matches_global=params['th_global'],
            accept_th=params['accept_th'],
            motion_weight=params['motion_weight']
        )
        
        tracking_data = []
        for f_idx in range(0, 3600, stride):
            if f_idx not in det_map: continue
            
            if f_idx not in frames_cache:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret: break
                frames_cache[f_idx] = frame
            
            frame = frames_cache[f_idx]
            det = det_map[f_idx]
            crops = []
            for box in det['b']:
                x1, y1, x2, y2 = map(int, box)
                pad = 15
                H, W = frame.shape[:2]
                crop = frame[max(0,y1-pad):min(H,y2+pad), max(0,x1-pad):min(W,x2+pad)]
                crops.append(crop)
            
            active = tracker.update(np.array(det['b']), crops, det['c'], f_idx, stride=stride)
            tracking_data.extend(active)
            
        # Filtro de Confiança (mínimo de frames visto)
        id_counts = {}
        for d in tracking_data: id_counts[d['id']] = id_counts.get(d['id'], 0) + 1
        uids = len(set(d['id'] for d in tracking_data if id_counts[d['id']] >= (3 if fps == 30 else 1)))
        results[fps] = uids
        
    cap.release()
    return results

def fitness(params):
    res = run_with_params(params, VIDEO_PATH)
    # Loss: Erro vs Baseline + Variância entre FPS
    error_target = abs(res[30] - TARGET_IDS)
    variance = abs(res[5] - res[30]) + abs(res[1] - res[30])
    return error_target + (2.0 * variance) # Peso dobrado na variância

if __name__ == "__main__":
    best_params = None
    min_loss = float('inf')
    
    print("Iniciando Autotuner de Pesos...")
    # Espaço de busca (Grid Search)
    for th_s in [8, 12, 16]:
        for th_g in [12, 18, 24]:
            for a_th in [0.95, 1.2, 1.5]:
                for m_w in [0.3, 0.5]:
                    params = {'th_short': th_s, 'th_global': th_g, 'accept_th': a_th, 'motion_weight': m_w}
                    loss = fitness(params)
                    print(f"Params: {params} -> Loss: {loss}")
                    if loss < min_loss:
                        min_loss, best_params = loss, params
                    
    print(f"\nMelhor Configuração Final: {best_params} com Loss: {min_loss}")
