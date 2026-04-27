import os
import sys

# Adicionar o diretório de scripts ao path para importar funções do render_mass
sys.path.append('/home/servidor/ArtigoLightglue/scripts')
import render_mass
import cv2
import json
from collections import defaultdict, deque
from tqdm import tqdm
import numpy as np

def render_target(target_video="GX010069"):
    root = "/home/servidor/ArtigoLightglue/mass_results"
    proxy_dir = "/home/servidor/ArtigoLightglue/proxies"
    json_dir = os.path.join(root, "jsons")
    video_out_dir = os.path.join(root, "videos")
    
    # Encontrar apenas os JSONs do vídeo alvo
    json_files = [f for f in os.listdir(json_dir) if f.startswith(target_video) and "_lg_" in f]
    colors = np.random.randint(0, 255, (1000, 3)).tolist()

    for j_name in sorted(json_files):
        v_path = os.path.join(proxy_dir, f"{target_video}.MP4")
        out_path = os.path.join(video_out_dir, j_name.replace(".json", ".mp4"))
        
        # Recriar sempre para garantir que pegamos as mudanças de filtragem
        j_path = os.path.join(json_dir, j_name)
        with open(j_path, 'r') as f: data = json.load(f)
        
        data_by_frame = defaultdict(list)
        for d in data: data_by_frame[d['frame']].append(d)
        
        fps_val = int(j_name.split('_')[-1].replace('fps.json', ''))
        stride = 30 // fps_val
        max_age = max(5, int(stride * 1.5))
        
        cap = cv2.VideoCapture(v_path)
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))
        
        trails = defaultdict(lambda: deque(maxlen=50))
        last_known_boxes, last_seen_frame = {}, {}
        
        print(f"Renderizando Visualização: {j_name}...")
        for f_idx in tqdm(range(3600)):
            ret, frame = cap.read()
            if not ret: break
            
            hits = data_by_frame[f_idx]
            info = f"{target_video} | {fps_val}FPS | LIGHTGLUE"
            frame_res = render_mass.draw_premium(frame, hits, trails, last_known_boxes, last_seen_frame, f_idx, colors, info, max_age)
            out.write(frame_res)
            
        cap.release()
        out.release()
        print(f"Vídeo salvo em: {out_path}")

if __name__ == "__main__":
    render_target("GX010069")
