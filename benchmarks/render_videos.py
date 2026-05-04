import os
import cv2
import json
from tqdm import tqdm

ROOT = "/home/servidor/ArtigoLightglue"
JSON_DIR = f"{ROOT}/mass_results/jsons"
OUT_DIR = f"{ROOT}/mass_results/videos"
os.makedirs(OUT_DIR, exist_ok=True)

# Lista de vídeos originais
VIDEOS = [
    f"{ROOT}/proxies/GX010083.MP4",
    f"{ROOT}/proxies/GX010084.MP4",
    f"{ROOT}/proxies/GX010069.MP4",
    f"{ROOT}/proxies/GX010076.MP4",
    f"{ROOT}/proxies/GX010086.MP4",
    f"{ROOT}/proxies/GX010067.MP4",
    f"{ROOT}/proxies/GX010080.MP4",
    f"{ROOT}/proxies/GX010081.MP4",
    f"{ROOT}/proxies/GX010085.MP4"
]

def render_json_on_video(video_path, json_path, output_path, fps_target):
    if not os.path.exists(json_path):
        print(f"Pulo: {json_path} não encontrado.")
        return

    cap = cv2.VideoCapture(video_path)
    # 30fps fixo nos proxies
    total_f = 3600 # Limite do experimento
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 15, (w, h)) # 15 fps para fluidez de review
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Indexar dados por frame
    frame_map = {}
    for d in data:
        f = d['frame']
        if f not in frame_map: frame_map[f] = []
        frame_map[f].append(d)
        
    pbar = tqdm(total=total_f, desc=os.path.basename(output_path))
    
    f_idx = 0
    stride = 30 // fps_target
    
    while f_idx < total_f:
        ret, frame = cap.read()
        if not ret: break
        
        # Só renderizamos os frames que o tracker viu (no stride)
        if f_idx % stride == 0:
            if f_idx in frame_map:
                for obj in frame_map[f_idx]:
                    box = obj['box']
                    tid = obj['id']
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3)
                    cv2.putText(frame, f"ID {tid}", (int(box[0]), int(box[1]-10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # Adicionar info de FPS no topo
            cv2.putText(frame, f"Original FPS Target: {fps_target}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            out.write(frame)
            
        f_idx += 1
        pbar.update(1)
        
    cap.release()
    out.release()
    pbar.close()

def main():
    for v_path in VIDEOS:
        v_id = os.path.basename(v_path).split('.')[0]
        for fps in [30, 5]:
            json_file = f"{JSON_DIR}/{v_id}_lg_{fps}fps.json"
            out_file = f"{OUT_DIR}/{v_id}_COMPARATIVO_{fps}fps.mp4"
            if not os.path.exists(out_file): # Evitar re-renderizar o que já está pronto
                render_json_on_video(v_path, json_file, out_file, fps)

if __name__ == "__main__":
    main()
