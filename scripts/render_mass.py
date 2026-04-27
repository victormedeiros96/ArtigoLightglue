import cv2
import json
import os
import numpy as np
from collections import defaultdict, deque
from tqdm import tqdm

def draw_premium(frame, tracks, trails, last_known_boxes, last_seen_frame, f_idx, colors, info, max_age):
    for trk in tracks:
        tid = trk['id']
        last_known_boxes[tid] = trk['box']
        last_seen_frame[tid] = f_idx
        box = trk['box']
        center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
        trails[tid].append(center)

    expired = [tid for tid, last_f in last_seen_frame.items() if (f_idx - last_f) > max_age]
    for tid in expired:
        del last_known_boxes[tid]
        del last_seen_frame[tid]

    for tid, box in last_known_boxes.items():
        x1, y1, x2, y2 = map(int, box)
        color = colors[tid % 1000]
        if len(trails[tid]) > 1:
            for i in range(1, len(trails[tid])):
                cv2.line(frame, trails[tid][i-1], trails[tid][i], color, 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"ID: {tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

    cv2.rectangle(frame, (0, 0), (700, 60), (0, 0, 0), -1)
    cv2.putText(frame, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return frame

def main():
    root = "/home/servidor/ArtigoLightglue/mass_results"
    proxy_dir = "/home/servidor/ArtigoLightglue/proxies" # USAR PROXIES 1080p
    json_dir = os.path.join(root, "jsons")
    video_out_dir = os.path.join(root, "videos")
    os.makedirs(video_out_dir, exist_ok=True)
    
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json") and "_lg_" in f]
    colors = np.random.randint(0, 255, (1000, 3)).tolist()

    for j_name in sorted(json_files):
        out_path = os.path.join(video_out_dir, j_name.replace(".json", ".mp4"))
        
        # PULAR SE JÁ EXISTIR
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000000:
            print(f"Skipping {j_name} (Already rendered)")
            continue

        v_base = j_name.split('_')[0]
        v_path = os.path.join(proxy_dir, f"{v_base}.MP4")
        if not os.path.exists(v_path):
            print(f"Proxy missing for {v_base}")
            continue
        
        j_path = os.path.join(json_dir, j_name)
        with open(j_path, 'r') as f: data = json.load(f)
        data_by_frame = defaultdict(list)
        for d in data: data_by_frame[d['frame']].append(d)
        
        fps_val = int(j_name.split('_')[-1].replace('fps.json', ''))
        stride = 30 // fps_val
        max_age = max(5, int(stride * 1.5))
        
        cap = cv2.VideoCapture(v_path)
        fps = 30
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1920, 1080))
        
        trails = defaultdict(lambda: deque(maxlen=50))
        last_known_boxes, last_seen_frame = {}, {}
        
        print(f"Rendering (1080p): {j_name}...")
        for f_idx in tqdm(range(3600)):
            ret, frame = cap.read()
            if not ret: break
            
            # Já está em 1080p (proxy)
            hits = data_by_frame[f_idx]
            info = f"{v_base} | {fps_val}FPS | LIGHTGLUE"
            frame_res = draw_premium(frame, hits, trails, last_known_boxes, last_seen_frame, f_idx, colors, info, max_age)
            out.write(frame_res)
            
        cap.release()
        out.release()

if __name__ == "__main__":
    main()
