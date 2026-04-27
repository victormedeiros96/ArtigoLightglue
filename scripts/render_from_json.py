import cv2
import json
import os
import numpy as np
from collections import defaultdict, deque
from tqdm import tqdm

def draw_premium(frame, tracks, trails, last_known_boxes, last_seen_frame, f_idx, colors, info, max_age):
    # 1. Update with new detections
    for trk in tracks:
        tid = trk['id']
        last_known_boxes[tid] = trk['box']
        last_seen_frame[tid] = f_idx
        
        # Trail update
        box = trk['box']
        center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
        trails[tid].append(center)

    # 2. Cleanup expired tracks
    expired = [tid for tid, last_f in last_seen_frame.items() if (f_idx - last_f) > max_age]
    for tid in expired:
        del last_known_boxes[tid]
        del last_seen_frame[tid]
        # We keep the trail but maybe we should let it fade? 
        # For now let's keep it for visual history as requested

    # 3. Draw active persistent tracks
    for tid, box in last_known_boxes.items():
        x1, y1, x2, y2 = map(int, box)
        color = colors[tid % 1000]
        
        # Trail
        if len(trails[tid]) > 1:
            for i in range(1, len(trails[tid])):
                cv2.line(frame, trails[tid][i-1], trails[tid][i], color, 2)
        
        # Neon Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"ID: {tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

    # Header
    cv2.rectangle(frame, (0, 0), (600, 60), (0, 0, 0), -1)
    cv2.putText(frame, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return frame

def main():
    video_path = "/home/servidor/ArtigoLightglue/segment_1080p_120s.mp4"
    results_dirs = ["/home/servidor/ArtigoLightglue/tracking_results", "/home/servidor/ArtigoLightglue/tracking_results_naive"]
    output_root = "/home/servidor/ArtigoLightglue/results/videos_final"
    os.makedirs(output_root, exist_ok=True)
    
    json_files = []
    for d in results_dirs:
        if os.path.exists(d):
            json_files.extend([os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json")])
    
    colors = np.random.randint(0, 255, (1000, 3)).tolist()

    for j_path in json_files:
        name = os.path.basename(j_path).replace(".json", "")
        out_path = os.path.join(output_root, f"{name}.mp4")
        
        tokens = name.split('_')
        fps_str = tokens[-1] # e.g. "30fps"
        fps_val = int(fps_str.replace('fps', ''))
        stride = 30 // fps_val
        # Define o tempo de vida da caixa na tela: 1.2x o intervalo de amostragem
        # Mas no mínimo 5 frames para não piscar no 30fps
        max_age = max(5, int(stride * 1.2))
        
        with open(j_path, 'r') as f:
            data = json.load(f)
        
        data_by_frame = defaultdict(list)
        for d in data:
            data_by_frame[d['frame']].append(d)
        
        cap = cv2.VideoCapture(video_path)
        w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        trails = defaultdict(lambda: deque(maxlen=50))
        last_known_boxes = {}
        last_seen_frame = {}
        
        print(f"Rendering (TTL={max_age}): {name}...")
        for f_idx in tqdm(range(3600)):
            ret, frame = cap.read()
            if not ret: break
            
            current_hits = data_by_frame[f_idx]
            info = f"{name.upper().replace('_', ' ')}"
            frame = draw_premium(frame, current_hits, trails, last_known_boxes, last_seen_frame, f_idx, colors, info, max_age)
            out.write(frame)
            
        cap.release()
        out.release()

if __name__ == "__main__":
    main()
